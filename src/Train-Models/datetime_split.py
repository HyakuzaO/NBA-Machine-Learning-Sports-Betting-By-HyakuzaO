from __future__ import annotations
from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

IndexLike = Union[pd.Index, np.ndarray, Iterable[int]]

@dataclass
class DateTimeSeriesSplitByDate:
    n_splits: int = 5
    date_column: Optional[str] = None
    test_size: Union[int, float] = 1
    train_size: Optional[Union[int, float]] = None
    gap: int = 0
    embargo: int = 0
    expanding: bool = True
    ensure_full_test: bool = True
    sort_stable: bool = True

    def _validate_and_prepare(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        dates: Optional[Iterable] = None,
    ) -> Tuple[pd.Index, pd.Series]:
        if isinstance(X, (pd.Series, pd.DataFrame)):
            idx = X.index
        else:
            idx = pd.RangeIndex(0, len(X))

        if dates is None:
            if self.date_column is None:
                raise ValueError("Provide `date_column` in the splitter or pass `dates` to split().")
            if not isinstance(X, (pd.DataFrame,)):
                raise ValueError("When using `date_column`, X must be a pandas DataFrame.")
            dates_ser = pd.to_datetime(X[self.date_column])
        else:
            dates_ser = pd.to_datetime(pd.Series(dates, index=idx))

        order = np.lexsort((idx.values, dates_ser.values)) if self.sort_stable else np.argsort(dates_ser.values)
        sorted_idx = idx.values[order]
        sorted_dates = dates_ser.values[order]
        sorted_dates_ser = pd.Series(sorted_dates, index=sorted_idx)
        return pd.Index(sorted_idx), sorted_dates_ser

    @staticmethod
    def _resolve_size(size: Union[int, float, None], n_unique: int, name: str) -> Optional[int]:
        if size is None:
            return None
        if isinstance(size, float):
            if not (0 < size <= 1):
                raise ValueError(f"{name} as float must be in (0,1], got {size}.")
            k = int(round(size * n_unique))
            return max(1, k)
        if isinstance(size, int):
            if size <= 0:
                raise ValueError(f"{name} must be > 0, got {size}.")
            return size
        raise TypeError(f"{name} must be int, float or None.")

    def split(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Iterable] = None,
        dates: Optional[Iterable] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        sorted_idx, sorted_dates_ser = self._validate_and_prepare(X, dates)
        unique_dates = pd.Index(pd.unique(sorted_dates_ser.dt.normalize()))
        n_unique = len(unique_dates)
        if n_unique < 2:
            raise ValueError("Not enough unique dates to split.")

        test_k = self._resolve_size(self.test_size, n_unique, "test_size")
        train_k = self._resolve_size(self.train_size, n_unique, "train_size")
        gap_k = int(self.gap)
        embargo_k = int(self.embargo)

        if train_k is None and not self.expanding:
            raise ValueError("expanding=False requires an explicit train_size.")

        max_start = n_unique - test_k
        if max_start <= 0:
            raise ValueError("test_size too large for the number of unique dates.")

        anchors = list(range(0, max_start + 1, test_k))
        if len(anchors) > self.n_splits:
            anchors = anchors[-self.n_splits:]

        for a in anchors:
            test_start_pos = a
            test_end_pos = a + test_k
            gap_start_pos = max(0, test_start_pos - gap_k)

            if train_k is None:
                train_end_pos = gap_start_pos
                train_start_pos = 0
            else:
                train_end_pos = gap_start_pos
                train_start_pos = max(0, train_end_pos - train_k)

            if self.ensure_full_test and test_end_pos > n_unique:
                continue

            train_dates = unique_dates[train_start_pos:train_end_pos]
            test_dates = unique_dates[test_start_pos:test_end_pos]

            if len(test_dates) == 0 or len(train_dates) == 0:
                continue

            train_mask = sorted_dates_ser.dt.normalize().isin(train_dates).values
            test_mask = sorted_dates_ser.dt.normalize().isin(test_dates).values

            if embargo_k > 0:
                embargo_start_pos = min(n_unique, test_end_pos)
                embargo_end_pos = min(n_unique, test_end_pos + embargo_k)
                embargo_dates = set(unique_dates[embargo_start_pos:embargo_end_pos])
                if len(embargo_dates) > 0:
                    embargo_mask = sorted_dates_ser.dt.normalize().isin(embargo_dates).values
                    train_mask = np.logical_and(train_mask, ~embargo_mask)

            train_index = sorted_idx[train_mask]
            test_index = sorted_idx[test_mask]

            if len(train_index) == 0 or len(test_index) == 0:
                continue

            yield (np.asarray(train_index), np.asarray(test_index))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


if __name__ == "__main__":
    rng = pd.date_range("2024-10-01", periods=120, freq="D")
    df = pd.DataFrame({
        "Date": np.repeat(rng, repeats=3),
        "feat1": np.random.randn(len(rng) * 3),
        "feat2": np.random.randn(len(rng) * 3),
        "y": (np.random.rand(len(rng) * 3) > 0.5).astype(int)
    })

    splitter = DateTimeSeriesSplitByDate(
        n_splits=4,
        date_column="Date",
        test_size=10,
        train_size=None,
        gap=1,
        embargo=1,
        expanding=True,
        ensure_full_test=True,
    )

    for i, (tr, te) in enumerate(splitter.split(df)):
        print(f"Fold {i+1}: train={len(tr)} | test={len(te)} | "
              f"train {df.loc[tr, 'Date'].min().date()}→{df.loc[tr, 'Date'].max().date()} | "
              f"test {df.loc[te, 'Date'].min().date()}→{df.loc[te, 'Date'].max().date()}")
