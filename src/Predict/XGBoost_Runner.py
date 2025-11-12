import copy
import numpy as np
import pandas as pd
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

THRESHOLD_ML = 0.594
THRESHOLD_OU = 0.5

init()
xgb_ml = xgb.Booster()
xgb_ml.load_model('Models/XGBoost_Models/XGBoost_0.712AUC_ML-4.json')
xgb_uo = xgb.Booster()
xgb_uo.load_model('Models/XGBoost_Models/XGBoost_56.8%_UO-9.json')

def _predict_proba_booster(booster, row):
    out = booster.predict(xgb.DMatrix(np.asarray([row], dtype=float)))
    out = np.asarray(out)
    if out.ndim == 1:
        p1 = float(out[0])
        p0 = 1.0 - p1
        return np.array([p0, p1], dtype=float)
    if out.shape[-1] == 1:
        p1 = float(out[0, 0])
        p0 = 1.0 - p1
        return np.array([p0, p1], dtype=float)
    return out[0]

def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    ml_predictions_array = []
    for row in data:
        ml_predictions_array.append(_predict_proba_booster(xgb_ml, row))

    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo, dtype=float)
    data_uo = frame_uo.values.astype(float)

    ou_predictions_array = []
    for row in data_uo:
        ou_predictions_array.append(_predict_proba_booster(xgb_uo, row))

    for idx, (home_team, away_team) in enumerate(games):
        p_away, p_home = float(ml_predictions_array[idx][0]), float(ml_predictions_array[idx][1])
        home_is_fav = p_home >= p_away
        fav_prob = p_home if home_is_fav else p_away
        p_under, p_over = float(ou_predictions_array[idx][0]), float(ou_predictions_array[idx][1])
        choose_over = p_over >= THRESHOLD_OU
        ou_label = 'OVER' if choose_over else 'UNDER'
        ou_prob = p_over if choose_over else p_under
        if home_is_fav:
            print(
                Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({fav_prob*100:.1f}%)" + Style.RESET_ALL +
                ' vs ' +
                Fore.RED + away_team + Style.RESET_ALL + ': ' +
                (Fore.BLUE if ou_label == 'OVER' else Fore.MAGENTA) + ou_label + Style.RESET_ALL + ' ' +
                str(todays_games_uo[idx]) + Fore.CYAN + f" ({ou_prob*100:.1f}%)" + Style.RESET_ALL
            )
        else:
            print(
                Fore.RED + home_team + Style.RESET_ALL + ' vs ' +
                Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({fav_prob*100:.1f}%)" + Style.RESET_ALL + ': ' +
                (Fore.BLUE if ou_label == 'OVER' else Fore.MAGENTA) + ou_label + Style.RESET_ALL + ' ' +
                str(todays_games_uo[idx]) + Fore.CYAN + f" ({ou_prob*100:.1f}%)" + Style.RESET_ALL
            )

    print("------------Expected Value & Kelly Criterion-----------" if kelly_criterion else "---------------------Expected Value--------------------")

    for idx, (home_team, away_team) in enumerate(games):
        p_away, p_home = float(ml_predictions_array[idx][0]), float(ml_predictions_array[idx][1])
        odd_home = home_team_odds[idx]
        odd_away = away_team_odds[idx]
        try:
            odd_home = float(str(odd_home).replace(',', '.')) if odd_home is not None else None
        except:
            odd_home = None
        try:
            odd_away = float(str(odd_away).replace(',', '.')) if odd_away is not None else None
        except:
            odd_away = None

        ev_home = Expected_Value.expected_value(p_home, odd_home) if (odd_home is not None) else 0
        ev_away = Expected_Value.expected_value(p_away, odd_away) if (odd_away is not None) else 0
        ev_home = float(ev_home) if ev_home is not None else 0.0
        ev_away = float(ev_away) if ev_away is not None else 0.0

        expected_value_colors = {
            'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
            'away_color': Fore.GREEN if ev_away > 0 else Fore.RED
        }

        bankroll_descriptor = ' Fraction of Bankroll: '
        bf_home = kc.calculate_kelly_criterion(odd_home, p_home) if (odd_home is not None) else 0
        bf_away = kc.calculate_kelly_criterion(odd_away, p_away) if (odd_away is not None) else 0
        bankroll_fraction_home = bankroll_descriptor + str(bf_home) + '%'
        bankroll_fraction_away = bankroll_descriptor + str(bf_away) + '%'

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (bankroll_fraction_away if kelly_criterion else ''))

    deinit()

