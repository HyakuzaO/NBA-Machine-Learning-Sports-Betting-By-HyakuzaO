import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

dataset = "dataset_2021-25_new"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

OU = data['OU-Cover']
total = data['OU']

d0 = pd.to_datetime(data['Date'], errors='coerce') if 'Date' in data.columns else pd.Series(pd.NaT, index=data.index)
d1 = pd.to_datetime(data['Date.1'], errors='coerce') if 'Date.1' in data.columns else pd.Series(pd.NaT, index=data.index)
d = d0.fillna(d1)
r = d.rank(method="first")
r = (r - r.min()) / (r.max() - r.min() + 1e-12)
n_steps = 20
step_idx = np.floor(r * n_steps).astype(int)
step_idx = np.clip(step_idx, 0, n_steps)
scale = np.linspace(0.6, 2.2, n_steps + 1)
weights = scale[step_idx.values]

data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)
data['OU'] = np.asarray(total)

data = data.values
data = data.astype(float)
acc_results = []

for x in tqdm(range(100)):
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(data, OU, weights, test_size=.1)

    train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
    test = xgb.DMatrix(x_test, label=y_test)

    param = {
        'max_depth': 20,
        'eta': 0.05,
        'objective': 'multi:softprob',
        'num_class': 3
    }
    epochs = 750

    model = xgb.train(param, train, epochs)
    predictions = model.predict(test)
    y = []

    for z in predictions:
        y.append(np.argmax(z))

    acc = round(accuracy_score(y_test, y) * 100, 1)
    print(f"{acc}%")
    acc_results.append(acc)
    if acc == max(acc_results):
        model.save_model('../../Models/XGBoost_{}%_UO-9.json'.format(acc))
