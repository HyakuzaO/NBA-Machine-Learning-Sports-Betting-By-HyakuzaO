# NBA Sports Betting Using Machine Learning 🏀
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

A machine learning AI used to predict the winners and under/overs of NBA games. Takes all team data from the 2021-22 season to current season, matched with odds of those games, using a neural network to predict winning bets for today's games. Achieves ~69% auc on money lines and ~55% on under/overs. Outputs expected value for teams money lines to provide better insight. The fraction of your bankroll to bet based on the Kelly Criterion is also outputted. Note that a popular, less risky approach is to bet 50% of the stake recommended by the Kelly Criterion.
## Packages Used

# NBA Machine Learning Sports Betting

A predictive system for NBA games built with **Python 3.11**, using advanced machine learning models to estimate win probabilities and totals (Over/Under), compute **Expected Value (EV)**, and apply the **Kelly Criterion** for bankroll management.  
Includes **Optuna** for hyperparameter optimization.

---

## 🧠 Tech Stack

- **TensorFlow** – Deep learning and neural network training  
- **XGBoost** – Gradient boosting framework  
- **Optuna** – Hyperparameter optimization (HPO)  
- **NumPy** – Scientific computing  
- **Pandas** – Data manipulation and analysis  
- **scikit-learn** – Metrics, preprocessing, and model evaluation  
- **tqdm** – Progress bars  
- **colorama** – Colored console output  
- **requests** – HTTP data fetching  

---

## ⚙️ Requirements

- Python **3.11**  
- Install dependencies:

```bash
pip install -r requirements.txt


Make sure all packages above are installed.

```bash
$ git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
$ cd NBA-Machine-Learning-Sports-Betting
$ pip3 install -r requirements.txt
$ python3 main.py -xgb -odds=fanduel
```

Odds data will be automatically fetched from sbrodds if the -odds option is provided with a sportsbook.  Options include: fanduel, draftkings, betmgm, pointsbet, caesars, wynn, bet_rivers_ny

If `-odds` is not given, enter the under/over and odds for today's games manually after starting the script.

Optionally, you can add '-kc' as a command line argument to see the recommended fraction of your bankroll to wager based on the model's edge

## Flask Web App
<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/Flask-App.png" width="922" height="580" />

This repo also includes a small Flask application to help view the data from this tool in the browser.  To run it:
```
cd Flask
flask --debug run
```

## Getting new data and training models
```
# Create dataset with the latest data for 2023-24 season
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO
```

## Contributing

All contributions welcomed and encouraged.
