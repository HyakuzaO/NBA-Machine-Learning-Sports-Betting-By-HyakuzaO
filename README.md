# 🏀 NBA Sports Betting Using Machine Learning  
**Author:** [HyakuzaO](https://github.com/HyakuzaO)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-red?logo=xgboost)
![Optuna](https://img.shields.io/badge/Optuna-HPO-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Stars](https://img.shields.io/github/stars/HyakuzaO/NBA-Machine-Learning-Sports-Betting-By-HyakuzaO?style=social)

<img src="https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/blob/master/Screenshots/output.png" width="1010" height="292" />

---

## 🧩 Overview

A **machine learning AI** built to predict the **winners and totals (Over/Under)** of NBA games.  
It uses all team data from the **2021–22 season to the current season**, matched with **sportsbook odds**, and applies advanced ML models (Neural Networks & XGBoost) to forecast **winning bets for today’s games**.

The model achieves approximately:
- 🧠 **~69% AUC** on **moneylines**
- 📊 **~55% AUC** on **over/unders**

It also calculates:
- **Expected Value (EV)** for each bet  
- **Optimal stake size** using the **Kelly Criterion**

> 💡 Note: A safer bankroll approach is to wager **50% of the Kelly-recommended stake**.

---

## ⚙️ System Overview

A predictive system for NBA games built with **Python 3.11**, using state-of-the-art **machine learning** techniques to estimate win probabilities and totals (O/U), compute **Expected Value (EV)**, and apply the **Kelly Criterion** for bankroll management.  
Includes **Optuna** for intelligent **hyperparameter optimization**.

---

## 🧠 Tech Stack

| Library | Purpose |
|----------|----------|
| **TensorFlow** | Deep learning and neural network training |
| **XGBoost** | Gradient boosting framework |
| **Optuna** | Hyperparameter optimization (HPO) |
| **NumPy** | Scientific computing |
| **Pandas** | Data manipulation and analysis |
| **scikit-learn** | Metrics, preprocessing, and evaluation |
| **tqdm** | Progress bars |
| **colorama** | Colored terminal output |
| **requests** | HTTP data fetching |

---

## 📦 Requirements

- Python **3.11**
- All dependencies listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/HyakuzaO/NBA-Machine-Learning-Sports-Betting-By-HyakuzaO.git
cd NBA-Machine-Learning-Sports-Betting-By-HyakuzaO
pip install -r requirements.txt

# Create dataset with the latest data for 2025-26 season
cd src/Process-Data
python -m Get_Data
python -m Get_Odds_Data
python -m Create_Games

# Train models
cd ../Train-Models
python -m XGBoost_Model_ML
python -m XGBoost_Model_UO

After train the models set the Threshold found for ML predicitons in /src/Predict XGBoost_Runner.py

python main.py -xgb -kc -odds=fanduel


