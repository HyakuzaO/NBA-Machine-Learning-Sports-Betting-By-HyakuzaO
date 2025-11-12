# 🏀 NBA Sports Betting Using Machine Learning  
**Author:** [HyakuzaO](https://github.com/HyakuzaO)

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

### # NBA Machine Learning Sports Betting

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

