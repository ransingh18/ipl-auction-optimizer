# IPL Auction Optimizer

A data-driven system that predicts IPL player auction prices using machine learning, identifies behavioural biases in auction outcomes, and builds optimal squads under real auction constraints using integer linear programming.

Built as a portfolio project combining **Data Wrangling (BUSAN302)** and **Business Analytics (BUSAN300)** coursework at the University of Auckland.

**[Live Demo →](https://ipl-auction-optimizer.streamlit.app)** *(Streamlit web app)*

---

## What It Does

| Module | Description |
|--------|-------------|
| **Price Prediction** | Trains Ridge, Random Forest, GBM, and XGBoost models on 37 engineered features to estimate fair auction prices for 1,279 IPL players |
| **Bias Analysis** | Quantifies five behavioural biases in auction pricing — anchoring, recency, star premium, overseas premium, and auction inflation — using statistical tests |
| **Squad Optimizer** | Uses integer linear programming (PuLP + HiGHS) to select optimal 25-player squads under real IPL constraints: ₹120 Cr budget, 8 overseas cap, role minimums |
| **Web Dashboard** | Interactive Streamlit app with Plotly charts for exploring predictions, running optimizations, and visualising bias patterns |

---

## Key Results

### Model Performance (Temporal Split: Train 2013–2022, Test 2023+2025)

| Model | Test RMSE (₹ Cr) | CV R² | 
|-------|-------------------|-------|
| **GBM** | **5.18** | **0.47** |
| XGBoost | 5.23 | 0.47 |
| Random Forest | 5.23 | 0.52 |
| Ridge | 5.63 | 0.42 |

### Bias Effect Sizes

- **Anchoring:** Base price explains 20.1% of sale price variance beyond performance (r = 0.609)
- **Star Premium:** 6+ POTM players average ₹7.10 Cr vs ₹1.59 Cr for no awards (t = 14.57, p < 0.001)
- **Overseas Premium:** 48% cost-per-performance premium over Indian players
- **Auction Inflation:** Mega auctions carry 77% price premium over mini auctions (p < 0.001)

### Optimization (Best-Overall Scenario)

25 players selected, ₹119.2 Cr / ₹120 Cr budget used, 8/8 overseas slots filled, composite score: 1,075.8

---

## Project Structure

```
ipl-auction-optimizer/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard (5 pages)
├── data/
│   ├── processed/
│   │   ├── predictions.csv       # ML predictions (1,279 players, 47 cols)
│   │   └── predictions_with_bias.csv  # + bias columns (52 cols)
│   └── raw/                      # Source data (not tracked — see Setup)
├── models/
│   ├── best_model.pkl            # Trained GBM regressor
│   └── model_metadata.json       # Feature importance & metrics
├── notebooks/
│   ├── 01_data_collection.ipynb  # Data ingestion & cleaning
│   ├── 02_eda_ml_models.ipynb    # EDA, feature engineering, model training
│   ├── 03_optimization_engine.ipynb  # ILP squad builder
│   └── 04_behavioural_bias.ipynb # Bias quantification & statistical tests
├── outputs/                      # Generated charts (.png)
├── scripts/
│   └── optimizer.py              # Reusable optimization module
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/ranveersingh/ipl-auction-optimizer.git
cd ipl-auction-optimizer
pip install -r requirements.txt
```

### 2. Data

The processed datasets (`data/processed/`) are included in the repo and are sufficient to run the app and notebooks 02–04.

To reproduce from scratch, place the raw IPL auction and match data in `data/raw/` and run `notebooks/01_data_collection.ipynb`.

### 3. Run the app

```bash
streamlit run app/streamlit_app.py
```

### 4. Run the notebooks

Notebooks are numbered in order. Each can be run independently using the processed data.

---

## Technical Details

### Feature Engineering (37 features)

- **Core stats (21):** Career and recent batting/bowling metrics, IPL experience
- **Categorical (4):** Role, origin, base price, auction year
- **Engineered (12):** Per-season rates, all-rounder index, log transforms, inflation factor, recent-form percentages

### Optimization Constraints

The squad optimizer enforces real IPL auction rules:

- Budget cap: ₹120 Cr (adjustable)
- Squad size: ≤ 25 players
- Overseas limit: ≤ 8 players
- Role minimums: ≥ 2 each of batters, bowlers, all-rounders, wicketkeepers
- One player per slot (binary selection)

Solver: PuLP 4.x with HiGHS backend.

### Statistical Methods

- Pearson correlation for anchoring strength
- Independent t-tests for star premium and auction-type comparisons
- R² incremental analysis for bias explanatory power
- Temporal train/test split to prevent data leakage

---

## Tools & Libraries

Python · pandas · NumPy · scikit-learn · XGBoost · PuLP · HiGHS · SciPy · Matplotlib · Seaborn · Streamlit · Plotly

---

## Author

**Ranveer Singh**  
BCom (Finance & Business Analytics), University of Auckland  
Graduating November 2027

---

## Licence

MIT
