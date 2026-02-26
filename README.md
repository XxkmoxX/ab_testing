# AB Testing

A practical collection of **A/B Testing**, **Causal Inference**, and **Bayesian Experimentation** notebooks, covering both classic statistical methods and industry-specific applications — e-commerce and O&G examples.

---

## 📂 Repository Structure

```
ab_testing/
├── 0_Experimento_A.ipynb               ← Classic A/B Test (Frequentist)
├── 1_Experimento_B.ipynb               ← Geo Testing with CausalImpact
├── 2_Expedia_EDA.ipynb                 ← Exploratory Data Analysis
├── 3_Expedia_Feature_Engineering.ipynb ← Feature Engineering Pipeline
└── Bayesian Approach/
    ├── 4_Boostraping.ipynb             ← Bootstrap Confidence Intervals
    ├── frac_stages.ipynb               ← Hydraulic Fracturing A/B Test
    └── geothermal.ipynb                ← Geothermal Reservoir Bayesian Analysis
```

---

## Notebooks

### 0 — Experimento A: Flight Checkout Upselling A/B Test
**Method:** Classic Frequentist A/B Testing

**Scenario:** Evaluates an aggressive luggage upselling intervention in the flight checkout flow of an OTA (Online Travel Agency).

- **Hypothesis:** The upselling will increase Revenue, but the added friction may reduce Conversion Rate.
- **Control (A):** Standard checkout flow.
- **Treatment (B):** Aggressive luggage upsell before payment.
- **Synthetic data:** 100,000 users, Bernoulli conversion simulation (A: 5.1%, B: 4.9%), log-normal revenue with realistic noise (negative values, corporate outliers).
- **Key steps:** Data cleaning, outlier treatment (Winsorizing at P99), metric calculation (Traffic, Conversion Rate, Revenue Per User, Total Revenue).
- **Metrics:** Conversion Rate, RPU (Revenue Per User), Total Revenue by variant.

---

### 1 — Experimento B: Geographic A/B Testing — Geo Testing
**Method:** Synthetic Control / Bayesian Structural Time Series (`CausalImpact` library)

**Scenario:** Tests whether changing the hotel ranking algorithm from *Most Popular* to *Best Price* increases total sales. Randomization by user is not possible due to inventory interference, so a geographic experiment is designed.

- **Hypothesis:** Prioritizing best price over popularity will increase bookings.
- **Control Pool:** Córdoba, Rosario, Salta (old algorithm).
- **Treatment Unit:** Mendoza (new algorithm applied on day 70 of 100).
- **Synthetic data:** Daily sales time series with weekly seasonality and linear trend. Treatment introduces a simulated +15% lift from day 70.
- **Validation:** Pre-period correlation > 0.9 between control cities and Mendoza; R² validation via Linear Regression (scikit-learn).
- **Model:** Bayesian regression using control cities as covariates to estimate the counterfactual (what Mendoza *would have sold* without the change).
- **Result:** +17.04% lift, 95% credibility interval [15.4%, 18.6%], p ≈ 0. **Recommendation: full Roll-Out.**

---

### 2 — Expedia Hotel Recommendations — EDA
**Dataset:** [Kaggle — Expedia Hotel Recommendations](https://www.kaggle.com/c/expedia-hotel-recommendations/data)  
**Scale:** ~37.6 million rows × 24 columns | Period: 2013–2014  
**Objective:** Predict which hotel cluster (0–99) a user will book.

**Analysis sections:**
1. Load & Preview — 1M random row sample for fast exploration
2. Shape, Data Types & Descriptive Statistics
3. Missing Values Analysis — `orig_destination_distance` ~NaN strategy
4. Target Variable Distribution — 100 classes, moderate imbalance
5. User Activity Analysis
6. Temporal Analysis — seasonality, lead time, booking hour
7. Geographic Analysis — high-cardinality country/city features
8. Booking vs Click Analysis — conversion patterns
9. Correlation Analysis — weak linear correlations → **non-linear models required**
10. Feature Distributions & Outlier Detection — log1p transform candidates
11. Key Insights & Recommendations

**Key finding:** All linear correlations with `hotel_cluster` are below 0.04 → **Recommended models: LightGBM / XGBoost / CatBoost**. Evaluation metric: **MAP@5:** ***Mean Average Precision at 5*** measures how well a model ranks the correct answer within the top 5 predictions..

---

### 3 — Expedia Hotel Recommendations — Feature Engineering
**Objective:** Transform raw columns into model-ready features encoding temporal, geographic, behavioural, and interaction signals.

**Pipeline sections:**
1. Setup & Data Loading — 1M-row random sample of `train.csv`
2. **Temporal Features** — booking month, week-of-year, hour, is_weekend, lead time, stay duration
3. **Stay & Search Features** — total guests, guests per room, room count signals
4. **Geographic Features** — same-continent flag, high-cardinality encoding
5. **User Behaviour Features** — search count, avg lead time, avg stay per user
6. **Destination Aggregation Features** — search count, popularity per destination/market
7. **Merge `destinations.csv`** — latent destination embeddings
8. **Missing Value Treatment** — binary `distance_known` flag + median imputation
9. **Encoding Categorical Features** — Label Encoding for tree-based models
10. **Feature Scaling** — StandardScaler for distance, lead time, stay duration (log1p variants)
11. **Feature Importance Preview** — shallow Random Forest (50 trees, depth 8) to validate signal
12. **Save Processed Dataset** — output to `train_features.parquet` (Snappy compression)

---

## Bayesian Approach

### 4 — Bootstrap Confidence Intervals for Ratio Metrics
**Method:** Vectorized Bootstrap (10,000 iterations)

**Problem:** Standard t-tests fail for **ratio metrics** like *Revenue / Clicks*, because the ratio of two random variables does not follow a simple known distribution.

- **Metric:** Revenue per Click (RPU = Revenue / Clicks)
- **Synthetic data:** 5,000 users per group. Clicks: Poisson(λ=2.0 / 2.1). Revenue: Log-Normal.
- **Bootstrap approach:** Generates a (10,000 × 5,000) resampling matrix to compute the full distribution of the lift.
- **Result:** Lift = **+5.94%** | 95% CI: **[+3.23%, +8.73%]** | p ≈ 0.0000
- **Decision:** CI does not cross 0% → Treatment B is statistically superior. 

---

### Hydraulic Fracturing Stage Spacing A/B Test (Vaca Muerta)
**Method:** Bayesian Inference with t-Student Posteriors  
**Domain:** Oil & Gas — Unconventional Reservoir Engineering

**Scenario:** YPF evaluates whether reducing fracture stage spacing (100m → 70m) in a Vaca Muerta shale well increases productivity enough to justify the extra completion cost.

- **Control (A):** Standard design — 100m spacing | Cost: $4,000,000 USD
- **Treatment (B):** Tighter spacing — 70m | Cost: $5,200,000 USD
- **Data:** 5 twin wells per group (PAD-scale test), 180-day cumulative production.
- **Why Bayesian?** Small sample (N=5) — frequentist tests require hundreds of observations. Bayesian naturally handles uncertainty with small datasets.
- **Posteriors:** t-Student distributions for production mean and economic profit.
- **Results:**
  - Technical success (B produces more): **98.98%**
  - Commercial success (B is more profitable): **90.90%**
  - Expected NPV A: $4,880,548 | Expected NPV B: $5,912,255

---

### Geothermal Reservoir Bayesian Risk Analysis
**Method:** Multivariate Bayesian Analysis with Informative Priors  
**Domain:** Geothermal Engineering — Thermo-Hydraulic Optimization

**Scenario:** A geothermal field evaluates whether **cyclic injection** (Group B) mitigates thermal breakthrough compared to **continuous injection** (Group A), without compromising reservoir pressure.

- **Control (A):** Continuous injection at constant flow rate (150 kg/s)
- **Treatment (B):** Cyclic injection (200 kg/s / 100 kg/s alternating every 12h)
- **Informative Priors:** Based on numerical simulation (TOUGH2/CMG) before field test.
- **Bivariate model:** Jointly models thermal benefit (tracer return delay) and hydraulic risk (reservoir pressure vs critical threshold of 40 bar).
- **Risk function:** If P_res < 40 bar → $2,000,000 USD penalty per well (artificial lift CAPEX).
- **Key insight:** B achieves ~99% thermal success but has ~27% probability of pressure failure, reducing overall commercial success to ~70% — demonstrating the risk of optimizing a single metric in isolation.

---

## Stack

| Library | Usage |
|---|---|
| `numpy` / `pandas` | Data simulation & manipulation |
| `scipy.stats` | Statistical tests, t-Student posteriors |
| `matplotlib` / `seaborn` | Visualizations |
| `scikit-learn` | Regression validation, feature scaling, Random Forest |
| `causalimpact` | Bayesian Structural Time Series (Geo Testing) |

---

## Notes

- The Expedia dataset CSV files (`train.csv`, `test.csv`, `destinations.csv`, `sample_submission.csv`) are **excluded** from this repository due to their large size (up to 3.8 GB). Download them from [Kaggle](https://www.kaggle.com/c/expedia-hotel-recommendations/data).
- All experiments use **synthetic data** unless stated otherwise, designed to replicate realistic OTA, O&G and geothermal scenarios.
