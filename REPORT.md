# Predictive Analysis of Public Transportation Delays

**Course:** AI (SEM-5)  
**Date:** December 19, 2025  
**Project Repository:** This folder (`AI/Project`)  

---

## 1) Executive Summary

This project builds an end-to-end machine learning pipeline to **predict public transportation (bus) delays** using a structured dataset of scheduled trips and contextual signals like weather, passenger count, and GPS coordinates.

We implemented the full workflow **from A to Z**:

1. **Data ingestion** from a raw “dirty” CSV file.
2. **Data cleaning** (normalization, type fixing, missing-value handling, edge-case handling for times).
3. **Feature engineering** (time-based features, weather severity, route frequency, and delay calculation).
4. **Model training and evaluation** with multiple regressors.
5. **Exploratory Data Analysis (EDA)** with saved plots.
6. **Explainability** using SHAP for the Random Forest model.
7. **Deployment-style interfaces**: a FastAPI API server and a simple web UI.

The pipeline generates reproducible artifacts (clean dataset, metrics, feature importances, plots, SHAP outputs) under `outputs/`.

---

## 2) Problem Statement

Public transport delays impact passenger satisfaction and operational planning. The goal is to learn a model that predicts:

- **Target:** `delay_minutes` (continuous regression target)

Given trip metadata:

- Route ID
- Scheduled time
- Weather condition
- Passenger count
- GPS location (latitude/longitude)

---

## 3) Project Structure (What each file does)

### Root-level scripts (entry points)

- `run_pipeline.py`  
  Runs the full pipeline: clean → engineer features → impute delay/time → train/evaluate → export outputs.

- `run_eda.py`  
  Generates EDA plots. If cleaned data doesn’t exist, it runs the pipeline first.

- `run_explainability.py`  
  Computes SHAP explainability outputs (for Random Forest). If cleaned data doesn’t exist, it runs the pipeline first.

- `run_web.py`  
  Runs the web UI server (`web/app.py`) via Uvicorn.

- `run_api_server.py`  
  Runs the API server (`api_server.py`) via Uvicorn.

### Data files

- `dirty_transport_dataset.csv`  
  Raw dataset (contains inconsistencies/missing values).

- `transport_cleaned_final_ready.csv`  
  A pre-cleaned/final dataset snapshot (separate from the pipeline outputs).

### Core Python package (`src/transport_delay/`)

- `cleaning.py`  
  Cleaning + normalization logic (route/weather normalization, parsing times, type conversion, filling missing values).

- `features.py`  
  Feature engineering and delay/time imputation.

- `modeling.py`  
  ML preprocessing + training + evaluation for multiple regressors, and feature importance extraction.

- `pipeline.py`  
  Orchestrates the end-to-end pipeline and exports CSV artifacts.

- `eda.py`  
  Generates visualizations (histograms, boxplots, scatterplots) and saves them as PNGs.

- `explainability.py`  
  SHAP-based model explanation using a Random Forest + preprocessor.

### Deployment / UI

- `api_server.py`  
  FastAPI-based REST API for delay prediction with input validation.

- `web/app.py`  
  FastAPI-based lightweight web interface (single page form) + `/predict` endpoint.

---

## 4) Data Understanding

### Dataset columns used

The raw dataset is read from CSV and expected to include:

- `route_id` (may be inconsistent format)
- `scheduled_time` (string, must parse to datetime)
- `actual_time` (string; may be missing or inconsistent)
- `weather` (may have spelling/format variations)
- `passenger_count` (may contain outliers or non-numeric values)
- `latitude`, `longitude` (may contain invalid values)

### Target construction

The project defines the delay target as:

$$
\text{delay\_minutes} = \frac{\text{actual\_dt} - \text{scheduled\_dt}}{60}
$$

This is calculated after parsing or imputing `actual_dt`.

---

## 5) Data Cleaning (Step-by-step)

**Module:** `src/transport_delay/cleaning.py`  
**Main function:** `clean_raw_dataset(df)`

### 5.1 Normalize Route IDs

Function `_normalize_route_id()` extracts digits from the route field and formats to:

- `R01`, `R02`, … (`R{n:02d}`)

Missing routes are finally filled with default:

- `R00`

### 5.2 Normalize Weather

Function `_normalize_weather()`:

- lowercases
- trims spaces
- maps synonyms like `sun` → `sunny`, `rain` → `rainy`

Missing weather is filled with the most frequent (mode), otherwise `"unknown"`.

### 5.3 Parse scheduled datetime

- `scheduled_dt = pd.to_datetime(scheduled_time, errors="coerce")`

### 5.4 Validate and Clean Numeric Fields

Passenger count:

- coerced to numeric
- values outside `[1, 200]` are set to NaN

GPS:

- coerced to numeric
- invalid lat/lon ranges are set to NaN
  - latitude must be in `[-90, 90]`
  - longitude must be in `[-180, 180]`

### 5.5 Parse actual time (robust parsing)

Function `_parse_actual_datetime()` handles:

- malformed time strings (e.g., `930` → `09:30`)
- AM/PM formats
- midnight edge cases
- trips that cross midnight (adjust day +1 if needed)

If parsing fails, `actual_dt` becomes `NaT`.

### 5.6 Missing-value imputation

The cleaner fills missing `passenger_count`, `latitude`, `longitude` using:

- per-route medians (grouped by `route_id`)
- global median fallback

---

## 6) Feature Engineering (Step-by-step)

**Module:** `src/transport_delay/features.py`

### 6.1 Time-based features

From `scheduled_dt`:

- `scheduled_hour`
- `scheduled_dayofweek` (0=Mon … 6=Sun)
- `day_type` = `weekend` if dayofweek ≥ 5 else `weekday`

### 6.2 Time-of-day bucket

`time_of_day` is binned into:

- morning (5–11)
- afternoon (12–16)
- evening (17–21)
- night (otherwise)

### 6.3 Weather severity

A simple ordinal mapping:

- sunny → 1
- cloudy → 2
- rainy → 3

### 6.4 Route frequency

`route_frequency` is computed as the count of how often a route appears in the dataset.

### 6.5 Delay minutes

Computed from timestamps:

- `delay_minutes = (actual_dt - scheduled_dt)` in minutes

### 6.6 Impute delay / actual time

Function `impute_delay_and_actual_time(df)`:

- fills missing `delay_minutes` using median within groups:
  - (`route_id`, `time_of_day`, `day_type`)
- global median fallback
- clips outliers using IQR rule
- creates export-friendly timestamps:
  - `scheduled_time_iso`
  - `actual_time_iso`

---

## 7) Modeling & Evaluation

**Module:** `src/transport_delay/modeling.py`  
**Main function:** `train_and_evaluate(df)`

### 7.1 Input features

Categorical:

- `route_id`, `weather`, `time_of_day`, `day_type`

Numeric:

- `scheduled_hour`, `scheduled_dayofweek`, `passenger_count`, `latitude`, `longitude`, `weather_severity`, `route_frequency`

### 7.2 Preprocessing

- One-hot encode categorical columns (`OneHotEncoder(handle_unknown="ignore")`)
- standardize numeric columns (`StandardScaler`)

### 7.3 Models trained

- Linear Regression
- KNN Regressor
- Gradient Boosting Regressor
- Random Forest Regressor

### 7.4 Metrics

Computed on a holdout split (80/20):

- MAE
- MSE
- RMSE
- R²

Also computed:

- Cross-validated RMSE (`CV_RMSE`) using 5-fold CV

### 7.5 Actual results (from `outputs/model_metrics.csv`)

| Model | MAE | RMSE | R² | CV_RMSE |
|---|---:|---:|---:|---:|
| random_forest | 43.6573 | 53.9912 | 0.2337 | 60.5832 |
| linear_regression | 44.0623 | 55.1547 | 0.2003 | 59.8798 |
| gradient_boosting | 49.2699 | 58.7360 | 0.0931 | 65.2285 |
| knn | 45.7127 | 59.5503 | 0.0678 | 62.6250 |

**Best RMSE:** Random Forest.

### 7.6 Feature importance (top signals)

From `outputs/feature_importance.csv` (Random Forest), strongest drivers include:

- `scheduled_hour`
- `longitude`
- `latitude`
- `passenger_count`
- `scheduled_dayofweek`
- time-of-day indicators (e.g., `time_of_day_night`)

This suggests the model relies heavily on **time patterns** and **location**.

---

## 8) Exploratory Data Analysis (EDA)

**Module:** `src/transport_delay/eda.py`  
**Runner:** `run_eda.py`

Generated plots saved to `outputs/figures/`:

- `delay_distribution.png` — distribution of delays
- `delay_by_weather.png` — boxplot of delay vs weather
- `delay_by_time_of_day.png` — boxplot of delay vs time buckets
- `delay_by_day_type.png` — boxplot of weekday vs weekend
- `passenger_by_route.png` — passenger distribution by route
- `route_frequency.png` — frequency of routes
- `gps_scatter.png` — location scatter colored by route

These plots support understanding how delay varies across conditions.

---

## 9) Explainability (SHAP)

**Module:** `src/transport_delay/explainability.py`  
**Runner:** `run_explainability.py`

We used SHAP (SHapley Additive exPlanations) to interpret the Random Forest.

Artifacts:

- `outputs/shap_mean_abs.csv` — global feature importance via mean absolute SHAP
- `outputs/figures/shap_summary.png` — summary plot

This provides model transparency and supports discussion of which features drive delay predictions.

---

## 10) Deployment Interfaces

### 10.1 REST API (`api_server.py`)

- Framework: FastAPI
- Endpoint: `POST /predict`
- Validates inputs (passenger range, GPS ranges, time format, weather categories)
- Preprocesses a single trip into the same feature space and predicts delay.

### 10.2 Web UI (`web/app.py`)

- Framework: FastAPI
- Serves a simple HTML form at `/`
- Calls `/predict` behind the scenes.

Both servers start with Uvicorn on port `8000`.

---

## 11) Outputs (Deliverables)

After running the pipeline:

- `outputs/cleaned_transport_dataset.csv`
- `outputs/model_metrics.csv`
- `outputs/feature_importance.csv`

After EDA:

- `outputs/figures/*.png`

After explainability:

- `outputs/shap_mean_abs.csv`
- `outputs/figures/shap_summary.png`

---

## 12) How to Run (Reproducibility)

### 12.1 Install dependencies

```powershell
pip install -r requirements.txt
```

### 12.2 Run the full pipeline

```powershell
python run_pipeline.py
```

### 12.3 Generate EDA plots

```powershell
python run_eda.py
```

### 12.4 Generate SHAP explainability outputs

```powershell
python run_explainability.py
```

### 12.5 Run the web UI

```powershell
python run_web.py
```

Then open: `http://localhost:8000`

### 12.6 Run the API server

```powershell
python run_api_server.py
```

API docs will be available at: `http://localhost:8000/docs`

---

## 13) Limitations

- Dataset size and quality may limit achievable accuracy (RMSE remains relatively high).
- Delay imputation uses medians; more sophisticated imputation could improve targets.
- Weather is simplified to only three categories.
- Model hyperparameters are basic; tuning could improve performance.

---

## 14) Required Reflections (Data & Modeling Risks)

This section reflects on sources of bias, noise, and instability that can affect the validity of our conclusions.

### 14.1 Bias introduced during imputation

We impute missing values in multiple places:

- In `src/transport_delay/cleaning.py`, missing or invalid `passenger_count`, `latitude`, and `longitude` are filled using **per-route medians**, with a global median fallback.
- In `src/transport_delay/features.py`, missing `delay_minutes` is filled using medians within groups (`route_id`, `time_of_day`, `day_type`), with a global median fallback.

This approach is simple and robust, but it can introduce **systematic bias**:

- It reduces variance (replacing missing values with “typical” values), which may make the model appear more stable than it truly is.
- It can hide rare but important scenarios (e.g., routes with genuinely unusual passenger distributions).
- If missingness is not random (e.g., sensors fail more often in bad weather), median imputation may bias the dataset toward “normal” conditions.

### 14.2 Correlations between time-related features

Several engineered features are derived from the same source (`scheduled_dt`) and are therefore correlated:

- `scheduled_hour` ↔ `time_of_day`
- `scheduled_dayofweek` ↔ `day_type`

This multicollinearity is especially relevant for linear models (e.g., Linear Regression), where correlated predictors can:

- inflate coefficient variance
- make interpretation of individual coefficients less reliable

Tree-based models (Random Forest / Gradient Boosting) are usually more tolerant, but correlated features can still split “importance” across multiple related columns (especially after one-hot encoding).

### 14.3 Noise affecting model stability

Delay prediction has multiple sources of noise that the dataset may not capture:

- traffic variability
- operational events (breakdowns, driver shift changes)
- random passenger loading time

Because the target `delay_minutes` is partially derived from parsed/imputed time fields, it can include measurement noise. We attempt to control extreme outliers by clipping delays using an IQR-based rule in `impute_delay_and_actual_time()`, but this can also remove legitimate extreme delays.

Overall, these noise sources can cause instability in model training and evaluation (e.g., different train/test splits can produce noticeably different metrics when the dataset is small).

### 14.4 Weather inconsistencies

Weather strings are normalized in `_normalize_weather()` to a small set (`sunny`, `cloudy`, `rainy`). While this reduces spelling issues, it also introduces limitations:

- It collapses potentially richer weather information (e.g., “storm”, “heavy rain”, “fog”) into coarse categories.
- If the raw dataset contains misreported weather, the model can learn wrong relationships.
- The `weather_severity` feature is an ordinal mapping (sunny=1, cloudy=2, rainy=3); if real-world severity does not match this ordering in the data, it may introduce modeling bias.

### 14.5 Possible overfitting in high-complexity models

Random Forest and Gradient Boosting are higher-capacity models than Linear Regression and can overfit if:

- the dataset is small
- there are noisy features
- there are many one-hot encoded route/time categories

We reduce this risk by evaluating on a held-out test set and also reporting cross-validated RMSE (`CV_RMSE`). However, additional regularization and careful hyperparameter tuning (depth, min samples per leaf, etc.) would provide stronger protection.

### 14.6 GPS errors limiting spatial analysis

We validate GPS ranges (latitude in [-90, 90], longitude in [-180, 180]) and impute missing coordinates using route medians. Still, GPS issues can limit spatial insights:

- GPS readings may be noisy (urban canyon, sensor drift).
- Different stops along the same route can have different delay patterns, but our route-level median imputation collapses that spatial detail.
- Without additional spatial context (stop IDs, road network distance, traffic), latitude/longitude may only partially explain delay.

---

## 15) Future Work

- Add more features (traffic, holidays, special events, route distance).
- Tune models (Bayesian optimization / grid search).
- Try advanced models (XGBoost/LightGBM/CatBoost).
- Improve time parsing with explicit timezone handling.
- Save “trained model artifacts” to disk and load them rather than training on server startup.

---

## 16) Conclusion

We delivered a complete ML project pipeline for predicting public transport delays, including cleaning, feature engineering, modeling, evaluation, visualization, interpretability, and practical deployment interfaces. The Random Forest model achieved the best overall RMSE among the models tested and the explainability analysis provides insights into which features drive predictions.
