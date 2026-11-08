# GNSS Error Prediction System

**A deep-learning pipeline that forecasts GNSS satellite navigation errors over several time horizons, with a prediction API and an ISRO-styled monitoring dashboard. Built for a Smart India Hackathon (SIH) problem.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

## Overview

GNSS positioning (GPS, Galileo, BeiDou, GLONASS, IRNSS, QZSS) is only as accurate as the satellite clock and orbit corrections it relies on. Those corrections drift because of ionospheric delay, satellite clock error and orbit inaccuracies. The idea is to learn those error patterns from past measurements and predict them ahead of time, so a receiver can correct for them before they degrade the fix.

This repo is my take on that problem for the Smart India Hackathon. The data spec came from the SIH brief (a PDF that is no longer in the repo), and the dashboard is branded as an "ISRO GNSS Dashboard". The system predicts four error channels (clock, x, y and z) at five lookahead horizons from 15 minutes up to 24 hours, and reports an uncertainty band with each prediction.

It is set up like a production ML service: a training and cross-validation pipeline, a FastAPI prediction endpoint, and a multi-page Streamlit dashboard with role-based login. To be straight about status: the pipeline, API and dashboard are all there, but the model is **not yet accurate**. Cross-validation R² is slightly negative (it doesn't beat the mean baseline), and no trained checkpoint, dataset or Docker setup is committed. Treat this as a work-in-progress prototype that shows the full architecture, not a finished, calibrated product.

## Key Features

- **Multi-horizon forecasting:** predicts clock/x/y/z error at horizons of 15, 30, 60, 120 and 1440 minutes from a single input sequence.
- **Two model designs:** a `MultiHorizonTransformer` with one prediction head per horizon, and an `EnsembleModel` that combines an LSTM branch and a Transformer branch.
- **Uncertainty estimation:** the Transformer's heads predict both a mean and a log-variance, trained with a Gaussian negative-log-likelihood loss, so every prediction comes with a standard deviation.
- **Probabilistic evaluation:** besides MAE/RMSE/R²/MAPE/sMAPE, the metrics module computes CRPS and per-confidence-level calibration (50/80/90/95/99%) to check whether the uncertainty estimates are honest.
- **Feature engineering for satellite time-series:** satellite-type and orbital-period derivation, cyclical time encodings, per-satellite rolling mean/std windows, and first/second error derivatives.
- **K-fold cross-validation training:** 5-fold CV with `SelectKBest` feature selection, AdamW, gradient clipping, and a warmup + cosine-annealing learning-rate schedule.
- **Prediction API:** FastAPI service exposing `/predict`, `/health` and `/info` under `/api/v1`, with request validation, CORS, and auto-generated Swagger/ReDoc docs.
- **Monitoring dashboard:** Streamlit app with four pages (Dashboard, Satellites, Alerts, Admin), session-based auth with admin/operator/viewer roles, and Plotly charts.
- **Config files:** model dimensions, horizons, sequence length and API settings live in `config/config.yaml`, and users live in `config/users.yaml`. The training script does not read the model settings from there (see How It Works).

## How It Works

The flow is the usual time-series ML shape: raw GNSS records → feature engineering → sequence windows → model → multi-horizon prediction with uncertainty → API/dashboard.

### Data and feature engineering

`AdvancedGNSSProcessor` (`src/data/advanced_processor.py`) is the main feature builder. Given a dataframe of satellite measurements it:

- Derives **satellite type** from the `sat_id` prefix (`G`→GEO, `M/E/C/R/J/I`→MEO) and maps an approximate **orbital period** (24h for GEO, 12h for MEO).
- Adds **temporal features** from `utc_time` (hour, day-of-week, day-of-month, month), plus cyclical sin/cos encodings and an orbital-phase feature so the model sees time as periodic rather than linear.
- Adds **error features** computed per satellite (grouped by `sat_id`): rolling mean and std over configurable windows (`[6, 12, 24, 48]` hours) and first/second differences of each error channel to capture rate-of-change.
- Scales numeric features with a `RobustScaler` and one-hot encodes the categorical satellite type inside a scikit-learn `ColumnTransformer`. Targets get their own `RobustScaler` so predictions can be inverse-transformed back to real units at serving time.

The targets are the four error columns: `clock_error`, `x_error`, `y_error`, `z_error`. `config/config.yaml` sets a sequence length of 96 steps (24 hours at 15-minute sampling), but `src/train.py` uses its own `GNSSDataProcessor` (`src/data/data_processor.py`) with a sequence length of 24.

### Models

There are two architectures in `src/models/`:

**MultiHorizonTransformer** (`advanced_model.py`) is the headline design. Input features are projected to `d_model`, given sinusoidal positional encoding, and run through a stack of `TransformerEncoder` layers. The last time-step representation is combined with a learned **horizon embedding**, then fed into a **separate prediction head per horizon**. Each head outputs `2 × output_size` values (a mean and a log-variance), so the model produces both a point estimate and an uncertainty per channel per horizon. It is trained with a custom `GaussianNLLLoss`. Config from `config.yaml`: `d_model=256`, `nhead=8`, `num_layers=6`, `dropout=0.1`, horizons `[15, 30, 60, 120, 1440]`.

**EnsembleModel** (`model.py`) has two branches: an LSTM branch (with layer norm and residual handling) and a Transformer branch run in parallel. Their last-step outputs are concatenated, passed through a `MultiheadAttention` block, and combined by an MLP head with learnable softmax ensemble weights. Both branches are normalised and Xavier-initialised for gradient stability.

### Training and cross-validation

`src/train.py` runs the 5-fold pipeline:

- Combines train/val splits, then selects the top features per target with `SelectKBest(f_regression)`.
- For each `KFold` split, builds `TensorDataset`/`DataLoader` pairs and trains an `EnsembleModel` with **AdamW** (weight decay), **MSE loss**, **gradient clipping** (`max_grad_norm=0.5`) and a custom **WarmupCosineScheduler** (linear warmup then cosine annealing to a floor LR).
- Tracks metrics every epoch, applies **early stopping** on validation loss, and saves the best per-fold checkpoint plus a metrics JSON.
- Aggregates average and best-validation metrics across folds and writes a timestamped `results/cv_results_*.json`.

The hyperparameters for this run are hardcoded in `main()` (a deliberately small model, with LSTM and Transformer sizes of 8), not read from `config/config.yaml`. The `MultiHorizonTransformer` is defined but is not what `src/train.py` trains.

### Metrics

`src/evaluation/metrics.py` computes MAE, MSE, RMSE, normalized RMSE, MAPE, sMAPE and a clipped R². When uncertainty is available it adds **CRPS** (Continuous Ranked Probability Score) for Gaussian forecasts and a **calibration** report comparing expected vs. actual coverage at the 50/80/90/95/99% confidence levels, plus a mean calibration error.

### Serving

`src/api/` wraps a trained model in a FastAPI app. A `ModelService` loads the checkpoint and the saved `joblib` processor on startup (CUDA if available, otherwise CPU). The `/api/v1/predict` endpoint takes a JSON sequence of feature dicts plus a horizon, runs the processor, calls the model, inverse-transforms the mean back to real units, and returns the prediction, the per-channel uncertainty and metadata. `/api/v1/health` and `/api/v1/info` report readiness, device and the supported horizons. If no checkpoint is present the API still starts but reports the model as not ready.

### Dashboard

`app.py` / `src/dashboard/` is a Streamlit app titled "ISRO GNSS Dashboard". A simple session-based authenticator (hashed passwords in `config/users.yaml`, roles `admin` / `operator` / `viewer`) gates access, and the sidebar switches between **Dashboard**, **Satellites**, **Alerts** and an admin-only **Admin** panel. The components in `src/dashboard/components/` render Plotly charts for error distributions, satellite views and confidence intervals, with custom ISRO-blue CSS theming. The users in `users.yaml` are demo accounts, so change them before any real deployment.

## Results / Highlights

From the latest cross-validation run (`results/cv_results_20251001_083945.json`, 5 folds, scaled targets):

- **Average validation MAE ≈ 0.261**, **RMSE ≈ 0.356**
- **Best validation R² ≈ -0.01** (average across folds ≈ **-0.06**)
- MAPE / sMAPE are high (sMAPE ≈ 190%), which mostly comes from error values sitting near zero

The honest read: the pipeline runs end-to-end, logs CV metrics, and the error magnitudes are small, but a non-positive R² means the model isn't predicting better than guessing the mean. The architecture (uncertainty-aware multi-horizon Transformer, LSTM+Transformer ensemble, calibration metrics, serving stack) is written, but the modelling needs more data and tuning before the numbers are usable. I'm keeping the metrics here as they are rather than dressing them up.

## Tech Stack

- **Language:** Python 3.8+
- **Deep learning:** PyTorch (LSTM, TransformerEncoder, MultiheadAttention, custom Gaussian NLL loss, positional encoding)
- **Classical ML / data:** scikit-learn (`RobustScaler`, `ColumnTransformer`, `OneHotEncoder`, `SelectKBest`, `KFold`), pandas, NumPy, SciPy, joblib
- **API:** FastAPI, Uvicorn, Pydantic
- **Dashboard / viz:** Streamlit, Plotly, matplotlib, seaborn
- **Config:** PyYAML
- **Testing:** pytest, pytest-cov, pytest-asyncio are in `requirements.txt`, but the repo has no tests

## Getting Started

### Prerequisites
- Python 3.8 or newer
- `pip` (a virtual environment is recommended)
- Optional: a CUDA-capable GPU for faster training (CPU works too)

### Installation
```bash
git clone https://github.com/DCode-v05/Space-Tech.git
cd Space-Tech
pip install -r requirements.txt
```

### Running

Train (runs 5-fold cross-validation and writes results to `results/`):
```bash
python -m src.train
```

Launch the dashboard:
```bash
streamlit run app.py
# opens on http://localhost:8501
```

Run the prediction API:
```bash
python -m src.api.main
# port comes from config/config.yaml (8000), docs at /docs
```

Note: training expects `data/raw/DATA_GEO_Train.csv` (the path is set in `src/train.py`), and the API needs a trained checkpoint plus a saved processor. Neither the data nor a checkpoint is committed (CSVs are gitignored), so you'll need to supply data and train first. The repo has no Dockerfile or Docker Compose file.

## Usage

**Dashboard:** open `http://localhost:8501`, log in (users defined in `config/users.yaml`) and move between the Dashboard / Satellites / Alerts pages. Admins also get the Admin panel.

**API:** POST a sequence and a horizon to `/api/v1/predict`:
```json
{
  "sequence": [
    {"clock_error": 0.10, "x_error": 0.20, "y_error": 0.30, "z_error": 0.40},
    {"clock_error": 0.15, "x_error": 0.25, "y_error": 0.35, "z_error": 0.45}
  ],
  "prediction_horizon": 15
}
```
The response contains the predicted error per channel, a per-channel uncertainty (standard deviation) and metadata. Hit `/api/v1/health` for readiness and `/api/v1/info` for the supported horizons and feature list.

**Retraining:** edit the `config` dict in `main()` of `src/train.py` (sequence length, model size, epochs) and rerun the training command. `config/config.yaml` does not drive training.

## Project Structure

```
Space-Tech/
├── app.py                          # Streamlit dashboard entry point (ISRO GNSS Dashboard)
├── config/
│   ├── config.yaml                 # data, model, training, evaluation, API config
│   └── users.yaml                  # dashboard users + roles (admin/operator/viewer)
├── results/                        # timestamped 5-fold cross-validation metrics (JSON)
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app, startup model loading, CORS
│   │   ├── endpoints.py            # /predict, /health, /info routes + validation
│   │   └── service.py              # ModelService: load checkpoint, predict, inverse-transform
│   ├── config/users.yaml           # second users file, different format
│   ├── data/
│   │   ├── advanced_processor.py   # feature engineering + scaling pipeline
│   │   └── data_processor.py       # sequence/data prep used by src/train.py
│   ├── models/
│   │   ├── advanced_model.py       # MultiHorizonTransformer + GaussianNLLLoss
│   │   ├── model.py                # EnsembleModel (LSTM + Transformer)
│   │   └── train.py                # model-level training entry (overlaps src/train.py)
│   ├── training/trainer.py         # training loop helpers
│   ├── evaluation/
│   │   ├── metrics.py              # MAE/RMSE/R²/MAPE + CRPS + calibration
│   │   └── visualization.py        # evaluation plots
│   ├── dashboard/
│   │   ├── app.py                  # dashboard app module
│   │   ├── config.py               # dashboard settings
│   │   ├── pages/                  # dashboard / satellites / alerts / admin pages
│   │   ├── components/             # data + satellite Plotly visual components
│   │   ├── simple_auth.py          # session auth + role checks (the one app.py uses)
│   │   └── auth.py, auth_new.py    # two older auth variants, unused by app.py
│   ├── analysis/target_analysis.py # target distribution analysis
│   ├── train.py                    # 5-fold CV training pipeline
│   └── utils/data_utils.py         # helpers
├── .gitignore
└── requirements.txt
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
