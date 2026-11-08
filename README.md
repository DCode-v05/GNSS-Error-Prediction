# Space-Tech

A prototype for forecasting GNSS satellite clock and orbit errors at several horizons, with a FastAPI prediction service and a Streamlit monitoring dashboard. It was written for a Smart India Hackathon problem on GNSS error prediction.

The model does not work yet. Cross-validation R2 on the scaled targets is slightly negative, which means it does not beat predicting the mean. No dataset and no trained checkpoint are committed. What is here is the pipeline, the model code, the API and the dashboard.

## What it does

Satellite positioning depends on clock and orbit corrections that drift over time. The goal is to learn those errors from past measurements and predict four channels (`clock_error`, `x_error`, `y_error`, `z_error`) at horizons of 15, 30, 60, 120 and 1440 minutes, with an uncertainty estimate for each.

## Parts

- **Features** (`src/data/advanced_processor.py`): satellite type and orbital period from the `sat_id` prefix, hour/day/month features with sin/cos encoding, per-satellite rolling mean and std, first and second differences of each error channel. Numeric features and targets use `RobustScaler`.
- **Models** (`src/models/`): `MultiHorizonTransformer` in `advanced_model.py` has one head per horizon that outputs a mean and log-variance, trained with a Gaussian NLL loss. `EnsembleModel` in `model.py` runs an LSTM branch and a Transformer branch in parallel and combines them with attention and an MLP head.
- **Training** (`src/train.py`): 5-fold cross-validation with `SelectKBest` feature selection, AdamW, gradient clipping, warmup plus cosine learning-rate schedule and early stopping. It trains the `EnsembleModel`, with its hyperparameters hardcoded in `main()` rather than read from `config/config.yaml`.
- **Metrics** (`src/evaluation/metrics.py`): MAE, MSE, RMSE, MAPE, sMAPE, R2, plus CRPS and calibration at 50/80/90/95/99% when uncertainty is available.
- **API** (`src/api/`): FastAPI with `POST /api/v1/predict`, `/api/v1/health` and `/api/v1/info`. It loads a checkpoint and a joblib processor on startup and starts without them, reporting the model as not ready.
- **Dashboard** (`app.py`, `src/dashboard/`): Streamlit pages for Dashboard, Satellites, Alerts and an admin-only Admin panel, behind a simple login with admin, operator and viewer roles from `config/users.yaml`.

## Results

Latest cross-validation run, `results/cv_results_20251001_083945.json` (5 folds, scaled targets): average validation MAE about 0.26 and RMSE about 0.36. Average validation R2 is negative, and sMAPE is around 190%, which is mostly a consequence of error values sitting near zero. More data and tuning are needed before these numbers mean anything.

## Setup and run

```bash
git clone https://github.com/DCode-v05/Space-Tech.git
cd Space-Tech
pip install -r requirements.txt
```

Training expects `data/raw/DATA_GEO_Train.csv` (path set in `src/train.py`), which is not in the repo.

```bash
python -m src.train           # cross-validation, writes results/cv_results_*.json
streamlit run app.py          # dashboard on http://localhost:8501
python -m src.api.main        # API, port from config/config.yaml; docs at /docs
```

The API needs a trained checkpoint and saved processor to answer `/predict`. Example request:

```json
{
  "sequence": [
    {"clock_error": 0.10, "x_error": 0.20, "y_error": 0.30, "z_error": 0.40},
    {"clock_error": 0.15, "x_error": 0.25, "y_error": 0.35, "z_error": 0.45}
  ],
  "prediction_horizon": 15
}
```

## Layout

```
app.py              Streamlit entry point
config/             config.yaml, users.yaml
src/api/            FastAPI app, endpoints, model service
src/dashboard/      pages, components, auth modules
src/data/           feature processors
src/models/         MultiHorizonTransformer, EnsembleModel
src/evaluation/     metrics and plots
src/train.py        cross-validation training
results/            saved cross-validation runs
```

## Known problems

- `src/dashboard/` contains three auth modules (`auth.py`, `auth_new.py`, `simple_auth.py`); the app uses `simple_auth.py`.
- There are two `users.yaml` files (`config/` and `src/config/`) with different formats, and `src/train.py`, `src/models/train.py` and `src/training/trainer.py` overlap.
- `config/config.yaml` describes a larger model (sequence length 96, d_model 256) than the one `src/train.py` trains.
