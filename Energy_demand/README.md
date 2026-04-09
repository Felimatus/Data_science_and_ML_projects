# Electricity Demand Forecasting - Czech Republic

## Project Overview

This project forecasts **daily electricity demand** for the Czech Republic using deep learning models. It compares traditional RNN-based architectures (LSTM, GRU) implemented in **TensorFlow/Keras** against a **Temporal Fusion Transformer (TFT)** implemented in **PyTorch** (via `pytorch-forecasting`). The goal is to predict electricity demand **14 days ahead**.

This is a portfolio project demonstrating:

- Multi-framework proficiency (TensorFlow + PyTorch)
- Time series forecasting with deep learning
- Domain-aware feature engineering (weather, energy crisis flags, regional climate data)
- Model comparison and interpretability (TFT variable importance and attention weights)

## Data Sources

### 1. Electricity Load Data - ENTSO-E Transparency Platform

- **Source:** [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- **Coverage:** Czech Republic (bidding zone CZ, code `10YCZ-CEPS-----N`)
- **Granularity:** Hourly (aggregated to daily min, mean, max)
- **Unit:** MW (megawatts)
- **Period:** 2022-07-01 to present
- **Access:** Requires free API key ([registration guide](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token))
- **Python library:** [`entsoe-py`](https://github.com/EnergieID/entsoe-py)

**Key datasets:**

| Dataset | API Method | Description | Use in Project |
| --- | --- | --- | --- |
| Actual Total Load | `query_load()` | Real electricity consumption (MW/h) | **Target variable** (daily min/mean/max) |
| Day-Ahead Load Forecast | `query_load_forecast()` | TSO's own demand prediction | Benchmark for model comparison |
| Day-Ahead Prices | `query_day_ahead_prices()` | Spot market price (EUR/MWh) | Exogenous feature (daily mean) |

### 2. Weather Data - Open-Meteo Archive API

- **Source:** [Open-Meteo Archive API](https://archive-api.open-meteo.com/v1/archive)
- **Coverage:** 5 weather stations across Czech Republic (see table below)
- **Granularity:** Daily
- **Period:** 2022-07-01 to present
- **Access:** Free, no API key required
- **Python library:** [`openmeteo-requests`](https://pypi.org/project/openmeteo-requests/)

**Weather variables collected per station:**

| Variable | Unit | Description |
| --- | --- | --- |
| `apparent_temperature_mean` | C | Daily mean "feels like" temperature |
| `apparent_temperature_max` | C | Daily maximum "feels like" temperature |
| `apparent_temperature_min` | C | Daily minimum "feels like" temperature |
| `precipitation_sum` | mm | Total daily precipitation |
| `rain_sum` | mm | Total daily rainfall |
| `pressure_msl_mean` | hPa | Daily mean sea-level pressure |
| `cloud_cover_max` | % | Daily maximum cloud cover |
| `dew_point_2m_mean` | C | Daily mean dew point at 2m height |

## Regional Weather Stations

Since ENTSO-E provides electricity data at the **national level** (not city-level), weather data is collected from **5 representative stations** across Czech macro-regions. Each region's weather features are **weighted by the macro-region's population share**, so that regions with more inhabitants (and thus higher energy consumption) have proportionally greater influence on the model inputs.

| Code | Macro-Region | Administrative Regions (kraje) | Population | Weight | Weather Station | Lat | Lon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CB | Central Bohemia | Praha + Středočeský | 2,864,095 | 0.2625 | **Prague** | 50.0880 | 14.4208 |
| NB | North Bohemia | Ústecký + Liberecký + Karlovarský + Královéhradecký + Pardubický | 2,637,437 | 0.2418 | **Ústí nad Labem** | 50.6607 | 14.0323 |
| SB | South Bohemia | Jihočeský + Plzeňský + Vysočina | 1,785,514 | 0.1637 | **České Budějovice** | 48.9757 | 14.4802 |
| SM | South Moravia | Jihomoravský + Zlínský | 1,808,341 | 0.1658 | **Brno** | 49.1951 | 16.6068 |
| NM | North Moravia | Moravskoslezský + Olomoucký | 1,814,113 | 0.1663 | **Ostrava** | 49.8209 | 18.2625 |
| | **Total** | **All 13 kraje and Prague** | **10,909,500** | **1.0000** | | | |

Population data: [Czech Statistical Office (ČSÚ), 31 December 2024](https://csu.gov.cz/rychle-informace/population-change-4-quarter-of-2024). Individual kraj figures via [Wikipedia — Regions of the Czech Republic](https://en.wikipedia.org/wiki/Regions_of_the_Czech_Republic).

## Engineered Features

### Energy Crisis Flag

A binary feature `crisis` is included to mark the **2022 European energy crisis** period, caused by the Russia-Ukraine war and gas supply disruptions:

| Period | `crisis` | Rationale |
| --- | --- | --- |
| 2022-07-01 to 2023-03-31 | 1 | Peak gas prices, EU emergency regulations, Czech government energy savings measures |
| 2023-04-01 onwards | 0 | Markets and demand patterns normalized |

### Temporal Features

- Day of week (one-hot encoded: weekday, Saturday, Sunday)
- Season (one-hot encoded: Winter, Spring, Summer, Fall)
- Holiday flag (Czech public holidays)
- Lagged load features (7-day and 14-day lags of mean load)

## Date Range

- **Start:** 2022-07-01 (post-COVID, demand patterns stabilized)
- **End:** Present (~2026-04-07)
- **Total:** ~1,370 daily observations

COVID-era data (pre-2022) was intentionally excluded due to heavily distorted electricity demand patterns from lockdowns and remote work.

## Forecasting Setup

- **Forecast horizon:** 14 days ahead
- **Input window:** 42 days (6 weeks)
- **Target:** Daily electricity load (min, mean, max in MW)
- **Total features:** 54 (weather, temporal, crisis flag, lagged load)

## Models

### TensorFlow/Keras (RNN-based)

1. **LSTM Seq2Seq** — Dilated causal Conv1D stack + Bidirectional LSTM (2 layers) for temporal modeling
1. **Conv1D + GRU Hybrid** — Same Conv1D front-end + Bidirectional GRU (2 layers)
1. **WaveNet-style** — Dilated causal convolutions with gated activation units, residual and skip connections

### PyTorch (Transformer-based)

4. **Temporal Fusion Transformer (TFT)** — via [`pytorch-forecasting`](https://pytorch-forecasting.readthedocs.io/) library
   - Multi-horizon forecasting with interpretable attention
   - Variable selection network (learns which features matter at each timestep)
   - Static vs. time-varying feature handling

## Model Sizing

Given the **modest** dataset (~1,370 observations), models are kept small to avoid overfitting and ensure fast training. All TensorFlow models share the same optimizer (SGD with momentum) and training callbacks (early stopping, learning rate reduction) defined in `fit_and_evaluate.py`. Hyperparameters were tuned to minimize validation MAE.

## GPU Compatibility

- **TensorFlow models** (LSTM, Conv+GRU, WaveNet): Work on most NVIDIA GPUs, including older cards (CUDA capability 6.1+).
- **PyTorch / TFT model**: Requires CUDA capability 7.0+ (Volta architecture or newer, e.g. GTX 1650+, RTX 2000/3000/4000/5000 series, Tesla V100+). Older GPUs (e.g. GTX 1050 Ti, GTX 1080) are not supported by recent PyTorch versions (this is my case).

If your GPU is not compatible with PyTorch, the TFT trainer falls back to CPU. To use GPU acceleration, change `accelerator="cpu"` to `accelerator="auto"` in `scripts/train_tft.py`.

## Sample Notebook & Testing Period

The `Sample/` notebook demonstrates model predictions on a backtest window that intentionally includes **Easter 2026** (April 3 and 6 are Czech public holidays). This makes the evaluation more realistic and challenging, as holiday periods break the regular weekday consumption pattern. The TFT model handles these disruptions significantly better than the LSTM, thanks to its variable selection and attention mechanisms.

## Results

All models are compared on the same held-out 14-day period (chronological split, no shuffling), which includes Easter 2026 holidays:

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
| --- | --- | --- | --- |
| **TFT** | **135.1** | **200.6** | **1.92** |
| LSTM | 407.0 | 524.9 | 5.87 |
| Conv+GRU | 439.1 | 564.7 | 6.36 |
| WaveNet | 497.7 | 581.4 | 7.07 |

The TFT significantly outperforms all other models, largely due to its variable selection mechanism which correctly identifies holidays and reduces predicted demand accordingly. The RNN-based models (LSTM, Conv+GRU) perform similarly to each other, while WaveNet — a purely convolutional architecture — shows the weakest results, likely due to the limited dataset size.

## Future Prediction with Forecasted Weather

The `download_weather.py` script also downloads a **14-day weather forecast** from the Open-Meteo Forecast API, saved as `data/raw/weather_forecast_CZ.csv`. This file is not used by any model currently — all training and backtesting uses real observed weather data.

However, the TFT model is already architecturally prepared for real future predictions: weather features are classified as `time_varying_known_reals`, meaning TFT expects them to be available for future timesteps. To actually predict the next 14 days into the future, you would need to:

1. Load `weather_forecast_CZ.csv` with the 14-day weather forecast
2. Append those 14 rows to the dataframe with future dates
3. Fill in the temporal features (`day_type`, `season`, `is_holiday`) for those dates
4. Run the TFT prediction on that extended dataframe

This is a natural next step for turning the project into a production-ready forecasting tool.

## Project Structure

```text
Energy_demand/
    README.md                      # This file
    config.py                      # Central path and parameter configuration
    data/
        raw/                       # Raw downloaded data
        processed/                 # Cleaned and merged datasets (.csv, .npz)
    models/                        # Saved trained models (.keras, .ckpt)
    scripts/
        download_energy.py         # ENTSO-E electricity load retrieval
        download_weather.py        # Open-Meteo weather data retrieval
        data_processing.py         # Merge raw data, add features, population-weight weather
        data_to_training.py        # Split, normalise, build tf.data.Datasets
        fit_and_evaluate.py        # Shared training utility for TF/Keras models
        train_lstm.py              # Bidirectional LSTM with dilated Conv1D
        train_conv_gru.py          # Bidirectional GRU with dilated Conv1D
        train_wavenet.py           # WaveNet-style model (dilated causal convolutions)
        train_tft.py               # Temporal Fusion Transformer (PyTorch)
    Sample/
        sample.ipynb               # Full pipeline demo and model comparison
```

The models saved in `models/` are the ones that provided the smallest MAE on the validation dataset. The data in `data/` is the one used for training and prediction in the notebook.

## Requirements

See `requirements.txt` for exact versions. Key dependencies:

- **TensorFlow** ~2.20 — LSTM, Conv+GRU, WaveNet models
- **PyTorch** ~2.10 + **pytorch-forecasting** ~1.7 + **Lightning** ~2.6 — TFT model
- **entsoe-py** ~0.7 — electricity load data from ENTSO-E API
- **openmeteo-requests** ~1.7 — weather data from Open-Meteo API
- **pandas**, **numpy**, **scikit-learn**, **matplotlib** — data processing and evaluation

## Reproducibility

To reproduce the weather data collection, use the coordinates listed in the Regional Weather Stations table above with the Open-Meteo Archive API. For electricity data, register for a free ENTSO-E API key and use the `entsoe-py` library with country code `CZ`.

## Author

Felipe Matus — [LinkedIn](https://www.linkedin.com/in/felipe-matus-3a5790285/)