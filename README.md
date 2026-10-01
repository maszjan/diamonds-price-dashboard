# Diamonds Price Dashboard

An interactive Streamlit dashboard that walks through a complete data-processing pipeline on the public `diamonds` dataset: data diagnosis and cleaning, price anomaly detection, dimensionality reduction (PCA and t-SNE), a Random Forest price model and a live valuation calculator. The dashboard UI is in Polish.

## What it covers

The app is a single scrolling page split into seven sections:

1. **Goal and data source**: what the analysis tries to answer (how much of a diamond's price can be explained by its measurable features) and where the data comes from.
2. **Dataset**: overview of the columns (carat, cut, color, clarity, depth, table, x, y, z, price) and their distributions.
3. **Cleaning and integration**: removal of duplicates, detection of impossible or inconsistent measurements, imputation, and a before/after normalization view.
4. **Price anomalies**: overpriced and undervalued diamonds found with a carat-only baseline model and an adjustable residual percentile, plus a comparison of Isolation Forest and Local Outlier Factor. Includes a search by diamond number.
5. **Dimensionality reduction**: PCA (scree plot and 2D projection) and t-SNE, colored by cut or color.
6. **Predictive model**: Random Forest metrics, correlation heatmap, actual vs predicted values and feature importance.
7. **Valuation calculator**: choose carat, cut, color and clarity and get an estimated market price with a confidence range.

## Key results

| Item                                     | Value                    |
| ---------------------------------------- | ------------------------ |
| Rows in raw dataset                      | 53,940                   |
| Duplicates removed                       | 146                      |
| Rows after cleaning                      | 53,794                   |
| Rows with zero x/y/z dimension           | 19                       |
| Rows with inconsistent measurements      | 27                       |
| Overpriced / undervalued diamonds        | 1,366 / 23               |
| PCA components for 90% of variance       | 4                        |
| Price vs carat correlation               | 0.92                     |
| Carat-only baseline R2                   | 0.83                     |
| Random Forest R2 / MAE / RMSE (log price)| 0.988 / 0.083 / 0.110    |

## Methodology

The pipeline lives in `data_pipline.ipynb` and is designed to avoid data leakage:

- The 80/20 train/test split (`random_state=42`) happens before any transformation learns from the data.
- Median imputation (per `cut`), group aggregates, `RobustScaler` and PCA are fitted on the training set only.
- Anomaly detection uses a baseline price model (price as a power of carat) with IQR thresholds computed on the training set, and Isolation Forest and LOF with `contamination=0.05`.
- Feature engineering uses deterministic formulas only (volume, depth recomputed from x/y/z, log of carat). Categorical features use fixed ordinal encodings.
- Features derived from the price itself are deliberately excluded from the model.
- The model is a `RandomForestRegressor` (120 trees, max depth 10, min leaf 3) predicting log price. All metrics are computed on the held-out test set.
- t-SNE (perplexity 30) is used for visualization only and does not feed the model.
- The calculator estimates x, y and z from linear relationships with carat observed in the training data, and derives the confidence range from the spread of predictions across the individual trees.

## Project structure

```
diamonds-price-dashboard/
├── app.py                  # Streamlit dashboard
├── data_pipline.ipynb      # full data pipeline and model training
├── diamonds.csv            # source dataset
├── artifacts/              # precomputed outputs loaded by the app
│   ├── model.pkl           # trained Random Forest
│   ├── model_metrics.json
│   ├── eda_overview.json
│   ├── scale_params.json, encoding_maps.json, calc_defaults.json
│   ├── df_sample.csv, diamonds_search.csv, predictions.csv
│   ├── pca_2d.csv, pca_scree.csv, tsne_2d.csv
│   └── feature_importance.csv, anomaly_summary.csv, ...
├── Dockerfile
└── requirements.txt
```

The dashboard does not retrain anything at startup. It only reads the files from `artifacts/`, which are produced by the notebook.

## Tech stack

- Streamlit
- pandas, NumPy
- Plotly
- scikit-learn

Pinned versions are in `requirements.txt`.

## Getting started

### Local

```bash
git clone https://github.com/maszjan/diamonds-price-dashboard.git
cd diamonds-price-dashboard

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at http://localhost:8501.

### Docker

```bash
docker build -t diamonds-dashboard .
docker run -p 8501:8501 diamonds-dashboard
```

### Regenerating the artifacts

To rebuild the files in `artifacts/`, run `data_pipline.ipynb` from top to bottom. The notebook needs `jupyter` in addition to the packages in `requirements.txt`. Use the same scikit-learn version as in `requirements.txt` so that `model.pkl` can be loaded by the app.

## Data source

The data is the public `diamonds` dataset from the ggplot2 / tidyverse project: https://github.com/tidyverse/ggplot2/blob/main/data-raw/diamonds.csv

## Limitations

- The calculator is a demo based on historical market data and is not a jeweler's appraisal.
- Depth and table are fixed at their median values in the calculator, and x, y, z are estimated from carat.
- The dataset is limited to diamonds up to about 2.3 carat, so the calculator range is restricted accordingly.
- Anomaly detection is descriptive. Isolation Forest and LOF have low precision against the baseline-based labels (about 0.19 and 0.09).
- The dashboard text is in Polish only.
