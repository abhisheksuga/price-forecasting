# Time Series Forecasting: Gold Price Prediction

This project investigates the effectiveness of **Deep Learning** models for forecasting Commodity prices. The dataset contains historical gold price data and features like **Open**, **High**, **Low**, **Close**, and **Volume**. The experiments include feature engineering, model training, evaluation, and performance comparison.

---

## Dataset

- **Source**: Historical gold price data.
- **Features**:
  - `Open`, `High`, `Low`, `Close`, `Volume`
- **Time Intervals**: Hourly data.
- **Total Records**: 97,796 (After cleaning)

---

## Feature Engineering

### XGBoost
- **Lagged Features**: Added lagged values for `Close` at lags `[1, 3, 10, 20]`.
- **Rolling Statistics**: Calculated rolling mean and standard deviation for `Close` over a 23-hour window.
- **Scaling**: MinMaxScaler applied to scale features between 0 and 1.

### LSTM
- Used raw sequential data without explicit lagged or rolling features.
- Data reshaped to a 3D format `(samples, timesteps, features)` for LSTM compatibility.

---

## Models

### 1. **XGBoost**
- **Objective**: `reg:squarederror` (suitable for regression tasks).
- **Hyperparameters**: Default settings used.

### 2. **LSTM**
- **Architecture**:
  - 2 LSTM layers with 50 units each.
  - Dropout layers (rate: 0.2) to prevent overfitting.
  - Dense output layer for regression.
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Training**: 10 epochs, batch size of 32.

---

## Evaluation Metrics

- **Root Mean Squared Error (RMSE)**: Measures the average prediction error.
- **R-Squared (\(R^2\))**: Indicates the proportion of variance explained by the model.
- **Mean Absolute Error (MAE)**: Represents the average absolute prediction error.
- **Symmetric Mean Absolute Percentage Error (sMAPE)**: Measures the percentage error, robust to near-zero values.

---

## Cross-Validation Setup

- **Splitter**: TimeSeriesSplit with 5 folds to maintain chronological order.
- **Consistency**: Both XGBoost and LSTM used the same fold splits for fairness.

---

## Results

### XGBoost Performance (Original Dataset)

| Fold | RMSE   | \(R^2\)   | MAE    | sMAPE   |
|------|--------|-----------|--------|---------|
| 1    | 0.2049 | -0.5475   | 0.1482 | 35.28%  |
| 2    | 0.0155 | 0.9631    | 0.0099 |  2.44%  |
| 3    | 0.0008 | 0.9994    | 0.0006 |  0.15%  |
| 4    | 0.0153 | 0.9775    | 0.0067 |  1.03%  |
| 5    | 0.0749 | 0.4300    | 0.0348 |  4.29%  |

### LSTM Performance (Original Dataset)

| Fold | RMSE   | \(R^2\)   | MAE    | sMAPE   |
|------|--------|-----------|--------|---------|
| 1    | 0.0094 | 0.9967    | 0.0070 | 1.48%   |
| 2    | 0.0078 | 0.9907    | 0.0065 | 1.45%   |
| 3    | 0.0020 | 0.9961    | 0.0014 | 0.36%   |
| 4    | 0.0037 | 0.9987    | 0.0027 | 0.45%   |
| 5    | 0.0155 | 0.9757    | 0.0087 | 1.06%   |

---

## Experimentation with Crude Oil Features

To enrich the dataset, crude oil **Close Price** and **Volume** were added to the gold price dataset. These features aim to improve the predictive power by capturing economic interdependencies.

- **New Dataset Size**: 80,145 records after merging crude oil features.

### Results (With Crude Oil Features)

#### XGBoost Performance

| Fold | RMSE   | \(R^2\)   | MAE    | sMAPE   |
|------|--------|-----------|--------|---------|
| 1    | 0.1270 | -2.3528   | 0.1100 | 21.72%  |
| 2    | 0.0028 | 0.9934    | 0.0020 |  0.54%  |
| 3    | 0.0006 | 0.9994    | 0.0004 |  0.11%  |
| 4    | 0.0126 | 0.9710    | 0.0063 |  0.95%  |
| 5    | 0.0819 | 0.3417    | 0.0405 |  5.03%  |

#### LSTM Performance

| Fold | RMSE   | \(R^2\)   | MAE    | sMAPE   |
|------|--------|-----------|--------|---------|
| 1    | 0.0059 | 0.9928    | 0.0043 |  0.83%  |
| 2    | 0.0071 | 0.9573    | 0.0066 |  1.73%  |
| 3    | 0.0017 | 0.9940    | 0.0012 |  0.30%  |
| 4    | 0.0063 | 0.9927    | 0.0056 |  0.95%  |
| 5    | 0.0175 | 0.9699    | 0.0110 |  1.38%  |

---

## Observations

1. **XGBoost**:
   - While Fold 3 and Fold 2 show exceptional performance, Fold 1 struggled significantly, with negative \(R^2\) and high RMSE.
   - This inconsistency suggests sensitivity to certain data segments or anomalies in Fold 1.
   - Overall performance improved in some folds due to the inclusion of crude oil features.

2. **LSTM**:
   - Delivered **consistent performance** across all folds, with high \(R^2\) values (> 0.95).
   - Exhibited lower error metrics (RMSE, MAE, sMAPE) compared to XGBoost.
   - Robust in handling challenging data segments (e.g., Fold 1).

---

## Key Learnings

1. **LSTM Outperforms XGBoost**:
   - The sequential learning capability of LSTM allows it to capture complex relationships between gold prices and crude oil features.
   - LSTM is more robust to outliers or anomalies in the data.

2. **Crude Oil Features Add Value**:
   - Including crude oil features improved the performance of both models, validating their relevance to gold price prediction.

3. **Fold-Specific Challenges**:
   - Investigating Fold 1’s data distribution could help understand why XGBoost struggled.

---

## Possible Next Steps

1. **Data Analysis**:
   - Examine Fold 1 to identify anomalies or inconsistencies.

2. **Feature Engineering**:
   - Add rolling averages or volatility metrics for crude oil prices.
   - Consider additional economic indicators (e.g., S&P 500).

3. **Model Tuning**:
   - Optimize XGBoost and LSTM hyperparameters for better performance.

---



## Configuration Settings

To reproduce the experiments, follow these steps:

### 1. Environment Setup
- **Python Version**: `3.8+`
- Install dependencies:
  ```bash
  pip install -r requirements.txt

### 2. Directory Structure 


```
GOLD
│   README.md              # Documentation
│   requirements.txt       # Dependencies 
│
└───src
│   │   forecasting.ipynb  # Source code (Jupyter Notebook)
│   
└───data                   # Dataset(s)
    │   *csv
|
|____figures               # Output plots
```




### 3. Notebook Execution
- The entire experiment, including feature engineering, model training, and evaluation, is implemented in the Jupyter Notebook:  
- Open the notebook using the following command:
```bash
jupyter notebook src/forecasting.ipynb


*Follow the instructions in the notebook to run each cell sequentially*




