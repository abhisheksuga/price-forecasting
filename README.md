

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

### XGBoost Performance


| Fold | RMSE   | \(R^2\)   | MAE    | sMAPE   |
|------|--------|-----------|--------|---------|
| 1    | 0.2049 | -0.5475   | 0.1482 | 35.28%  |
| 2    | 0.0155 | 0.9631    | 0.0099 |  2.44%  |
| 3    | 0.0008 | 0.9994    | 0.0006 |  0.15%  |
| 4    | 0.0153 | 0.9775    | 0.0067 |  1.03%  |
| 5    | 0.0749 | 0.4300    | 0.0348 |  4.29%  |


### LSTM Performance
| Fold | RMSE   | \(R^2\)   | MAE    | sMAPE   |
|------|--------|-----------|--------|---------|
| 1    | 0.0094 | 0.9967    | 0.0070 | 1.48%   |
| 2    | 0.0078 | 0.9907    | 0.0065 | 1.45%   |
| 3    | 0.0020 | 0.9961    | 0.0014 | 0.36%   |
| 4    | 0.0037 | 0.9987    | 0.0027 | 0.45%   |
| 5    | 0.0155 | 0.9757    | 0.0087 | 1.06%   |

---

## Observations

- **XGBoost**:
  - Demonstrated strong performance with engineered features.
  - Relies heavily on lagged values and rolling statistics for temporal context.

- **LSTM**:
  - Performed exceptionally well with raw sequential data, learning patterns directly.
  - Slightly underperformed in Fold 5, indicating possible variability in certain data segments.

---

## Possible Next Steps

1. **Hyperparameter Tuning**:
   - Optimize LSTM architecture and parameters (e.g., units, dropout rate, learning rate).
   - Tune XGBoost hyperparameters for comparison(if required).

2. **Model Benchmarking**:
   - Experiment with other deep learning models like GRU, 1D CNN, or Transformer-based models.

3. **Test on Unseen Data**:
   - Evaluate the best-performing model on a separate holdout dataset to assess robustness.

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




