# Data Flow Documentation

## Overview

This document provides a detailed walkthrough of how data flows through the sales forecasting application, from raw CSV upload to final predictions.

## 1. Data Ingestion Phase

### 1.1 File Upload

```
User Uploads CSV → Streamlit File Handler → Pandas DataFrame
```

**Process:**

- User selects CSV file through Streamlit file uploader
- File is read into pandas DataFrame using `pd.read_csv()`
- Basic validation checks for file format and content

**Key Functions:**

- `st.file_uploader()` - Streamlit file handling
- `pd.read_csv()` - Data loading

### 1.2 Column Mapping

```
Raw DataFrame → AI Analysis → Mapping Dictionary → Standardized Structure
```

**Two-Path Approach:**

**Path A: Gemini AI Mapping (Primary)**

```python
def ask_gemini_for_column_mapping_from_sample(df_sample, api_key):
    # 1. Sample data (100 rows max)
    # 2. Send to Gemini with specific prompt
    # 3. Parse response as Python dict
    # 4. Validate mapping structure
    # 5. Return mapping or None
```

**Path B: Smart Fallback Mapping (Secondary)**

```python
def smart_column_mapping(df):
    # 1. Keyword-based column detection
    # 2. Pattern matching for common names
    # 3. Type inference for data columns
    # 4. Default value assignment
    # 5. Return standardized mapping
```

**Mapping Output:**

```python
{
    'date': 'Date_Column_Name',
    'store': 'Store_Column_Name',
    'item': 'Item_Column_Name',
    'sales': 'Sales_Column_Name'  # or formula like 'Quantity * Price'
}
```

## 2. Data Standardization Phase

### 2.1 Timeseries Conversion

```
Raw Data → Column Processing → Encoding → Standardized DataFrame
```

**Process Flow:**

**Store Processing:**

```python
# Handle missing store column
if not store_col:
    out_df['store'] = 1
    encoding_maps['store'] = {1: 'Default Store'}
else:
    # Encode categorical stores to numeric
    store_codes, store_uniques = pd.factorize(store_data)
    out_df['store'] = store_codes + 1
    encoding_maps['store'] = {i+1: str(val) for i, val in enumerate(store_uniques)}
```

**Item Processing:**

```python
# Similar to store processing
# Creates numeric item codes with mapping dictionary
```

**Date Processing:**

```python
# Convert to datetime with error handling
out_df['date'] = robust_parse_dates(df[date_col])

# Fallback: create default date range if no date column
if not date_col:
    out_df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
```

**Sales Processing:**

```python
# Handle direct sales column
if sales_map in df.columns:
    out_df['sales'] = df[sales_map].fillna(0)

# Handle formula-based sales (e.g., "Quantity * Price")
elif '*' in sales_map:
    parts = sales_map.split('*')
    sales = df[parts[0]].fillna(0)
    for p in parts[1:]:
        sales = sales * df[p].fillna(1)
    out_df['sales'] = sales
```

**Original Feature Preservation:**

```python
# Keep all non-mapped columns as additional features
for col in df.columns:
    if col not in mapped_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            original_features[f'orig_{col}'] = df[col].fillna(df[col].median())
        elif pd.api.types.is_object_dtype(df[col]):
            # Encode categorical columns
            codes, uniques = pd.factorize(df[col].fillna('Unknown'))
            original_features[f'orig_{col}_encoded'] = codes + 1
```

### 2.2 Data Validation & Cleaning

```
Standardized Data → Type Conversion → NaN Handling → Clean Data
```

**Validation Steps:**

1. **Type Conversion**: Ensure correct data types
2. **NaN Handling**: Fill missing values appropriately
3. **Infinite Value Removal**: Replace inf/-inf with NaN
4. **Data Sorting**: Sort by store, item, date for consistency

## 3. Feature Engineering Phase

### 3.1 Data Preparation

```
Clean Data → Grouping → Lag Features → Rolling Features → Calendar Features
```

**Initial Setup:**

```python
# Sort and group data
df = df.sort_values(['store', 'item', 'date'])
grouped = df.groupby(['store', 'item'])
```

### 3.2 Lag Feature Generation

```python
# Create lag features for different time periods
for lag in [1, 2, 3, 7, 14, 30]:
    df[f'sales_lag_{lag}'] = grouped['sales'].shift(lag).fillna(0)
```

**Purpose:** Capture recent sales history and trends

### 3.3 Rolling Feature Generation

```python
# Create rolling statistics for different windows
for window in [3, 7, 14, 30]:
    df[f'sales_rolling_mean_{window}'] = grouped['sales'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    ).fillna(0)

    df[f'sales_rolling_std_{window}'] = grouped['sales'].transform(
        lambda x: x.rolling(window, min_periods=1).std()
    ).fillna(0)

    df[f'sales_rolling_min_{window}'] = grouped['sales'].transform(
        lambda x: x.rolling(window, min_periods=1).min()
    ).fillna(0)

    df[f'sales_rolling_max_{window}'] = grouped['sales'].transform(
        lambda x: x.rolling(window, min_periods=1).max()
    ).fillna(0)
```

**Purpose:** Capture moving averages and volatility patterns

### 3.4 Calendar Feature Generation

```python
# Basic calendar features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# Seasonal indicators
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)

# Trigonometric encoding for smooth seasonal transitions
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
```

**Purpose:** Capture seasonal patterns and cyclical behavior

### 3.5 Expanding Feature Generation

```python
# Cumulative statistics
df['sales_expanding_mean'] = grouped['sales'].transform(
    lambda x: x.expanding().mean()
).fillna(0)

# Price ratio (current vs rolling average)
df['sales_price_ratio'] = grouped['sales'].transform(
    lambda x: x / x.rolling(7, min_periods=1).mean().replace(0, 1)
).fillna(1)
```

**Purpose:** Capture long-term trends and relative performance

## 4. Data Splitting Phase

### 4.1 Train/Validation/Test Split

```
Complete Dataset → Date-Based Split → Three Datasets
```

**Split Logic:**

```python
# Get unique dates and determine split points
unique_dates = out_df['date'].sort_values().unique()
n_dates = len(unique_dates)

# Calculate split sizes (20% each for validation and test)
test_size = min(90, max(1, n_dates // 5))
val_size = min(90, max(1, n_dates // 5))

# Define split boundaries
test_start = unique_dates[-test_size]
val_start = unique_dates[-(test_size + val_size)]

# Create masks
train_mask = out_df['date'] < val_start
val_mask = (out_df['date'] >= val_start) & (out_df['date'] < test_start)
test_mask = out_df['date'] >= test_start

# Split datasets
train_df = out_df[train_mask].copy()
val_df = out_df[val_mask].copy()
test_df = out_df[test_mask].copy()
```

**Purpose:** Ensure temporal integrity and prevent data leakage

## 5. Feature Selection Phase

### 5.1 Feature Collection

```
All Features → Categorization → Selection → Final Feature Set
```

**Feature Categories:**

- **Original Features**: Preserved from input data
- **Lag Features**: Historical sales values
- **Rolling Features**: Moving window statistics
- **Calendar Features**: Time-based indicators
- **Trigonometric Features**: Cyclical encodings
- **Expanding Features**: Cumulative statistics

### 5.2 AI-Powered Feature Selection (Optional)

```python
def gemini_select_best_features(feature_importance_df, api_key):
    # 1. Train quick model for feature importance
    # 2. Send importance ranking to Gemini
    # 3. AI selects optimal feature subset
    # 4. Validate selected features exist
    # 5. Return selected feature list
```

**Selection Criteria:**

- Predictive power (importance score)
- Feature diversity (different types)
- Business logic (interpretability)
- Correlation avoidance

## 6. Model Training Phase

### 6.1 Hyperparameter Optimization

```
Data Sample → AI Analysis → Parameter Ranges → Bayesian Optimization
```

**Process:**

1. **AI Parameter Suggestion**: Gemini analyzes data and suggests parameter ranges
2. **Optuna Optimization**: Bayesian optimization finds best parameters
3. **Validation**: Parameters tested on validation set
4. **Selection**: Best parameters selected based on validation performance

### 6.2 Multiple Model Training

```
Feature Set → Three Model Variants → Performance Comparison → Best Model Selection
```

**Model Variants:**

**Model 1: Original Target**

```python
# Direct sales prediction
y_train = train_features['sales']
model_original = lgb.train(params, train_data, ...)
```

**Model 2: Log-Transformed Target**

```python
# Log-transformed for skewed distributions
y_train_log = np.log1p(y_train)
model_log = lgb.train(params, train_data_log, ...)
# Predictions: np.expm1(y_pred_log)
```

**Model 3: RFE-Pruned Features**

```python
# Recursive Feature Elimination
rfe = RFE(estimator, n_features_to_select=15)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_]
model_rfe = lgb.train(params, train_data_rfe, ...)
```

### 6.3 Model Evaluation

```
Trained Models → Test Predictions → Performance Metrics → Model Selection
```

**Evaluation Metrics:**

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

**Selection Logic:**

```python
# Compare all models and select best based on R² score
best_model_name = max(model_results, key=lambda x: model_results[x]['R²'])
```

## 7. Prediction Generation Phase

### 7.1 Recursive Forecasting

```
Last Observation → Iterative Prediction → Multi-step Forecast → Results
```

**Algorithm:**

```python
def recursive_forecast(model, last_obs, feature_cols, steps=5):
    preds = []
    current = last_obs.copy()

    for i in range(steps):
        # 1. Predict next value
        X = current[feature_cols]
        pred = model.predict(X)[0]
        preds.append(pred)

        # 2. Update lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            lag_col = f'sales_lag_{lag}'
            if lag_col in current.columns:
                if i == 0:
                    current[lag_col] = pred
                else:
                    current[lag_col] = preds[-min(lag, len(preds))]

        # 3. Update rolling features
        for window in [3, 7, 14, 30]:
            roll_col = f'sales_rolling_mean_{window}'
            if roll_col in current.columns:
                vals = preds[-window:] if len(preds) >= window else preds
                current[roll_col] = np.mean(vals)

        # 4. Update expanding features
        if 'sales_expanding_mean' in current.columns:
            current['sales_expanding_mean'] = np.mean(preds)

    return preds
```

### 7.2 Prediction Modes

**Mode 1: Item-Specific Prediction**

- Select specific items
- Generate forecasts for each item
- Display results with item names

**Mode 2: Store-Specific Prediction**

- Select specific stores
- Generate forecasts for all items in stores
- Compare performance across locations

**Mode 3: Item-Store Combination**

- Select specific item-store pairs
- Generate targeted forecasts
- Detailed analysis for specific combinations

**Mode 4: Random Sampling**

- Select random items for quick insights
- Generate sample predictions
- Explore data patterns

**Mode 5: Top-Selling Analysis**

- Forecast all items for extended period
- Rank by total predicted sales
- Strategic planning insights

## 8. Results Export Phase

### 8.1 CSV Export

```
Predictions → DataFrame Conversion → CSV Format → Download
```

**Export Structure:**

```python
{
    'prediction_type': 'specific_item',
    'item_id': 1,
    'item_name': 'Product A',
    'store_id': 1,
    'store_name': 'Store 1',
    'day': 1,
    'predicted_sales': 150.5,
    'total_days': 7
}
```

## Data Flow Summary

```
CSV Upload → Column Mapping → Data Standardization → Feature Engineering →
Data Splitting → Feature Selection → Model Training → Prediction Generation →
Results Export
```

**Key Characteristics:**

- **Temporal Integrity**: Maintains chronological order throughout
- **Error Handling**: Graceful degradation at each step
- **Scalability**: Handles various data sizes and formats
- **Flexibility**: Multiple prediction modes and configurations
- **Robustness**: Fallback mechanisms for AI failures
