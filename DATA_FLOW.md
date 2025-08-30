# Data Flow Documentation

## Overview

This document provides a detailed walkthrough of how data flows through the sales forecasting application, from raw CSV upload to final predictions, including **advanced data-driven hyperparameter optimization**.

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

# Trigonometric features for smooth seasonal transitions
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
```

**Purpose:** Capture seasonal patterns and cyclical time effects

### 3.5 Expanding Feature Generation

```python
# Cumulative statistics
df['sales_expanding_mean'] = grouped['sales'].transform(
    lambda x: x.expanding().mean()
).fillna(0)

# Price ratio (if price data available)
df['sales_price_ratio'] = grouped['sales'].transform(
    lambda x: x / x.rolling(7, min_periods=1).mean().replace(0, 1)
).fillna(1)
```

**Purpose:** Capture long-term trends and price relationships

## 4. Advanced Hyperparameter Optimization Phase

### 4.1 Data Analysis for Hyperparameter Tuning

```
Feature Set → Data Statistics Analysis → AI Parameter Suggestion → Bayesian Optimization
```

**Data Analysis Process:**

```python
def analyze_data_for_hyperparameters(df_sample, features):
    # Calculate dataset characteristics
    n_rows = len(df_sample)
    n_features = len(features)

    # Sales statistics analysis
    sales_stats = df_sample['sales'].describe()
    sales_range = sales_stats['max'] - sales_stats['min']
    sales_std = sales_stats['std']
    zero_ratio = (df_sample['sales'] == 0).mean()

    # Dataset size categorization
    if n_rows < 1000:
        size_category = "small"
        max_leaves_factor = 0.5
        min_data_factor = 2
    elif n_rows < 10000:
        size_category = "medium"
        max_leaves_factor = 0.7
        min_data_factor = 1.5
    else:
        size_category = "large"
        max_leaves_factor = 1.0
        min_data_factor = 1.0

    return {
        'size_category': size_category,
        'n_rows': n_rows,
        'n_features': n_features,
        'sales_stats': sales_stats,
        'sales_range': sales_range,
        'sales_std': sales_std,
        'zero_ratio': zero_ratio,
        'max_leaves_factor': max_leaves_factor,
        'min_data_factor': min_data_factor
    }
```

### 4.2 AI-Powered Hyperparameter Suggestion

**Enhanced Prompt Engineering:**

```python
def gemini_suggest_lgbm_param_ranges(df_sample, selected_features, api_key):
    # Analyze data characteristics
    data_analysis = analyze_data_for_hyperparameters(df_sample, selected_features)

    # Create comprehensive prompt with data-driven guidance
    prompt = f"""
    You are an expert data scientist specializing in sales forecasting with LightGBM.

    **DATA ANALYSIS:**
    - Dataset size: {data_analysis['n_rows']} rows, {data_analysis['n_features']} features ({data_analysis['size_category']} dataset)
    - Sales statistics: min={data_analysis['sales_stats']['min']:.2f}, max={data_analysis['sales_stats']['max']:.2f}, mean={data_analysis['sales_stats']['mean']:.2f}, std={data_analysis['sales_stats']['std']:.2f}
    - Sales range: {data_analysis['sales_range']:.2f}, coefficient of variation: {cv_str}
    - Zero sales ratio: {data_analysis['zero_ratio']:.1%} {'(sparse data - consider higher regularization)' if data_analysis['zero_ratio'] > 0.3 else ''}

    **SALES FORECASTING CONTEXT:**
    - Time series regression with seasonality and trends
    - High variance data with potential outliers
    - Balance between pattern capture and overfitting

    **HYPERPARAMETER GUIDANCE:**
    [Detailed parameter-specific guidance based on data characteristics]

    **OUTPUT FORMAT:**
    Return ONLY a valid Python dictionary for Optuna optimization.
    """

    # Send to Gemini and process response
    return process_ai_response(prompt, api_key)
```

**Key Improvements:**

- **Data-Driven Analysis**: Uses actual dataset characteristics
- **Domain-Specific Guidance**: Sales forecasting specific recommendations
- **Adaptive Parameters**: Scales based on dataset size and characteristics
- **Sparsity Awareness**: Adjusts regularization based on zero sales ratio

### 4.3 Bayesian Optimization with Optuna

```python
def optimize_hyperparameters(param_ranges, X_train, y_train, X_val, y_val):
    def objective(trial):
        # Sample parameters from AI-suggested ranges
        params = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, (list, tuple)):
                if all(isinstance(x, int) for x in param_range):
                    params[param_name] = trial.suggest_int(param_name, min(param_range), max(param_range))
                elif all(isinstance(x, float) for x in param_range):
                    params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range), log=True if min(param_range) > 0 else False)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)

        # Add fixed parameters
        params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'seed': 42
        })

        # Train model and evaluate
        try:
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        except Exception as e:
            return float('inf')  # Penalize failed trials

    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    return study.best_trial.params
```

## 5. Model Training Phase

### 5.1 Multiple Model Variants

```
Feature Set → Model 1 (Original) → Model 2 (Log-Transformed) → Model 3 (AI-Pruned) → Comparison
```

**Model Training Process:**

```python
def train_multiple_models(train_features, val_features, test_features, all_feature_cols, best_params):
    models = {}

    # Model 1: Original Target (All Features)
    X_train = train_features[all_feature_cols]
    X_val = val_features[all_feature_cols]
    X_test = test_features[all_feature_cols]
    y_train = train_features['sales']
    y_val = val_features['sales']
    y_test = test_features['sales']

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model_original = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Model 2: Log-Transformed Target
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    train_data_log = lgb.Dataset(X_train, label=y_train_log)
    valid_data_log = lgb.Dataset(X_val, label=y_val_log, reference=train_data_log)

    model_log = lgb.train(
        best_params,
        train_data_log,
        num_boost_round=1000,
        valid_sets=[valid_data_log],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Model 3: AI-Pruned Features (if available)
    if valid_selected_features:
        X_train_ai = train_features[valid_selected_features]
        X_val_ai = val_features[valid_selected_features]
        X_test_ai = test_features[valid_selected_features]

        train_data_ai = lgb.Dataset(X_train_ai, label=y_train)
        valid_data_ai = lgb.Dataset(X_val_ai, label=y_val, reference=train_data_ai)

        model_ai = lgb.train(
            best_params,
            train_data_ai,
            num_boost_round=1000,
            valid_sets=[valid_data_ai],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

    return {
        'original': model_original,
        'log_transformed': model_log,
        'ai_pruned': model_ai if valid_selected_features else None
    }
```

### 5.2 Model Evaluation and Selection

```python
def evaluate_and_select_best_model(models, X_test, y_test, all_feature_cols, valid_selected_features):
    results = {}

    # Evaluate original model
    y_pred_original = models['original'].predict(X_test[all_feature_cols])
    results['Original Target (All Features)'] = {
        'MSE': mean_squared_error(y_test, y_pred_original),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_original)),
        'R²': r2_score(y_test, y_pred_original)
    }

    # Evaluate log-transformed model
    y_pred_log = models['log_transformed'].predict(X_test[all_feature_cols])
    y_pred_log_transformed = np.expm1(y_pred_log)
    results['Log-Transformed Target (All Features)'] = {
        'MSE': mean_squared_error(y_test, y_pred_log_transformed),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_log_transformed)),
        'R²': r2_score(y_test, y_pred_log_transformed)
    }

    # Evaluate AI-pruned model (if available)
    if models['ai_pruned'] and valid_selected_features:
        y_pred_ai = models['ai_pruned'].predict(X_test[valid_selected_features])
        results['AI-Pruned Features'] = {
            'MSE': mean_squared_error(y_test, y_pred_ai),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ai)),
            'R²': r2_score(y_test, y_pred_ai)
        }

    # Select best model based on R² score
    best_model_name = max(results, key=lambda x: results[x]['R²'])

    return results, best_model_name, models[best_model_name.lower().split()[0]]
```

## 6. Prediction Generation Phase

### 6.1 Recursive Forecasting Algorithm

```
Last Observation → Feature Update → Prediction → Feature Update → Next Prediction
```

**Recursive Forecasting Process:**

```python
def recursive_forecast(model, last_obs, feature_cols, steps=5):
    predictions = []
    current_state = last_obs.copy()

    for step in range(steps):
        # 1. Make prediction using current features
        X_current = current_state[feature_cols]
        prediction = model.predict(X_current)[0]
        predictions.append(prediction)

        # 2. Update lag features with new prediction
        for lag in [1, 2, 3, 7, 14, 30]:
            lag_col = f'sales_lag_{lag}'
            if lag_col in current_state.columns:
                if step == 0:
                    current_state[lag_col] = prediction
                else:
                    current_state[lag_col] = predictions[-min(lag, len(predictions))]

        # 3. Update rolling features
        for window in [3, 7, 14, 30]:
            roll_col = f'sales_rolling_mean_{window}'
            if roll_col in current_state.columns:
                recent_predictions = predictions[-window:] if len(predictions) >= window else predictions
                current_state[roll_col] = np.mean(recent_predictions)

        # 4. Update expanding features
        if 'sales_expanding_mean' in current_state.columns:
            current_state['sales_expanding_mean'] = np.mean(predictions)

        # 5. Update price ratio (if applicable)
        if 'sales_price_ratio' in current_state.columns and 'sales_rolling_mean_7' in current_state.columns:
            rolling_mean_val = current_state['sales_rolling_mean_7'].iloc[0]
            current_state['sales_price_ratio'] = prediction / rolling_mean_val if rolling_mean_val != 0 else 1.0

    return predictions
```

### 6.2 Multi-Mode Prediction Generation

**Prediction Modes:**

1. **Item-Specific Predictions:**

   ```python
   def predict_for_items(selected_items, days, model, test_features, feature_cols, encoding_maps):
       predictions = []
       for item_name in selected_items:
           item_id = get_item_id(item_name, encoding_maps)
           item_data = test_features[test_features['item'] == item_id]
           if len(item_data) > 0:
               last_obs = item_data.sort_values('date').iloc[[-1]].copy()
               preds = recursive_forecast(model, last_obs, feature_cols, steps=days)
               predictions.append({
                   'item': item_name,
                   'predictions': preds
               })
       return predictions
   ```

2. **Store-Specific Predictions:**

   ```python
   def predict_for_stores(selected_stores, days, model, test_features, feature_cols, encoding_maps):
       predictions = []
       for store_name in selected_stores:
           store_id = get_store_id(store_name, encoding_maps)
           store_data = test_features[test_features['store'] == store_id]
           store_items = store_data['item'].unique()

           for item in store_items:
               item_data = store_data[store_data['item'] == item]
               last_obs = item_data.sort_values('date').iloc[[-1]].copy()
               preds = recursive_forecast(model, last_obs, feature_cols, steps=days)
               predictions.append({
                   'store': store_name,
                   'item': get_item_name(item, encoding_maps),
                   'predictions': preds
               })
       return predictions
   ```

3. **Top-Selling Analysis:**
   ```python
   def analyze_top_selling_products(days, model, test_features, feature_cols, encoding_maps):
       all_items = test_features['item'].unique()
       item_predictions = []

       for item in all_items:
           item_data = test_features[test_features['item'] == item]
           if len(item_data) > 0:
               last_obs = item_data.sort_values('date').iloc[[-1]].copy()
               preds = recursive_forecast(model, last_obs, feature_cols, steps=days)

               total_sales = sum(preds)
               avg_daily_sales = total_sales / days

               item_predictions.append({
                   'item_id': item,
                   'item_name': get_item_name(item, encoding_maps),
                   'total_predicted_sales': total_sales,
                   'avg_daily_sales': avg_daily_sales,
                   'predictions': preds
               })

       # Sort by total predicted sales
       item_predictions.sort(key=lambda x: x['total_predicted_sales'], reverse=True)
       return item_predictions
   ```

## 7. Results Processing and Export

### 7.1 Prediction Formatting

```python
def format_predictions_for_export(predictions):
    formatted_rows = []

    for pred in predictions:
        for day, value in enumerate(pred['predictions'], 1):
            row = {
                'prediction_type': pred.get('type', 'unknown'),
                'item_id': pred.get('item_id', pred.get('item')),
                'item_name': pred.get('item_name', f"Item {pred.get('item')}"),
                'store_id': pred.get('store_id', pred.get('store')),
                'store_name': pred.get('store_name', f"Store {pred.get('store')}"),
                'day': day,
                'predicted_sales': value,
                'total_days': len(pred['predictions'])
            }
            formatted_rows.append(row)

    return pd.DataFrame(formatted_rows)
```

### 7.2 CSV Export

```python
def export_predictions_to_csv(predictions):
    df = format_predictions_for_export(predictions)
    csv_data = df.to_csv(index=False).encode('utf-8')
    return csv_data
```

## 8. Error Handling and Fallbacks

### 8.1 Graceful Degradation

**AI Service Failures:**

```python
def handle_ai_failure(operation, error):
    if operation == 'column_mapping':
        return smart_column_mapping(df)
    elif operation == 'hyperparameter_suggestion':
        return get_default_lgbm_params()
    elif operation == 'feature_selection':
        return all_feature_cols  # Use all features
    else:
        raise Exception(f"Unknown operation: {operation}")
```

**Data Processing Failures:**

```python
def handle_data_processing_error(error, data):
    # Clean data and retry
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0)
    return data
```

### 8.2 Validation and Sanity Checks

```python
def validate_predictions(predictions):
    # Check for negative predictions
    if any(p < 0 for pred in predictions for p in pred['predictions']):
        st.warning("Negative predictions detected - applying corrections")
        predictions = apply_prediction_corrections(predictions)

    # Check for unrealistic values
    if any(p > 1000000 for pred in predictions for p in pred['predictions']):
        st.warning("Unrealistically high predictions detected")

    return predictions
```

## 9. Performance Monitoring

### 9.1 Timing and Metrics

```python
def monitor_performance():
    metrics = {
        'data_processing_time': time.time() - start_time,
        'feature_engineering_time': feature_time - data_time,
        'model_training_time': training_time - feature_time,
        'prediction_time': prediction_time - training_time,
        'total_time': time.time() - start_time
    }

    return metrics
```

### 9.2 Memory Usage Tracking

```python
def track_memory_usage():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024   # MB
    }
```

---

This comprehensive data flow ensures robust, efficient, and accurate sales forecasting with **advanced AI integration and data-driven optimization** at every step.
