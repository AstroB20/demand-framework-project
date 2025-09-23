import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import google.generativeai as genai
import time
import ast
import sys
from sklearn.metrics import mean_squared_error, r2_score
import random
from sklearn.feature_selection import RFE
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterSampler
try:
    import optuna
except ImportError:
    optuna = None
import os
from dotenv import load_dotenv
load_dotenv()

# Gemini setup constants
MODEL_NAME = "gemini-1.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 2
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')  # Load from environment variable, default to empty string

# --- Gemini Functions ---
def ask_gemini_for_column_mapping_from_sample(df_sample, api_key):
    sample_csv = df_sample.to_csv(index=False)
    prompt = (
        f"You are given a sales data sample (CSV):\n{sample_csv}\n\n"
        "Map the following standard roles to the best-matching column names: date, store, item, sales. "
        "For the 'sales' role, if a column named 'sales' exists, use it. "
        "If not, prefer a column related to quantity (e.g., quantity, qty, units, volume) as the sales column. "
        "Only if neither a sales nor a quantity-related column is available, provide a formula using column names (e.g., 'Quantity * Price'). "
        "If a role is missing, set it to a default (e.g., store=1 for all rows). "
        "Respond only with a valid Python dict, no explanations, no markdown, no code blocks. "
        "Keys should be the expected columns, values should be the input column name or a formula."
    )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            code = response.text.strip()
            mapping = ast.literal_eval(code)
            if isinstance(mapping, dict):
                return mapping
        except Exception as e:
            st.warning(f"Gemini column mapping attempt {attempt + 1} failed: {str(e)}")
            time.sleep(RETRY_DELAY)
    return None

def gemini_suggest_lgbm_param_ranges(df_sample, selected_features, api_key):
    sample_csv = df_sample[selected_features + ['sales']].to_csv(index=False)
    n_rows = len(df_sample)
    n_features = len(selected_features)
    
    # Analyze data characteristics
    sales_stats = df_sample['sales'].describe()
    sales_range = sales_stats['max'] - sales_stats['min']
    sales_std = sales_stats['std']
    has_zeros = (df_sample['sales'] == 0).sum() > 0
    zero_ratio = (df_sample['sales'] == 0).mean()
    
    # Determine data size category
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
    
    # Calculate coefficient of variation safely
    cv = sales_std/sales_stats['mean'] if sales_stats['mean'] > 0 else float('nan')
    cv_str = f"{cv:.2f}" if not np.isnan(cv) else "N/A"
    
    prompt = f"""You are an expert data scientist specializing in sales forecasting with LightGBM. Analyze this sales data and suggest optimal hyperparameter ranges for Bayesian optimization.

**DATA ANALYSIS:**
- Dataset size: {n_rows} rows, {n_features} features ({size_category} dataset)
- Sales statistics: min={sales_stats['min']:.2f}, max={sales_stats['max']:.2f}, mean={sales_stats['mean']:.2f}, std={sales_stats['std']:.2f}
- Sales range: {sales_range:.2f}, coefficient of variation: {cv_str}
- Zero sales ratio: {zero_ratio:.1%} {'(sparse data - consider higher regularization)' if zero_ratio > 0.3 else ''}

**FEATURES:**
{selected_features}

**SALES FORECASTING CONTEXT:**
- This is a time series regression problem with potential seasonality and trends
- Sales data often has high variance and may contain outliers
- Need to balance between capturing patterns and avoiding overfitting
- Computational efficiency matters for production deployment

**HYPERPARAMETER GUIDANCE:**

1. **num_leaves**: Controls tree complexity. For {size_category} datasets:
   - Small datasets: 15-63 (prevent overfitting)
   - Medium datasets: 31-127 (balance complexity)
   - Large datasets: 63-255 (capture patterns)
   - Consider: {int(31 * max_leaves_factor)}-{int(127 * max_leaves_factor)}

2. **learning_rate**: Critical for convergence and generalization:
   - Start conservative: 0.01-0.1 (better generalization)
   - For noisy data: 0.005-0.05 (more stable)
   - For clean patterns: 0.05-0.2 (faster convergence)

3. **n_estimators**: Number of boosting rounds:
   - Small datasets: 100-500
   - Medium datasets: 200-1000
   - Large datasets: 500-2000
   - Higher for lower learning rates

4. **min_data_in_leaf**: Prevents overfitting on small samples:
   - Small datasets: 10-50
   - Medium datasets: 20-100
   - Large datasets: 50-200
   - Consider: {int(20 * min_data_factor)}-{int(100 * min_data_factor)}

5. **feature_fraction**: Reduces overfitting, especially with many features:
   - Few features (<20): 0.8-1.0
   - Many features (20-50): 0.6-0.9
   - Many features (>50): 0.5-0.8

6. **bagging_fraction**: Subsampling for regularization:
   - Small datasets: 0.7-0.9
   - Medium/Large datasets: 0.6-0.8

7. **lambda_l1/lambda_l2**: Regularization strength:
   - For sparse data (high zero ratio): 0.1-1.0
   - For dense data: 0.01-0.1
   - L1 for feature selection, L2 for general regularization

8. **max_depth**: Tree depth limit:
   - Conservative: 6-10
   - Balanced: 8-15
   - Aggressive: 12-20

**SPECIFIC RECOMMENDATIONS FOR THIS DATA:**
- Dataset size suggests {size_category} category parameters
- {'High' if zero_ratio > 0.3 else 'Moderate'} sparsity suggests {'stronger' if zero_ratio > 0.3 else 'moderate'} regularization
- {'High' if cv > 1 else 'Moderate'} variance suggests {'conservative' if cv > 1 else 'balanced'} learning rates

**OUTPUT FORMAT:**
Return ONLY a valid Python dictionary with hyperparameter ranges suitable for Optuna optimization.
Use lists for categorical parameters, tuples for numeric ranges.
Example format:
{{
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 500, 1000],
    'min_data_in_leaf': [20, 50, 100],
    'feature_fraction': [0.7, 0.8, 0.9],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'lambda_l1': [0.01, 0.1, 0.5],
    'lambda_l2': [0.01, 0.1, 0.5],
    'max_depth': [8, 12, 16]
}}

**CRITICAL:** Return only the dictionary, no explanations, no markdown, no code blocks."""
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            code = response.text.strip()
            if '{' in code and '}' in code:
                code = code[code.find('{'):code.rfind('}')+1]
            code = code.replace('```python', '').replace('```', '').strip()
            param_ranges = ast.literal_eval(code)
            if isinstance(param_ranges, dict):
                return param_ranges
        except Exception as e:
            st.warning(f"Gemini hyperparameter range suggestion attempt {attempt + 1} failed: {str(e)}")
            time.sleep(RETRY_DELAY)
    return None

def gemini_select_best_features(feature_importance_df, api_key):
    """Gemini analyzes feature importance and suggests which features to keep"""
    importance_text = feature_importance_df.to_string()
    prompt = f"""
    You are a data scientist analyzing feature importance for sales forecasting.
    
    Given these feature importances (higher = more important):
    {importance_text}
    
    Your task: Select the top 15-20 most predictive features for sales forecasting.
    
    **Selection Criteria:**
    1. **Predictive Power**: Higher importance = better
    2. **Feature Diversity**: Include different types (lags, rolling, calendar, original)
    3. **Business Logic**: Prefer interpretable features
    4. **Correlation**: Avoid highly correlated features
    
    **Required Feature Types (if available):**
    - At least 2-3 lag features (sales_lag_*)
    - At least 2-3 rolling features (sales_rolling_*)
    - At least 2-3 calendar features (year, month, dayofweek, etc.)
    - At least 1-2 original features (orig_*)
    - At least 1 expanding feature (sales_expanding_*)
    
    **Output Format:**
    Return ONLY a Python list of feature names, no explanations, no markdown.
    Example: ['sales_lag_1', 'sales_rolling_mean_7', 'year', 'orig_price', ...]
    
    **Important:** Return exactly the feature names as they appear in the data.
    """
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt)
            code = response.text.strip()
            selected = ast.literal_eval(code)
            if isinstance(selected, list) and all(isinstance(f, str) for f in selected):
                return selected
        except Exception as e:
            st.warning(f"Gemini feature selection attempt {attempt + 1} failed: {str(e)}")
            time.sleep(RETRY_DELAY)
    return None

# --- Universal Manual Fallback Functions ---
def smart_column_mapping(df):
    """Universal column mapping that works with any data format"""
    mapping = {}
    columns = [col.lower() for col in df.columns]
    
    # Find date column
    date_keywords = ['date', 'time', 'timestamp', 'day', 'month', 'year']
    date_col = None
    for keyword in date_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                date_col = df.columns[i]
                break
        if date_col:
            break
    
    # Find item column
    item_keywords = ['item', 'product', 'sku', 'goods', 'commodity', 'article']
    item_col = None
    for keyword in item_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                item_col = df.columns[i]
                break
        if item_col:
            break
    
    # Find store column
    store_keywords = ['store', 'shop', 'location', 'branch', 'outlet', 'market']
    store_col = None
    for keyword in store_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                store_col = df.columns[i]
                break
        if store_col:
            break
    
    # Find sales column
    sales_keywords = ['sales', 'quantity', 'amount', 'volume', 'units', 'revenue']
    sales_col = None
    for keyword in sales_keywords:
        for i, col in enumerate(columns):
            if keyword in col:
                sales_col = df.columns[i]
                break
        if sales_col:
            break
    
    # If no sales column found, try numeric columns
    if not sales_col:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Exclude potential date columns (year, month, etc.)
            exclude_keywords = ['year', 'month', 'day', 'week']
            for col in numeric_cols:
                if not any(keyword in col.lower() for keyword in exclude_keywords):
                    sales_col = col
                    break
    
    mapping['date'] = date_col
    mapping['item'] = item_col
    mapping['store'] = store_col
    mapping['sales'] = sales_col
    
    return mapping

def robust_parse_dates(series):
    try:
        return pd.to_datetime(series, dayfirst=True, errors='coerce')
    except Exception:
        return series.astype(str)

def convert_to_timeseries(df, mapping):
    out_df = pd.DataFrame()
    encoding_maps = {}
    
    # Handle store column
    store_col = mapping.get('store')
    if (not store_col or store_col == mapping.get('item') or store_col not in df.columns):
        out_df['store'] = 1
        encoding_maps['store'] = {1: 'Default Store'}
    else:
        # Handle NaN values in store column
        store_data = df[store_col].fillna('Unknown Store')
        store_codes, store_uniques = pd.factorize(store_data)
        out_df['store'] = store_codes + 1
        encoding_maps['store'] = {i+1: str(val) for i, val in enumerate(store_uniques)}
    
    # Handle item column - create default if missing
    item_col = mapping.get('item')
    if not item_col or item_col not in df.columns:
        # Create a default item column
        out_df['item'] = 1
        encoding_maps['item'] = {1: 'Default Item'}
    else:
        # Handle NaN values in item column
        item_data = df[item_col].fillna('Unknown Item')
        item_codes, item_uniques = pd.factorize(item_data)
        out_df['item'] = item_codes + 1
        encoding_maps['item'] = {i+1: str(val) for i, val in enumerate(item_uniques)}
    
    # Handle date column
    date_col = mapping.get('date')
    if date_col and date_col in df.columns:
        out_df['date'] = robust_parse_dates(df[date_col])
    else:
        # Create a default date range if no date column
        out_df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
    
    # Handle sales column
    sales_map = mapping.get('sales')
    if sales_map and isinstance(sales_map, str) and '*' in sales_map:
        try:
            parts = [p.strip() for p in sales_map.split('*')]
            if all(p in df.columns for p in parts):
                sales = df[parts[0]].fillna(0)
                for p in parts[1:]:
                    sales = sales * df[p].fillna(1)
                out_df['sales'] = sales
            else:
                out_df['sales'] = 0
        except Exception:
            out_df['sales'] = 0
    elif sales_map and isinstance(sales_map, str) and sales_map in df.columns:
        out_df['sales'] = df[sales_map].fillna(0)
    else:
        # Create default sales if no sales column found
        out_df['sales'] = 0
    
    # Preserve ALL other columns that might be useful
    original_features = {}
    for col in df.columns:
        # Skip only the columns we've already mapped to core roles
        mapped_cols = [mapping.get('date'), mapping.get('store'), mapping.get('item'), mapping.get('sales')]
        mapped_cols = [col for col in mapped_cols if col is not None]  # Remove None values
        
        if col not in mapped_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Keep numeric columns as-is, handle NaN values
                original_features[f'orig_{col}'] = df[col].fillna(df[col].median() if df[col].median() is not None else 0)
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Encode categorical columns, handle NaN values
                cat_data = df[col].fillna('Unknown')
                codes, uniques = pd.factorize(cat_data)
                original_features[f'orig_{col}_encoded'] = codes + 1
                encoding_maps[f'orig_{col}'] = {i+1: str(val) for i, val in enumerate(uniques)}
            else:
                # Handle other data types (datetime, etc.) by converting to numeric
                try:
                    # Try to convert to numeric
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_data.isna().all():  # If conversion was successful
                        original_features[f'orig_{col}'] = numeric_data.fillna(numeric_data.median() if numeric_data.median() is not None else 0)
                    else:
                        # If conversion failed, treat as categorical
                        cat_data = df[col].astype(str).fillna('Unknown')
                        codes, uniques = pd.factorize(cat_data)
                        original_features[f'orig_{col}_encoded'] = codes + 1
                        encoding_maps[f'orig_{col}'] = {i+1: str(val) for i, val in enumerate(uniques)}
                except Exception:
                    # If all else fails, treat as categorical
                    cat_data = df[col].astype(str).fillna('Unknown')
                    codes, uniques = pd.factorize(cat_data)
                    original_features[f'orig_{col}_encoded'] = codes + 1
                    encoding_maps[f'orig_{col}'] = {i+1: str(val) for i, val in enumerate(uniques)}
    
    # Add original features to output
    for col_name, col_data in original_features.items():
        out_df[col_name] = col_data
    
    # Safe type conversion with NaN handling
    out_df['store'] = out_df['store'].fillna(1).astype(int)
    out_df['item'] = out_df['item'].fillna(1).astype(int)
    out_df['sales'] = out_df['sales'].fillna(0).astype(float)
    
    return out_df, encoding_maps

def manual_feature_engineering(df):
    # Clean the data first
    df = df.copy()
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Ensure sales is numeric and handle NaN
    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    
    # Ensure store and item are integers
    df['store'] = pd.to_numeric(df['store'], errors='coerce').fillna(1).astype(int)
    df['item'] = pd.to_numeric(df['item'], errors='coerce').fillna(1).astype(int)
    
    # Sort and group
    df = df.sort_values(['store', 'item', 'date'])
    grouped = df.groupby(['store', 'item'])
    
    # Create lag features with NaN handling (no leakage - only past data)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'sales_lag_{lag}'] = grouped['sales'].shift(lag).fillna(0)
    
    # Create rolling features with NaN handling (no leakage - only past data)
    for window in [3, 7, 14, 30]:
        df[f'sales_rolling_mean_{window}'] = grouped['sales'].transform(lambda x: x.rolling(window, min_periods=1).mean()).fillna(0)
        df[f'sales_rolling_std_{window}'] = grouped['sales'].transform(lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        df[f'sales_rolling_min_{window}'] = grouped['sales'].transform(lambda x: x.rolling(window, min_periods=1).min()).fillna(0)
        df[f'sales_rolling_max_{window}'] = grouped['sales'].transform(lambda x: x.rolling(window, min_periods=1).max()).fillna(0)
    
    # Create expanding features with NaN handling (no leakage - cumulative past data)
    df['sales_expanding_mean'] = grouped['sales'].transform(lambda x: x.expanding().mean()).fillna(0)
    
    # Create price ratio with safe division (no leakage - uses past rolling mean)
    df['sales_price_ratio'] = grouped['sales'].transform(
        lambda x: x / x.rolling(7, min_periods=1).mean().replace(0, 1)
    ).fillna(1)
    
    # Create calendar features (no leakage - deterministic from date)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)

    # --- Additional Hard-Codeable Features with Error Handling ---
    try:
        # Days since last sale
        df['days_since_last_sale'] = grouped['sales'].apply(lambda x: x.cumsum().where(x == 0).ffill().fillna(0).astype(int))
    except Exception:
        pass
    try:
        # Days until next sale (reverse, then forward fill, then reverse back)
        df['days_until_next_sale'] = grouped['sales'].apply(lambda x: x[::-1].cumsum().where(x[::-1] == 0).ffill().fillna(0)[::-1].astype(int))
    except Exception:
        pass
    try:
        # Cumulative sales
        df['cumulative_sales'] = grouped['sales'].cumsum()
    except Exception:
        pass
    try:
        # Sales moving average deviation (7-day)
        df['sales_ma7_dev'] = df['sales'] - grouped['sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    except Exception:
        pass
    # Promotion/Discount features
    if 'discount' in df.columns:
        try:
            df['is_discount'] = (df['discount'] > 0).astype(int)
        except Exception:
            pass
        try:
            df['discount_lag_1'] = grouped['discount'].shift(1).fillna(0)
        except Exception:
            pass
    # Price features
    if 'price' in df.columns:
        try:
            df['price_change'] = grouped['price'].diff().fillna(0)
        except Exception:
            pass
        try:
            df['price_to_avg'] = df['price'] / grouped['price'].expanding().mean().replace(0, 1)
        except Exception:
            pass
    # Stock/Inventory features
    if 'stock' in df.columns:
        try:
            df['is_stockout'] = (df['stock'] == 0).astype(int)
        except Exception:
            pass
    
    # Final cleanup - replace any remaining NaN or infinite values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def get_default_lgbm_params():
    return {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'early_stopping_rounds': 10,
        'seed': 42,
        'verbose': -1
    }

def recursive_forecast(model, last_obs, feature_cols, steps=5):
    preds = []
    current = last_obs.copy()
    for i in range(steps):
        X = current[feature_cols]
        pred = model.predict(X)[0]
        preds.append(pred)
        for lag in [1, 2, 3, 7, 14, 30]:
            lag_col = f'sales_lag_{lag}'
            if lag_col in current.columns:
                if i == 0:
                    current[lag_col] = pred
                else:
                    current[lag_col] = preds[-min(lag, len(preds))]
        for window in [3, 7, 14, 30]:
            roll_col = f'sales_rolling_mean_{window}'
            if roll_col in current.columns:
                vals = preds[-window:] if len(preds) >= window else preds
                current[roll_col] = np.mean(vals)
        if 'sales_expanding_mean' in current.columns:
            current['sales_expanding_mean'] = np.mean(preds)
        if 'sales_price_ratio' in current.columns and 'sales_rolling_mean_7' in current.columns:
            rolling_mean_val = current['sales_rolling_mean_7'].iloc[0] if hasattr(current['sales_rolling_mean_7'], 'iloc') else current['sales_rolling_mean_7']
            current['sales_price_ratio'] = pred / rolling_mean_val if rolling_mean_val != 0 else 1.0
    return preds

def save_predictions_to_csv(predictions):
    rows = []
    for pred in predictions:
        for i, value in enumerate(pred['predictions'], 1):
            row = {
                'prediction_type': pred['type'],
                'item_id': pred['item'],
                'item_name': pred.get('item_name', f"Item {pred['item']}"),
                'store_id': pred['store'],
                'store_name': pred.get('store_name', f"Store {pred['store']}"),
                'day': i,
                'predicted_sales': value,
                'total_days': pred['days']
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

# --- Streamlit App Logic ---
st.set_page_config(page_title="Sales Forecasting App", layout="wide")
st.title("🛒 Sales Forecasting Streamlit App")

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Gemini API Key
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
# Remove sidebar input for API key
use_gemini = st.sidebar.checkbox("Use Gemini AI", value=True, help="Enable Gemini AI for automated column mapping, feature engineering, and hyperparameter optimization")

# Gemini toggles for each step
if use_gemini and GEMINI_API_KEY:
    st.sidebar.subheader("Gemini Features")
    use_gemini_mapping = st.sidebar.checkbox("Gemini Column Mapping", value=True)
    use_gemini_features = st.sidebar.checkbox("Gemini Feature Engineering", value=True)
    use_gemini_selection = st.sidebar.checkbox("Gemini Feature Selection", value=True)
    use_gemini_params = st.sidebar.checkbox("Gemini Hyperparameters", value=True)
else:
    use_gemini_mapping = False
    use_gemini_features = False
    use_gemini_selection = False
    use_gemini_params = False


st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your sales data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("## Data Preview", df.head())
    st.success("Data loaded! Proceed to model training and prediction.")

    # --- Pipeline ---
    with st.spinner("Mapping columns and preparing data..."):
        if use_gemini_mapping and GEMINI_API_KEY:
            st.info("🤖 Using Gemini for column mapping...")
            df_sample = df.sample(n=min(100, len(df)), random_state=42)
            mapping = ask_gemini_for_column_mapping_from_sample(df_sample, GEMINI_API_KEY)
            if mapping:
                st.success("✅ Gemini column mapping successful!")
            else:
                st.warning("⚠️ Gemini column mapping failed, using smart mapping...")
                mapping = smart_column_mapping(df)
        else:
            st.info("🔧 Using smart column mapping...")
            mapping = smart_column_mapping(df)
        
        # Show mapping results
        st.write("**Column Mapping:**", mapping)
        
        if mapping['date']:
            df[mapping['date']] = pd.to_datetime(df[mapping['date']], dayfirst=True, errors='coerce')
        out_df, encoding_maps = convert_to_timeseries(df, mapping)
        
        # Show what additional columns were preserved
        preserved_columns = [col for col in out_df.columns if col.startswith('orig_')]
        if preserved_columns:
            st.write(f"**Preserved {len(preserved_columns)} additional columns:**")
            for col in preserved_columns:
                original_col = col.replace('orig_', '').replace('_encoded', '')
                if col.endswith('_encoded'):
                    unique_values = len(encoding_maps.get(col, {}))
                    st.write(f"  - {original_col} (categorical, {unique_values} unique values)")
                else:
                    st.write(f"  - {original_col} (numeric)")
        else:
            st.write("**No additional columns preserved**")
        out_df = out_df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
        unique_dates = out_df['date'].sort_values().unique()
        n_dates = len(unique_dates)
        test_size = min(90, max(1, n_dates // 5))
        val_size = min(90, max(1, n_dates // 5))
        test_start = unique_dates[-test_size]
        val_start = unique_dates[-(test_size + val_size)]
        
        # Ensure temporal integrity - no future data leakage
        train_mask = out_df['date'] < val_start
        val_mask = (out_df['date'] >= val_start) & (out_df['date'] < test_start)
        test_mask = out_df['date'] >= test_start
        
        train_df = out_df[train_mask].copy()
        val_df = out_df[val_mask].copy()
        test_df = out_df[test_mask].copy()
        
        # Validate temporal split integrity
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            st.error("Insufficient data for temporal split. Need at least 3 time periods.")
            st.stop()
        
        # Ensure no overlap in dates
        train_dates = set(train_df['date'].unique())
        val_dates = set(val_df['date'].unique())
        test_dates = set(test_df['date'].unique())
        
        if train_dates.intersection(val_dates) or val_dates.intersection(test_dates) or train_dates.intersection(test_dates):
            st.error("Temporal split error: Date overlap detected")
            st.stop()

    with st.spinner("Engineering features..."):
        st.info("🔧 Using manual feature engineering...")
        engineer_features = manual_feature_engineering
        
        train_features = engineer_features(train_df.copy())
        val_features = engineer_features(val_df.copy())
        test_features = engineer_features(test_df.copy())
        
        # Get all available features
        feature_list = [col for col in train_features.columns if col not in ['date', 'store', 'item', 'sales']]
        all_feature_cols = [col for col in feature_list if col in train_features.columns]
        
        # Debug: Show all available features
        # st.write("## 🔍 DEBUG: All Available Features")
        # st.write(f"**Total Features:** {len(all_feature_cols)}")
        
        # Categorize features (keep for internal use, but don't print)
        original_numeric = [f for f in all_feature_cols if f.startswith('orig_') and not f.endswith('_encoded')]
        original_categorical = [f for f in all_feature_cols if f.startswith('orig_') and f.endswith('_encoded')]
        lag_features = [f for f in all_feature_cols if f.startswith('sales_lag_')]
        rolling_features = [f for f in all_feature_cols if f.startswith('sales_rolling_')]
        calendar_features = [f for f in all_feature_cols if f in ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end', 'is_summer', 'is_winter', 'is_holiday_season']]
        trigonometric_features = [f for f in all_feature_cols if f.endswith('_sin') or f.endswith('_cos')]
        expanding_features = [f for f in all_feature_cols if f.startswith('sales_expanding') or f.startswith('sales_price_ratio')]
        other_features = [f for f in all_feature_cols if f not in original_numeric + original_categorical + lag_features + rolling_features + calendar_features + trigonometric_features + expanding_features]
        
        # Show a brief summary of feature categories
        st.write(f"**Feature summary:** {len(all_feature_cols)} total features, including lag, rolling, calendar, and original columns.")
        
        # Show a small sample of features in an expander (optional, less verbose)
        with st.expander("View sample features", expanded=False):
            st.write(all_feature_cols[:10])
        
        # Show data sample with features
        st.write("**📈 Sample Data with Features (First 5 rows):**")
        sample_data = train_features[['date', 'store', 'item', 'sales'] + all_feature_cols[:10]].head()
        st.dataframe(sample_data)
        
        # Always use all_feature_cols for the all-features model
        X_train_all = train_features[all_feature_cols]
        X_val_all = val_features[all_feature_cols]
        X_test_all = test_features[all_feature_cols]
        y_train = train_features['sales']
        y_val = val_features['sales']
        y_test = test_features['sales']

        # Prepare Gemini-pruned feature set if available
        valid_selected_features = []
        # --- Dynamic Feature Count Selection ---
        # Calculate cumulative importance and select features covering at least 95% of total importance
        feature_importance_df = None
        dynamic_selected_features = []
        if len(all_feature_cols) > 0:
            quick_model = LGBMRegressor(n_estimators=50, random_state=42, verbose=-1)
            quick_model.fit(train_features[all_feature_cols], train_features['sales'])
            feature_importance_df = pd.DataFrame({
                'Feature': all_feature_cols,
                'Importance': quick_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_importance_df['Cumulative'] = feature_importance_df['Importance'].cumsum() / feature_importance_df['Importance'].sum()
            threshold = 0.95
            dynamic_selected_features = feature_importance_df[feature_importance_df['Cumulative'] <= threshold]['Feature'].tolist()
            min_features = 10
            if len(dynamic_selected_features) < min_features:
                dynamic_selected_features = feature_importance_df['Feature'].head(min_features).tolist()
            st.write(f"**Dynamic Feature Selection:** {len(dynamic_selected_features)} features cover 95% of total importance.")
            with st.expander("View dynamically selected features", expanded=False):
                st.write(dynamic_selected_features)
        # --- Gemini selection (if enabled) ---
        if use_gemini_selection and GEMINI_API_KEY and feature_importance_df is not None:
            st.info("🤖 Using Gemini for intelligent feature selection...")
            st.write("**Feature Importance Ranking:**")
            st.dataframe(feature_importance_df.head(20))
            selected_features = gemini_select_best_features(feature_importance_df, GEMINI_API_KEY)
            if selected_features:
                valid_selected_features = [f for f in selected_features if f in dynamic_selected_features]
                if len(valid_selected_features) >= 10:
                    st.success(f"✅ Gemini selected {len(valid_selected_features)} features (from dynamic set)")
                    st.write(f"**Selected Features:** {valid_selected_features}")
                else:
                    st.warning("⚠️ Gemini selected too few valid features, using dynamic set instead")
                    valid_selected_features = dynamic_selected_features
            else:
                st.warning("⚠️ Gemini feature selection failed, using dynamic set instead")
                valid_selected_features = dynamic_selected_features
        else:
            st.info("🔧 Gemini feature selection not enabled or API key missing; using dynamic set")
            valid_selected_features = dynamic_selected_features

        # Set LightGBM parameters (either from Gemini/Optuna or default)
        if use_gemini_params and GEMINI_API_KEY:
            st.info("🤖 Using Gemini for hyperparameter search (Bayesian/Optuna)...")
            df_sample = train_features.sample(n=min(100, len(train_features)), random_state=42)
            param_ranges = gemini_suggest_lgbm_param_ranges(df_sample[all_feature_cols + ['sales']], all_feature_cols, GEMINI_API_KEY)
            if param_ranges:
                st.success("✅ Gemini hyperparameter range suggestion successful!")
                st.write("Suggested parameter ranges:", param_ranges)
                if optuna is None:
                    st.warning("Optuna is not installed. Please install it with 'pip install optuna' to use Bayesian optimization.")
                    lgbm_params = get_default_lgbm_params()
                else:
                    def objective(trial):
                        params = {}
                        for k, v in param_ranges.items():
                            if isinstance(v, (list, tuple)) and len(v) > 0:
                                if all(isinstance(x, int) for x in v):
                                    params[k] = trial.suggest_int(k, min(v), max(v))
                                elif all(isinstance(x, float) for x in v):
                                    params[k] = trial.suggest_float(k, min(v), max(v), log=True if min(v) > 0 else False)
                                else:
                                    params[k] = trial.suggest_categorical(k, v)
                            else:
                                params[k] = v[0] if isinstance(v, (list, tuple)) and len(v) > 0 else v
                        params.update({'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'seed': 42})
                        if params.get('boosting_type') == 'goss':
                            params.pop('bagging_fraction', None)
                            params.pop('bagging_freq', None)
                        use_early_stopping = True
                        if params.get('boosting_type') == 'dart':
                            params.pop('early_stopping_rounds', None)
                            use_early_stopping = False
                        try:
                            train_data = lgb.Dataset(X_train_all, label=y_train)
                            valid_data = lgb.Dataset(X_val_all, label=y_val, reference=train_data)
                            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] if use_early_stopping else []
                            model = lgb.train(
                                params,
                                train_data,
                                num_boost_round=1000,
                                valid_sets=[valid_data],
                                callbacks=callbacks,
                            )
                            y_pred = model.predict(X_val_all, num_iteration=model.best_iteration)
                            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                            return rmse
                        except Exception as e:
                            return float('inf')
                    n_trials = 30
                    study = optuna.create_study(direction='minimize')
                    study.optimize(objective, n_trials=n_trials)
                    best_params = study.best_trial.params
                    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'seed': 42})
                    if best_params.get('boosting_type') == 'goss':
                        best_params.pop('bagging_fraction', None)
                        best_params.pop('bagging_freq', None)
                    if best_params.get('boosting_type') == 'dart':
                        best_params.pop('early_stopping_rounds', None)
                    lgbm_params = best_params
                    st.success(f"Best hyperparameters found by Optuna (val RMSE={study.best_value:.4f}):")
                    st.write(lgbm_params)
            else:
                st.warning("⚠️ Gemini hyperparameter range suggestion failed, using default parameters...")
                lgbm_params = get_default_lgbm_params()
        else:
            st.info("🔧 Using default LightGBM parameters...")
            lgbm_params = get_default_lgbm_params()
        
        # For log-transformed and original models, always use all features
        st.info(f"✅ Ready to train with {len(all_feature_cols)} features (all features model)")
        # st.write("All features used for all-features model:", all_feature_cols)
        
        # Model 1: Original target (all features)
        st.write("### Training Model 1: Original Target (All Features)")
        train_data = lgb.Dataset(X_train_all, label=y_train)
        valid_data = lgb.Dataset(X_val_all, label=y_val, reference=train_data)
        model_original = lgb.train(
            lgbm_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ]
        )
        y_pred_original = model_original.predict(X_test_all, num_iteration=model_original.best_iteration)
        mse_original = mean_squared_error(y_test, y_pred_original)
        rmse_original = np.sqrt(mse_original)
        # r2_original = r2_score(y_test, y_pred_original)
        st.success(f"Model 1 (Original, All Features) - Test MSE: {mse_original:.4f}, RMSE: {rmse_original:.4f}")
        
        # Model 2: Log-transformed target (all features)
        st.write("### Training Model 2: Log-Transformed Target (All Features)")
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        y_test_log = np.log1p(y_test)
        train_data_log = lgb.Dataset(X_train_all, label=y_train_log)
        valid_data_log = lgb.Dataset(X_val_all, label=y_val_log, reference=train_data_log)
        model_log = lgb.train(
            lgbm_params,
            train_data_log,
            num_boost_round=1000,
            valid_sets=[valid_data_log],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
            ]
        )
        y_pred_log = model_log.predict(X_test_all, num_iteration=model_log.best_iteration)
        y_pred_log_transformed = np.expm1(y_pred_log)
        mse_log = mean_squared_error(y_test, y_pred_log_transformed)
        rmse_log = np.sqrt(mse_log)
        # r2_log = r2_score(y_test, y_pred_log_transformed)
        st.success(f"Model 2 (Log-Transformed, All Features) - Test MSE: {mse_log:.4f}, RMSE: {rmse_log:.4f}")
        
        # Model 3: Gemini-Pruned Features (if enabled and successful)
        gemini_model_trained = False
        if valid_selected_features:
            st.write("Gemini-pruned features used for training:", valid_selected_features)
            st.write("Number of Gemini-pruned features:", len(valid_selected_features))
            # st.write("All features (original):", all_feature_cols)
            # st.write("Number of all features:", len(all_feature_cols))
            # st.write("Are feature sets identical?", set(valid_selected_features) == set(all_feature_cols))
            # st.write("Features in all but not in Gemini-pruned:", list(set(all_feature_cols) - set(valid_selected_features)))
            # st.write("Features in Gemini-pruned but not in all:", list(set(valid_selected_features) - set(all_feature_cols)))
            X_train_gemini = train_features[valid_selected_features]
            X_val_gemini = val_features[valid_selected_features]
            X_test_gemini = test_features[valid_selected_features]
            train_data_gemini = lgb.Dataset(X_train_gemini, label=y_train)
            valid_data_gemini = lgb.Dataset(X_val_gemini, label=y_val, reference=train_data_gemini)
            model_gemini = lgb.train(
                lgbm_params,
                train_data_gemini,
                num_boost_round=1000,
                valid_sets=[valid_data_gemini],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                ]
            )
            y_pred_gemini = model_gemini.predict(X_test_gemini, num_iteration=model_gemini.best_iteration)
            mse_gemini = mean_squared_error(y_test, y_pred_gemini)
            rmse_gemini = np.sqrt(mse_gemini)
            # r2_gemini = r2_score(y_test, y_pred_gemini)
            st.success(f"Model 3 (Gemini-Pruned Features) - Test MSE: {mse_gemini:.4f}, RMSE: {rmse_gemini:.4f}")
            gemini_model_trained = True
        
        # Model Comparison
        st.write("### Model Comparison")
        model_results = {
            'Original Target (All Features)': {'MSE': mse_original, 'RMSE': rmse_original},
            'Log-Transformed Target (All Features)': {'MSE': mse_log, 'RMSE': rmse_log},
        }
        if gemini_model_trained:
            model_results['Gemini-Pruned Features'] = {'MSE': mse_gemini, 'RMSE': rmse_gemini}
        
        comparison_df = pd.DataFrame({
            'Model': list(model_results.keys()),
            'Test MSE': [model_results[model]['MSE'] for model in model_results.keys()],
            'Test RMSE': [model_results[model]['RMSE'] for model in model_results.keys()]
        })
        st.dataframe(comparison_df)
        
        # Select best model based on RMSE (lower is better)
        best_model_name = min(model_results, key=lambda x: model_results[x]['RMSE'])
        best_mse = model_results[best_model_name]['MSE']
        best_rmse = model_results[best_model_name]['RMSE']
        st.success(f"🏆 Best Model: {best_model_name} (MSE: {best_mse:.4f}, RMSE: {best_rmse:.4f})")
        
        # Select the best model for predictions
        if best_model_name == 'Original Target (All Features)':
            best_model = model_original
            best_feature_cols = all_feature_cols
            best_test_features = test_features
        elif best_model_name == 'Log-Transformed Target (All Features)':
            best_model = model_log
            best_feature_cols = all_feature_cols
            best_test_features = test_features
        else:  # Gemini-Pruned Features
            best_model = model_gemini
            best_feature_cols = valid_selected_features
            best_test_features = test_features[valid_selected_features + ['date', 'store', 'item', 'sales']]
        
        st.write("### Best Model Predictions (first 20 rows)")
        if best_model_name == 'Log-Transformed Target (All Features)':
            y_pred_best = np.expm1(best_model.predict(X_test_all[best_feature_cols], num_iteration=best_model.best_iteration))
        else:
            y_pred_best = best_model.predict(X_test_all[best_feature_cols], num_iteration=best_model.best_iteration)
        
        pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best})
        pred_df['Error (%)'] = ((pred_df['Predicted'] - pred_df['Actual']) / (pred_df['Actual'] + 1e-8)) * 100
        st.dataframe(pred_df.head(20))

    # --- Interactive Prediction ---
    st.header("3. Interactive Prediction")
    available_items = sorted(test_features['item'].unique())
    available_stores = sorted(test_features['store'].unique())
    
    # Ensure all names are strings
    item_names = [str(encoding_maps['item'].get(item, f"Item {item}")) for item in available_items]
    store_names = [str(encoding_maps['store'].get(store, f"Store {store}")) for store in available_stores]
    
    st.write(f"**Available Items:** {len(item_names)} items")
    st.write(f"**Available Stores:** {len(store_names)} stores")
    st.write(f"**Date Range:** {test_features['date'].min()} to {test_features['date'].max()}")
    
    # Show items in expandable section
    with st.expander(f"📋 View All {len(item_names)} Items"):
        for i, item in enumerate(item_names, 1):
            st.write(f"{i}. {item}")
    
    # Show stores in expandable section
    with st.expander(f"🏪 View All {len(store_names)} Stores"):
        for i, store in enumerate(store_names, 1):
            st.write(f"{i}. {store}")


    # Default to top-selling analysis for immediate insights
    
    st.write("### 🏆 Top-Selling Products Analysis")
    st.write("This will predict sales for all items over the next 2 weeks and rank them by total predicted sales.")
    
    # Performance mode selection
    performance_mode = st.radio(
        "Analysis Mode",
        ["Fast (Top 50 items)", "Standard (All items)", "Quick Sample (Top 20 items)"],
        help="Choose analysis speed vs. coverage"
    )
    
    # User can specify time period (max 2 weeks = 14 days)
    days = st.number_input("Forecast period (days)", min_value=1, max_value=14, value=14, 
                          help="Maximum 2 weeks (14 days)")
    
    if st.button("Analyze Top-Selling Products"):
        with st.spinner("Predicting sales for all items..."):
            # Get all unique items
            all_items = test_features['item'].unique()
            
            # Performance optimization based on selected mode
            if performance_mode == "Fast (Top 50 items)":
                if len(all_items) > 50:
                    st.info(f"📊 Fast Mode: Analyzing top 50 items out of {len(all_items)} total items")
                    all_items = all_items[:50]
            elif performance_mode == "Quick Sample (Top 20 items)":
                if len(all_items) > 20:
                    st.info(f"⚡ Quick Mode: Analyzing top 20 items out of {len(all_items)} total items")
                    all_items = all_items[:20]
            else:  # Standard mode
                st.info(f"📊 Standard Mode: Analyzing all {len(all_items)} items")
            
            # Store results for each item
            item_predictions = []
            
            # Progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Performance optimization: Batch processing
            batch_size = 10
            for batch_start in range(0, len(all_items), batch_size):
                batch_end = min(batch_start + batch_size, len(all_items))
                batch_items = all_items[batch_start:batch_end]
                
                for idx, item in enumerate(batch_items):
                    progress_text.text(f"Predicting for item {batch_start + idx + 1}/{len(all_items)}")
                    
                    # Get last observation for this item
                    item_data = test_features[test_features['item'] == item]
                    if len(item_data) > 0:
                        last_obs = item_data.sort_values('date').iloc[[-1]].copy()
                        
                        # Make predictions
                        preds = recursive_forecast(best_model, last_obs, best_feature_cols, steps=days)
                        
                        # Calculate total predicted sales
                        total_predicted_sales = sum(preds)
                        avg_daily_sales = total_predicted_sales / days
                        
                        item_name = str(encoding_maps['item'].get(item, f"Item {item}"))
                        
                        item_predictions.append({
                            'item_id': item,
                            'item_name': item_name,
                            'total_predicted_sales': total_predicted_sales,
                            'avg_daily_sales': avg_daily_sales,
                            'predictions': preds
                        })
                    
                    # Update progress
                    progress_bar.progress((batch_start + idx + 1) / len(all_items))
            
            progress_text.text("Analysis complete!")
            
            # Sort by total predicted sales (descending)
            item_predictions.sort(key=lambda x: x['total_predicted_sales'], reverse=True)
            
            # Display results
            st.write(f"### 📊 Top-Selling Products (Next {days} Days)")
            
            # Create summary dataframe
            summary_data = []
            for i, item_pred in enumerate(item_predictions[:20]):  # Show top 20
                summary_data.append({
                    'Rank': i + 1,
                    'Item Name': item_pred['item_name'],
                    'Total Predicted Sales': f"{item_pred['total_predicted_sales']:.0f}",
                    'Avg Daily Sales': f"{item_pred['avg_daily_sales']:.1f}",
                    'Item ID': item_pred['item_id']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Show detailed analysis for top 5
            st.write("### 📈 Detailed Analysis - Top 5 Products")
            
            for i, item_pred in enumerate(item_predictions[:5]):
                with st.expander(f"#{i+1} - {item_pred['item_name']} (Total: {item_pred['total_predicted_sales']:.0f} units)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Total Predicted Sales:** {item_pred['total_predicted_sales']:.0f} units")
                        st.write(f"**Average Daily Sales:** {item_pred['avg_daily_sales']:.1f} units")
                        st.write(f"**Item ID:** {item_pred['item_id']}")
                    
                    with col2:
                        # Create a simple line chart of predictions
                        pred_df = pd.DataFrame({
                            'Day': range(1, days + 1),
                            'Predicted Sales': item_pred['predictions']
                        })
                        st.line_chart(pred_df.set_index('Day'))
                    
                    # Show weekly breakdown
                    weekly_data = []
                    for week in range(0, days, 7):
                        week_end = min(week + 7, days)
                        week_sales = sum(item_pred['predictions'][week:week_end])
                        weekly_data.append({
                            'Week': f"Week {(week//7)+1}",
                            'Sales': week_sales
                        })
                    
                    st.write("**Weekly Breakdown:**")
                    weekly_df = pd.DataFrame(weekly_data)
                    st.dataframe(weekly_df)
            
            # Insights section
            st.write("### 💡 Key Insights")
            
            if len(item_predictions) > 0:
                top_item = item_predictions[0]
                bottom_item = item_predictions[-1]
                
                st.write(f"**🏆 Best Performer:** {top_item['item_name']}")
                st.write(f"   - Predicted to sell {top_item['total_predicted_sales']:.0f} units")
                st.write(f"   - Average of {top_item['avg_daily_sales']:.1f} units per day")
                
                st.write(f"**📉 Lowest Performer:** {bottom_item['item_name']}")
                st.write(f"   - Predicted to sell {bottom_item['total_predicted_sales']:.0f} units")
                st.write(f"   - Average of {bottom_item['avg_daily_sales']:.1f} units per day")
                
                # Calculate some statistics
                total_sales = sum(item['total_predicted_sales'] for item in item_predictions)
                avg_sales = total_sales / len(item_predictions)
                
                st.write(f"**📊 Overall Statistics:**")
                st.write(f"   - Total predicted sales across all items: {total_sales:.0f} units")
                st.write(f"   - Average predicted sales per item: {avg_sales:.0f} units")
                st.write(f"   - Top item contributes {((top_item['total_predicted_sales']/total_sales)*100):.1f}% of total sales")
            all_predictions = []
            # Add to all_predictions for CSV download
            for item_pred in item_predictions:
                all_predictions.append({
                    'type': 'top_selling_analysis',
                    'item': item_pred['item_id'],
                    'item_name': item_pred['item_name'],
                    'store': 'all',
                    'days': days,
                    'predictions': item_pred['predictions']
                })
    
    # Initialize predictions list for CSV download
    all_predictions = []
    
    # Additional prediction modes in expandable section
    with st.expander("🔮 Other Prediction Modes"):
        prediction_mode = st.selectbox("Prediction mode", [
            "Predict for specific item(s)",
            "Predict for specific store(s)",
            "Predict for specific item-store combination",
            "Show sample predictions for random items"
        ])
    if prediction_mode == "Predict for specific item(s)":
        selected_items = st.multiselect("Select item(s)", item_names)
        days = st.number_input("How many days to forecast?", min_value=1, max_value=30, value=7)
        if st.button("Forecast") and selected_items:
            for item_name in selected_items:
                item_id = next((k for k, v in encoding_maps['item'].items() if str(v) == item_name), None)
                if item_id is not None:
                    item_data = test_features[test_features['item'] == item_id]
                    if len(item_data) > 0:
                        last_obs = item_data.sort_values('date').iloc[[-1]].copy()
                        preds = recursive_forecast(best_model, last_obs, best_feature_cols, steps=days)
                        st.write(f"#### {item_name} forecast:")
                        st.write(pd.DataFrame({"Day": range(1, days+1), "Predicted Sales": preds}))
                        all_predictions.append({
                            'type': 'specific_item',
                            'item': item_id,
                            'item_name': item_name,
                            'store': 'all',
                            'days': days,
                            'predictions': preds
                        })
    elif prediction_mode == "Predict for specific store(s)":
        selected_stores = st.multiselect("Select store(s)", store_names)
        days = st.number_input("How many days to forecast?", min_value=1, max_value=30, value=7)
        if st.button("Forecast") and selected_stores:
            for store_name in selected_stores:
                store_id = next((k for k, v in encoding_maps['store'].items() if str(v) == store_name), None)
                if store_id is not None:
                    store_data = test_features[test_features['store'] == store_id]
                    store_items = store_data['item'].unique()
                    st.write(f"#### {store_name} (items: {[str(encoding_maps['item'][item]) for item in sorted(store_items)]})")
                    for item in sorted(store_items):
                        item_data = store_data[store_data['item'] == item]
                        last_obs = item_data.sort_values('date').iloc[[-1]].copy()
                        preds = recursive_forecast(best_model, last_obs, best_feature_cols, steps=days)
                        item_name = str(encoding_maps['item'].get(item, f"Item {item}"))
                        st.write(f"{item_name}: {preds}")
                        all_predictions.append({
                            'type': 'store_item',
                            'item': item,
                            'item_name': item_name,
                            'store': store_id,
                            'store_name': store_name,
                            'days': days,
                            'predictions': preds
                        })
    elif prediction_mode == "Predict for specific item-store combination":
        item_name = st.selectbox("Select item", item_names)
        store_name = st.selectbox("Select store", store_names)
        days = st.number_input("How many days to forecast?", min_value=1, max_value=30, value=7)
        if st.button("Forecast"):
            item_id = next((k for k, v in encoding_maps['item'].items() if str(v) == item_name), None)
            store_id = next((k for k, v in encoding_maps['store'].items() if str(v) == store_name), None)
            if item_id is not None and store_id is not None:
                item_store_data = test_features[(test_features['item'] == item_id) & (test_features['store'] == store_id)]
                if len(item_store_data) > 0:
                    last_obs = item_store_data.sort_values('date').iloc[[-1]].copy()
                    preds = recursive_forecast(best_model, last_obs, best_feature_cols, steps=days)
                    st.write(f"#### Forecast for {item_name} in {store_name}")
                    st.write(pd.DataFrame({"Day": range(1, days+1), "Predicted Sales": preds}))
                    all_predictions.append({
                        'type': 'specific_combination',
                        'item': item_id,
                        'item_name': item_name,
                        'store': store_id,
                        'store_name': store_name,
                        'days': days,
                        'predictions': preds
                    })
    elif prediction_mode == "Show sample predictions for random items":
        num_items = st.number_input("How many random items to show?", min_value=1, max_value=len(available_items), value=3)
        days = st.number_input("How many days to forecast?", min_value=1, max_value=30, value=5)
        if st.button("Show Random Forecasts"):
            selected_items = random.sample(available_items, num_items)
            for item in selected_items:
                item_data = test_features[test_features['item'] == item]
                if len(item_data) > 0:
                    last_obs = item_data.sort_values('date').iloc[[-1]].copy()
                    preds = recursive_forecast(best_model, last_obs, best_feature_cols, steps=days)
                    item_name = str(encoding_maps['item'].get(item, f"Item {item}"))
                    st.write(f"{item_name}: {preds}")
                    all_predictions.append({
                        'type': 'random_item',
                        'item': item,
                        'item_name': item_name,
                        'store': 'all',
                        'days': days,
                        'predictions': preds
                    })
    elif prediction_mode == "Suggest top-selling products (next 2 weeks)":
        st.write("### 🏆 Top-Selling Products Analysis")
        st.write("This will predict sales for all items over the next 2 weeks and rank them by total predicted sales.")
        
        # User can specify time period (max 2 weeks = 14 days)
        days = st.number_input("Forecast period (days)", min_value=1, max_value=14, value=14, 
                              help="Maximum 2 weeks (14 days)")
        all_predictions = []
        if st.button("Analyze Top-Selling Products"):
            with st.spinner("Predicting sales for all items..."):
                # Get all unique items
                all_items = test_features['item'].unique()
                
                # Store results for each item
                item_predictions = []
                
                # Progress bar
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                for idx, item in enumerate(all_items):
                    progress_text.text(f"Predicting for item {idx+1}/{len(all_items)}")
                    
                    # Get last observation for this item
                    item_data = test_features[test_features['item'] == item]
                    if len(item_data) > 0:
                        last_obs = item_data.sort_values('date').iloc[[-1]].copy()
                        
                        # Make predictions
                        preds = recursive_forecast(best_model, last_obs, best_feature_cols, steps=days)
                        
                        # Calculate total predicted sales
                        total_predicted_sales = sum(preds)
                        avg_daily_sales = total_predicted_sales / days
                        
                        item_name = str(encoding_maps['item'].get(item, f"Item {item}"))
                        
                        item_predictions.append({
                            'item_id': item,
                            'item_name': item_name,
                            'total_predicted_sales': total_predicted_sales,
                            'avg_daily_sales': avg_daily_sales,
                            'predictions': preds
                        })
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(all_items))
                
                progress_text.text("Analysis complete!")
                
                # Sort by total predicted sales (descending)
                item_predictions.sort(key=lambda x: x['total_predicted_sales'], reverse=True)
                
                # Display results
                st.write(f"### 📊 Top-Selling Products (Next {days} Days)")
                
                # Create summary dataframe
                summary_data = []
                for i, item_pred in enumerate(item_predictions[:20]):  # Show top 20
                    summary_data.append({
                        'Rank': i + 1,
                        'Item Name': item_pred['item_name'],
                        'Total Predicted Sales': f"{item_pred['total_predicted_sales']:.0f}",
                        'Avg Daily Sales': f"{item_pred['avg_daily_sales']:.1f}",
                        'Item ID': item_pred['item_id']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Show detailed analysis for top 5
                st.write("### 📈 Detailed Analysis - Top 5 Products")
                
                for i, item_pred in enumerate(item_predictions[:5]):
                    with st.expander(f"#{i+1} - {item_pred['item_name']} (Total: {item_pred['total_predicted_sales']:.0f} units)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Total Predicted Sales:** {item_pred['total_predicted_sales']:.0f} units")
                            st.write(f"**Average Daily Sales:** {item_pred['avg_daily_sales']:.1f} units")
                            st.write(f"**Item ID:** {item_pred['item_id']}")
                        
                        with col2:
                            # Create a simple line chart of predictions
                            pred_df = pd.DataFrame({
                                'Day': range(1, days + 1),
                                'Predicted Sales': item_pred['predictions']
                            })
                            st.line_chart(pred_df.set_index('Day'))
                        
                        # Show weekly breakdown
                        weekly_data = []
                        for week in range(0, days, 7):
                            week_end = min(week + 7, days)
                            week_sales = sum(item_pred['predictions'][week:week_end])
                            weekly_data.append({
                                'Week': f"Week {(week//7)+1}",
                                'Sales': week_sales
                            })
                        
                        st.write("**Weekly Breakdown:**")
                        weekly_df = pd.DataFrame(weekly_data)
                        st.dataframe(weekly_df)
                
                # Insights section
                st.write("### 💡 Key Insights")
                
                if len(item_predictions) > 0:
                    top_item = item_predictions[0]
                    bottom_item = item_predictions[-1]
                    
                    st.write(f"**🏆 Best Performer:** {top_item['item_name']}")
                    st.write(f"   - Predicted to sell {top_item['total_predicted_sales']:.0f} units")
                    st.write(f"   - Average of {top_item['avg_daily_sales']:.1f} units per day")
                    
                    st.write(f"**📉 Lowest Performer:** {bottom_item['item_name']}")
                    st.write(f"   - Predicted to sell {bottom_item['total_predicted_sales']:.0f} units")
                    st.write(f"   - Average of {bottom_item['avg_daily_sales']:.1f} units per day")
                    
                    # Calculate some statistics
                    total_sales = sum(item['total_predicted_sales'] for item in item_predictions)
                    avg_sales = total_sales / len(item_predictions)
                    
                    st.write(f"**📊 Overall Statistics:**")
                    st.write(f"   - Total predicted sales across all items: {total_sales:.0f} units")
                    st.write(f"   - Average predicted sales per item: {avg_sales:.0f} units")
                    st.write(f"   - Top item contributes {((top_item['total_predicted_sales']/total_sales)*100):.1f}% of total sales")
                
                # Add to all_predictions for CSV download
                for item_pred in item_predictions:
                    all_predictions.append({
                        'type': 'top_selling_analysis',
                        'item': item_pred['item_id'],
                        'item_name': item_pred['item_name'],
                        'store': 'all',
                        'days': days,
                        'predictions': item_pred['predictions']
                    })
    # Download predictions
    if all_predictions:
        pred_df = save_predictions_to_csv(all_predictions)
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to get started.") 