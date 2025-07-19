# Sales Forecasting Streamlit Application

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Core Components](#core-components)
4. [Technical Implementation](#technical-implementation)
5. [Features & Capabilities](#features--capabilities)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [API Integration](#api-integration)
9. [Model Performance](#model-performance)
10. [Troubleshooting](#troubleshooting)

## 🎯 Overview

This is a comprehensive sales forecasting application built with Streamlit that leverages machine learning (LightGBM) and AI (Google Gemini) to provide accurate sales predictions. The application can handle various data formats, automatically map columns, engineer features, and generate forecasts for different business scenarios.

### Key Features

- **Universal Data Handling**: Works with any CSV format through intelligent column mapping
- **AI-Powered Automation**: Uses Google Gemini for column mapping, feature selection, and hyperparameter optimization
- **Multiple Model Approaches**: Compares original target, log-transformed, and RFE-pruned models
- **Interactive Predictions**: Supports various prediction modes (items, stores, combinations)
- **Top-Selling Analysis**: Identifies best-performing products for strategic planning

## 🏗️ Architecture & Data Flow

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CSV Upload    │───▶│  Column Mapping │───▶│ Data Processing │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Predictions   │◀───│ Model Training  │◀───│ Feature Eng.    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Detailed Data Flow

#### 1. Data Ingestion & Preprocessing

```
Raw CSV → Column Detection → Mapping → Timeseries Conversion → Encoding
```

**Steps:**

1. **File Upload**: User uploads CSV file through Streamlit interface
2. **Column Detection**:
   - Gemini AI analyzes sample data for intelligent mapping
   - Fallback to smart keyword-based mapping if AI fails
3. **Data Standardization**: Converts to standard format (date, store, item, sales)
4. **Encoding**: Converts categorical variables to numeric codes with mapping dictionaries

#### 2. Feature Engineering Pipeline

```
Standardized Data → Lag Features → Rolling Features → Calendar Features → Original Features
```

**Feature Categories:**

- **Lag Features**: Previous sales values (1, 2, 3, 7, 14, 30 days)
- **Rolling Features**: Moving averages, std, min, max (3, 7, 14, 30 windows)
- **Calendar Features**: Year, month, day, dayofweek, seasonal indicators
- **Trigonometric Features**: Cyclical encoding for time patterns
- **Expanding Features**: Cumulative statistics
- **Original Features**: Preserved from input data (numeric/categorical)

#### 3. Model Training & Selection

```
Feature Set → Multiple Models → Performance Comparison → Best Model Selection
```

**Model Variants:**

1. **Original Target**: Direct sales prediction
2. **Log-Transformed**: Log-transformed target for skewed distributions
3. **RFE Pruned**: Recursive Feature Elimination for optimal feature subset

#### 4. Prediction Generation

```
Best Model → Recursive Forecasting → Multi-step Predictions → Results Display
```

## 🔧 Core Components

### 1. Column Mapping System

#### Gemini AI Mapping (`ask_gemini_for_column_mapping_from_sample`)

```python
def ask_gemini_for_column_mapping_from_sample(df_sample, api_key):
    # Sends sample data to Gemini with specific prompt
    # Returns mapping dictionary: {'date': 'Date', 'store': 'Store', ...}
```

#### Smart Fallback Mapping (`smart_column_mapping`)

```python
def smart_column_mapping(df):
    # Keyword-based column detection
    # Handles common naming conventions
    # Returns standardized mapping
```

### 2. Data Conversion Engine (`convert_to_timeseries`)

**Process:**

1. **Store Handling**: Creates numeric codes for stores, handles missing values
2. **Item Handling**: Creates numeric codes for items, handles missing values
3. **Date Processing**: Converts to datetime, creates default range if missing
4. **Sales Calculation**: Handles direct columns or formulas (e.g., "Quantity \* Price")
5. **Feature Preservation**: Keeps all original columns as additional features

**Output Structure:**

```python
{
    'store': [1, 2, 3, ...],           # Numeric store codes
    'item': [1, 2, 3, ...],           # Numeric item codes
    'date': [datetime objects],       # Standardized dates
    'sales': [float values],          # Sales quantities
    'orig_price': [float values],     # Preserved original features
    'orig_category_encoded': [1, 2, 3, ...]  # Encoded categorical features
}
```

### 3. Feature Engineering Engine (`manual_feature_engineering`)

**Generated Features:**

**Lag Features:**

- `sales_lag_1`, `sales_lag_2`, `sales_lag_3`, `sales_lag_7`, `sales_lag_14`, `sales_lag_30`
- **Purpose**: Previous sales values for trend analysis

**Rolling Features:**

- `sales_rolling_mean_3/7/14/30`, `sales_rolling_std_3/7/14/30`, `sales_rolling_min_3/7/14/30`, `sales_rolling_max_3/7/14/30`
- **Purpose**: Moving window statistics for pattern recognition

**Calendar Features:**

- `year`, `month`, `day`, `dayofweek`, `quarter`, `is_weekend`, `is_month_start`, `is_month_end`, `is_summer`, `is_winter`, `is_holiday_season`
- **Purpose**: Time-based indicators for seasonal patterns

**Trigonometric Features:**

- `month_sin`, `month_cos`, `dayofweek_sin`, `dayofweek_cos`
- **Purpose**: Cyclical time encoding for smooth seasonal transitions

**Expanding Features:**

- `sales_expanding_mean`, `sales_price_ratio`
- **Purpose**: Cumulative statistics and price relationships

### 4. Model Training System

#### Hyperparameter Optimization

```python
# Gemini AI suggests parameter ranges
param_ranges = gemini_suggest_lgbm_param_ranges(df_sample, features, api_key)

# Optuna performs Bayesian optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
```

#### Model Comparison Logic

```python
# Train three model variants
models = {
    'Original': train_original_target(),
    'Log-Transformed': train_log_transformed(),
    'RFE-Pruned': train_rfe_pruned()
}

# Select best based on R² score
best_model = max(models, key=lambda x: models[x]['r2_score'])
```

### 5. Prediction Engine (`recursive_forecast`)

**Algorithm:**

1. **Initial State**: Use last available observation
2. **Iterative Prediction**: For each forecast step:
   - Predict next value using current features
   - Update lag features with new prediction
   - Update rolling features with new window
   - Update expanding features with cumulative data
3. **Return**: List of predictions for requested horizon

## 🚀 Technical Implementation

### Dependencies

```python
# Core ML & Data Processing
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import ParameterSampler

# AI Integration
import google.generativeai as genai

# Optimization
import optuna  # Optional: for hyperparameter optimization

# Web Framework
import streamlit as st
```

### Key Functions & Their Purposes

#### Data Processing Functions

- `robust_parse_dates()`: Handles various date formats with error recovery
- `convert_to_timeseries()`: Main data conversion pipeline
- `manual_feature_engineering()`: Comprehensive feature generation

#### AI Integration Functions

- `ask_gemini_for_column_mapping_from_sample()`: Intelligent column mapping
- `gemini_suggest_lgbm_param_ranges()`: Hyperparameter range suggestions
- `gemini_select_best_features()`: Feature importance analysis and selection

#### Model Functions

- `get_default_lgbm_params()`: Default LightGBM configuration
- `recursive_forecast()`: Multi-step prediction algorithm
- `save_predictions_to_csv()`: Export functionality

### Error Handling & Robustness

#### Graceful Degradation

```python
# AI failure fallbacks
if gemini_mapping_fails:
    use_smart_column_mapping()

if gemini_params_fail:
    use_default_parameters()

if optuna_not_available:
    use_grid_search_or_defaults()
```

#### Data Validation

```python
# Handle missing/invalid data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)  # or appropriate defaults
```

## 🎨 Features & Capabilities

### 1. Universal Data Handling

- **Flexible Column Mapping**: Works with any CSV structure
- **Automatic Type Detection**: Handles numeric, categorical, and datetime data
- **Missing Value Handling**: Robust imputation strategies
- **Formula Support**: Can calculate sales from multiple columns (e.g., "Quantity \* Price")

### 2. AI-Powered Automation

- **Intelligent Column Mapping**: Gemini AI analyzes data structure
- **Feature Selection**: AI-driven feature importance analysis
- **Hyperparameter Optimization**: Automated parameter tuning
- **Business Logic Integration**: AI considers domain knowledge

### 3. Multiple Prediction Modes

- **Item-Specific**: Forecast for selected products
- **Store-Specific**: Forecast for selected locations
- **Combination**: Item-store specific predictions
- **Random Sampling**: Quick insights with random items
- **Top-Selling Analysis**: Strategic product ranking

### 4. Advanced Analytics

- **Model Comparison**: Multiple approaches with performance metrics
- **Feature Importance**: Understanding what drives predictions
- **Error Analysis**: Prediction accuracy assessment
- **Trend Analysis**: Seasonal and cyclical pattern detection

## 📦 Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install required packages
pip install streamlit pandas numpy lightgbm scikit-learn google-generativeai

# Optional: For advanced hyperparameter optimization
pip install optuna
```

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd sales-forecasting-app

# Install dependencies
pip install -r requirements.txt

# Set up API key (optional)
export GEMINI_API_KEY="your-api-key-here"
```

### Running the Application

```bash
# Start Streamlit app
streamlit run app2.py

# Access at http://localhost:8501
```

## 📖 Usage Guide

### 1. Data Preparation

- **Format**: CSV file with sales data
- **Required Columns**: At least one of date, store, item, sales
- **Data Quality**: Clean data preferred, but app handles missing values

### 2. Configuration

- **Gemini API Key**: Enable AI features (optional but recommended)
- **Feature Toggles**: Control which AI features to use
- **Model Settings**: Adjust training parameters

### 3. Workflow

1. **Upload Data**: Select CSV file
2. **Review Mapping**: Check column mapping results
3. **Monitor Training**: Watch model training progress
4. **Analyze Results**: Review model performance
5. **Generate Predictions**: Use interactive prediction modes
6. **Export Results**: Download predictions as CSV

### 4. Prediction Modes

#### Specific Item Prediction

- Select one or more items
- Choose forecast horizon (1-30 days)
- Get detailed predictions with confidence intervals

#### Store Analysis

- Select stores to analyze
- View all items in selected stores
- Compare performance across locations

#### Top-Selling Analysis

- Forecast all items for 2 months
- Rank by total predicted sales
- Strategic planning insights

## 🔌 API Integration

### Google Gemini API

```python
# Configuration
MODEL_NAME = "gemini-1.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Usage
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)
response = model.generate_content(prompt)
```

### API Key Management

- **Security**: API key stored securely in Streamlit
- **Fallback**: Graceful degradation when API unavailable
- **Rate Limiting**: Built-in retry logic with delays

## 📊 Model Performance

### Evaluation Metrics

- **MSE (Mean Squared Error)**: Overall prediction accuracy
- **RMSE (Root Mean Squared Error)**: Error in original units
- **R² Score**: Model fit quality (0-1, higher is better)

### Model Selection Strategy

```python
# Compare multiple approaches
model_scores = {
    'Original': calculate_metrics(original_model),
    'Log-Transformed': calculate_metrics(log_model),
    'RFE-Pruned': calculate_metrics(rfe_model)
}

# Select best based on R² score
best_model = max(model_scores, key=lambda x: model_scores[x]['r2'])
```

### Performance Optimization

- **Early Stopping**: Prevents overfitting
- **Cross-Validation**: Robust performance estimation
- **Feature Selection**: Reduces noise and improves speed
- **Hyperparameter Tuning**: Optimizes model configuration

## 🔧 Troubleshooting

### Common Issues

#### 1. Column Mapping Failures

**Symptoms**: Incorrect column identification
**Solutions**:

- Check data format and column names
- Use manual mapping if AI fails
- Verify data quality and completeness

#### 2. Model Training Issues

**Symptoms**: Poor performance or training errors
**Solutions**:

- Increase training data size
- Adjust hyperparameters
- Check for data quality issues
- Try different model variants

#### 3. Prediction Errors

**Symptoms**: Invalid or unrealistic predictions
**Solutions**:

- Verify feature engineering
- Check for data leakage
- Validate input data format
- Review model performance metrics

#### 4. API Integration Problems

**Symptoms**: Gemini features not working
**Solutions**:

- Verify API key validity
- Check internet connectivity
- Review API usage limits
- Use fallback methods

### Debug Mode

```python
# Enable debug information
st.write("## 🔍 DEBUG: All Available Features")
st.write(f"**Total Features:** {len(all_feature_cols)}")

# Show detailed feature breakdown
with st.expander("🔍 View ALL Features (Click to expand)"):
    for i, feature in enumerate(all_feature_cols, 1):
        st.write(f"{i:2d}. {feature}")
```

### Performance Monitoring

- **Training Time**: Monitor model training duration
- **Memory Usage**: Track resource consumption
- **Prediction Speed**: Measure inference time
- **Accuracy Metrics**: Regular performance assessment

## 🔮 Future Enhancements

### Planned Features

1. **Real-time Data Integration**: Connect to live data sources
2. **Advanced Visualization**: Interactive charts and dashboards
3. **Ensemble Methods**: Combine multiple model predictions
4. **Anomaly Detection**: Identify unusual sales patterns
5. **Automated Reporting**: Generate business intelligence reports

### Technical Improvements

1. **Model Persistence**: Save and load trained models
2. **Batch Processing**: Handle larger datasets efficiently
3. **API Endpoints**: RESTful API for external integration
4. **Cloud Deployment**: Scalable cloud infrastructure
5. **Real-time Predictions**: Streaming prediction capabilities

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support and questions, please open an issue in the repository or contact the development team.

# Environment Variables

This project uses a `.env` file to store sensitive information such as the Gemini API key. **Do not commit your `.env` file to version control.**

## How to set up

1. Create a file named `.env` in the project root.
2. Add your Gemini API key:

   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

3. The application will automatically load this key at runtime.

## Security

- The `.env` file is included in `.gitignore` and will not be committed to GitHub.
- Each user should provide their own API key in their local `.env` file.
