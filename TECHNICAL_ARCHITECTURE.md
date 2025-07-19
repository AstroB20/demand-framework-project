# Technical Architecture Documentation

## System Overview

The Sales Forecasting Application is built as a modular, AI-enhanced machine learning system that combines traditional ML techniques with modern AI capabilities. The architecture follows a pipeline-based approach with multiple fallback mechanisms for robustness.

## Architecture Layers

### 1. Presentation Layer (Streamlit)

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Interface                      │
├─────────────────────────────────────────────────────────────┤
│ • File Upload Widgets                                       │
│ • Configuration Sidebar                                     │
│ • Progress Indicators                                       │
│ • Results Display                                           │
│ • Interactive Controls                                       │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

- **File Upload**: `st.file_uploader()` for CSV ingestion
- **Configuration Panel**: API keys, feature toggles, model settings
- **Progress Tracking**: `st.spinner()`, `st.progress()` for user feedback
- **Results Display**: Dataframes, charts, expandable sections

### 2. Business Logic Layer

```
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                      │
├─────────────────────────────────────────────────────────────┤
│ • Data Processing Pipeline                                  │
│ • Feature Engineering Engine                                │
│ • Model Training Orchestrator                               │
│ • Prediction Engine                                         │
│ • Results Formatter                                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Responsibilities:**

- Orchestrate data flow between components
- Manage AI integration and fallbacks
- Coordinate model training and selection
- Handle prediction generation and formatting

### 3. AI Integration Layer

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Integration Layer                      │
├─────────────────────────────────────────────────────────────┤
│ • Gemini API Client                                         │
│ • Prompt Engineering                                        │
│ • Response Parser                                           │
│ • Error Handler                                             │
│ • Fallback Manager                                          │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

- **API Client**: Google Generative AI integration
- **Prompt Templates**: Structured prompts for different tasks
- **Response Processing**: JSON parsing and validation
- **Retry Logic**: Exponential backoff for API failures

### 4. Machine Learning Layer

```
┌─────────────────────────────────────────────────────────────┐
│                Machine Learning Layer                       │
├─────────────────────────────────────────────────────────────┤
│ • LightGBM Models                                          │
│ • Feature Engineering Pipeline                              │
│ • Hyperparameter Optimization                               │
│ • Model Evaluation Engine                                   │
│ • Prediction Algorithms                                     │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

- **Model Variants**: Original, Log-transformed, RFE-pruned
- **Feature Engineering**: Lag, rolling, calendar, expanding features
- **Optimization**: Optuna for Bayesian hyperparameter tuning
- **Evaluation**: Multiple metrics (MSE, RMSE, R²)

### 5. Data Processing Layer

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                      │
├─────────────────────────────────────────────────────────────┤
│ • Column Mapping Engine                                     │
│ • Data Standardization                                      │
│ • Feature Generation                                        │
│ • Data Validation                                           │
│ • Encoding Management                                       │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

- **Universal Mapper**: Handles any CSV structure
- **Type Conversion**: Automatic data type inference
- **Feature Preservation**: Maintains original data as features
- **Encoding Maps**: Categorical to numeric conversion

## Component Design

### 1. Column Mapping System

**Design Pattern:** Strategy Pattern with Fallback Chain

```python
class ColumnMappingStrategy:
    def map_columns(self, df):
        raise NotImplementedError

class GeminiMappingStrategy(ColumnMappingStrategy):
    def map_columns(self, df):
        # AI-powered mapping
        pass

class SmartMappingStrategy(ColumnMappingStrategy):
    def map_columns(self, df):
        # Keyword-based mapping
        pass

class MappingOrchestrator:
    def __init__(self, strategies):
        self.strategies = strategies

    def execute_mapping(self, df):
        for strategy in self.strategies:
            try:
                result = strategy.map_columns(df)
                if result:
                    return result
            except Exception:
                continue
        return None
```

**Benefits:**

- **Extensibility**: Easy to add new mapping strategies
- **Robustness**: Multiple fallback options
- **Testability**: Each strategy can be tested independently

### 2. Feature Engineering Pipeline

**Design Pattern:** Pipeline Pattern with Configurable Stages

```python
class FeatureEngineeringPipeline:
    def __init__(self):
        self.stages = []

    def add_stage(self, stage):
        self.stages.append(stage)

    def process(self, data):
        for stage in self.stages:
            data = stage.transform(data)
        return data

class LagFeatureStage:
    def transform(self, data):
        # Generate lag features
        return data

class RollingFeatureStage:
    def transform(self, data):
        # Generate rolling features
        return data
```

**Pipeline Stages:**

1. **Data Preparation**: Sorting, grouping, cleaning
2. **Lag Generation**: Historical value features
3. **Rolling Generation**: Moving window statistics
4. **Calendar Generation**: Time-based features
5. **Expanding Generation**: Cumulative statistics
6. **Final Cleanup**: NaN handling, type conversion

### 3. Model Training Orchestrator

**Design Pattern:** Factory Pattern with Strategy Selection

```python
class ModelFactory:
    def create_model(self, model_type, params):
        if model_type == "original":
            return OriginalTargetModel(params)
        elif model_type == "log_transformed":
            return LogTransformedModel(params)
        elif model_type == "rfe_pruned":
            return RFEPrunedModel(params)

class ModelOrchestrator:
    def __init__(self, factory):
        self.factory = factory

    def train_models(self, data, model_types):
        results = {}
        for model_type in model_types:
            model = self.factory.create_model(model_type, params)
            results[model_type] = model.train(data)
        return results
```

**Model Variants:**

- **Original Target**: Direct sales prediction
- **Log-Transformed**: Handles skewed distributions
- **RFE-Pruned**: Optimized feature subset

### 4. Prediction Engine

**Design Pattern:** Template Method Pattern

```python
class PredictionEngine:
    def predict(self, model, data, steps):
        # Template method
        prepared_data = self.prepare_data(data)
        predictions = self.generate_predictions(model, prepared_data, steps)
        return self.format_results(predictions)

    def prepare_data(self, data):
        # Abstract method - implemented by subclasses
        pass

    def generate_predictions(self, model, data, steps):
        # Abstract method - implemented by subclasses
        pass

    def format_results(self, predictions):
        # Abstract method - implemented by subclasses
        pass

class RecursivePredictionEngine(PredictionEngine):
    def generate_predictions(self, model, data, steps):
        # Recursive forecasting implementation
        pass
```

## Data Flow Architecture

### 1. Request Flow

```
User Action → Streamlit Event → Business Logic → AI/ML Processing → Response
```

### 2. Error Handling Flow

```
Exception → Error Handler → Fallback Strategy → Graceful Degradation → User Notification
```

### 3. AI Integration Flow

```
Request → API Client → Prompt Engineering → Gemini API → Response Parsing → Validation
```

## Technical Decisions & Rationale

### 1. Framework Selection

**Streamlit:**

- **Pros**: Rapid prototyping, built-in widgets, easy deployment
- **Cons**: Limited customization, performance constraints
- **Rationale**: Perfect for ML demo and internal tools

**LightGBM:**

- **Pros**: Fast training, good performance, handles categorical data
- **Cons**: Less interpretable than some alternatives
- **Rationale**: Excellent for tabular data with mixed types

**Google Gemini:**

- **Pros**: Advanced reasoning, structured output, cost-effective
- **Cons**: API dependency, potential rate limits
- **Rationale**: Best balance of capability and cost for AI features

### 2. Architecture Patterns

**Pipeline Pattern:**

- **Why**: Modular, testable, extensible data processing
- **Implementation**: Feature engineering stages

**Strategy Pattern:**

- **Why**: Multiple approaches for same problem (mapping, models)
- **Implementation**: Column mapping strategies

**Factory Pattern:**

- **Why**: Create different model types dynamically
- **Implementation**: Model creation and training

**Template Method:**

- **Why**: Consistent prediction interface with different algorithms
- **Implementation**: Prediction engine variants

### 3. Error Handling Strategy

**Graceful Degradation:**

```python
# Primary: AI-powered mapping
try:
    mapping = gemini_mapping(df)
except Exception:
    # Fallback: Smart keyword mapping
    mapping = smart_mapping(df)
```

**Retry Logic:**

```python
for attempt in range(MAX_RETRIES):
    try:
        response = api_call()
        return response
    except Exception:
        time.sleep(RETRY_DELAY * (2 ** attempt))
```

**User Feedback:**

- Progress indicators for long operations
- Clear error messages with suggestions
- Fallback notifications

### 4. Performance Considerations

**Data Processing:**

- **Chunking**: Process large datasets in chunks
- **Caching**: Cache intermediate results
- **Parallelization**: Use pandas vectorized operations

**Model Training:**

- **Early Stopping**: Prevent overfitting
- **Feature Selection**: Reduce dimensionality
- **Parameter Optimization**: Efficient hyperparameter search

**Memory Management:**

- **Lazy Loading**: Load data only when needed
- **Garbage Collection**: Clear intermediate variables
- **Data Types**: Use appropriate data types (int32 vs int64)

## Security Considerations

### 1. API Key Management

```python
# Secure storage in Streamlit
api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Environment variable fallback
if not api_key:
    api_key = os.getenv('GEMINI_API_KEY')
```

### 2. Data Privacy

- **Local Processing**: All data processed locally
- **No Data Storage**: No persistent storage of uploaded files
- **Temporary Variables**: Clear sensitive data after processing

### 3. Input Validation

```python
# File type validation
if uploaded_file.type != "text/csv":
    st.error("Please upload a CSV file")

# Data size limits
if len(df) > MAX_ROWS:
    st.error(f"File too large. Maximum {MAX_ROWS} rows allowed")
```

## Scalability Considerations

### 1. Horizontal Scaling

- **Stateless Design**: No session dependencies
- **Containerization**: Docker deployment ready
- **Load Balancing**: Multiple instances support

### 2. Vertical Scaling

- **Memory Optimization**: Efficient data structures
- **CPU Utilization**: Parallel processing where possible
- **GPU Support**: Future enhancement for large models

### 3. Performance Monitoring

```python
# Timing decorators
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        st.write(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper
```

## Deployment Architecture

### 1. Local Development

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │───▶│   Local Files   │───▶│   Browser       │
│   Development   │    │   (CSV Data)    │    │   Interface     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Production Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│   Streamlit     │───▶│   File Storage  │
│   (Optional)    │    │   Instances     │    │   (S3/GCS)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Gemini API    │
                       │   (External)    │
                       └─────────────────┘
```

### 3. Containerization

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app2.py", "--server.port=8501"]
```

## Future Architecture Enhancements

### 1. Microservices Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │───▶│   Data Service  │───▶│   ML Service    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                       │
                              ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Cache Layer   │    │   Model Store   │
                       └─────────────────┘    └─────────────────┘
```

### 2. Event-Driven Architecture

- **Event Streaming**: Real-time data processing
- **Message Queues**: Asynchronous processing
- **Webhooks**: External system integration

### 3. AI/ML Pipeline Enhancement

- **Model Versioning**: Track model iterations
- **A/B Testing**: Compare model performance
- **AutoML**: Automated model selection
- **Ensemble Methods**: Combine multiple models

## Monitoring & Observability

### 1. Logging Strategy

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_model_performance(model_name, metrics):
    logger.info(f"Model {model_name}: {metrics}")
```

### 2. Metrics Collection

- **Model Performance**: Accuracy, training time
- **System Performance**: Memory usage, response time
- **User Behavior**: Feature usage, error rates

### 3. Alerting

- **Model Degradation**: Performance below thresholds
- **System Issues**: High error rates, slow responses
- **API Limits**: Approaching rate limits

This architecture provides a solid foundation for a robust, scalable, and maintainable sales forecasting application while maintaining flexibility for future enhancements.
