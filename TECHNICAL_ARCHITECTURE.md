# Technical Architecture Documentation

## System Overview

The Sales Forecasting Application is built as a modular, AI-enhanced machine learning system that combines traditional ML techniques with modern AI capabilities. The architecture follows a pipeline-based approach with multiple fallback mechanisms for robustness and **advanced data-driven hyperparameter optimization**.

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
│ • Advanced Prompt Engineering                               │
│ • Response Parser                                           │
│ • Error Handler                                             │
│ • Fallback Manager                                          │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

- **API Client**: Google Generative AI integration
- **Advanced Prompt Templates**: **Data-driven prompts with domain-specific guidance**
- **Response Processing**: JSON parsing and validation
- **Retry Logic**: Exponential backoff for API failures

### 4. Machine Learning Layer

```
┌─────────────────────────────────────────────────────────────┐
│                Machine Learning Layer                       │
├─────────────────────────────────────────────────────────────┤
│ • LightGBM Models                                          │
│ • Feature Engineering Pipeline                              │
│ • Advanced Hyperparameter Optimization                      │
│ • Model Evaluation Engine                                   │
│ • Prediction Algorithms                                     │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

- **Model Variants**: Original, Log-transformed, AI-pruned
- **Feature Engineering**: Lag, rolling, calendar, expanding features
- **Advanced Optimization**: **Data-driven hyperparameter tuning with Optuna**
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

### 3. Advanced Hyperparameter Optimization System

**Design Pattern:** Data-Driven Strategy with Domain Knowledge

```python
class HyperparameterOptimizer:
    def __init__(self, data_analyzer, ai_suggester, optimizer):
        self.data_analyzer = data_analyzer
        self.ai_suggester = ai_suggester
        self.optimizer = optimizer

    def optimize(self, data, features):
        # 1. Analyze data characteristics
        data_stats = self.data_analyzer.analyze(data)

        # 2. Get AI-suggested parameter ranges
        param_ranges = self.ai_suggester.suggest_ranges(data_stats, features)

        # 3. Perform Bayesian optimization
        best_params = self.optimizer.optimize(param_ranges)

        return best_params

class DataAnalyzer:
    def analyze(self, data):
        return {
            'size_category': self._categorize_size(len(data)),
            'sales_stats': data['sales'].describe(),
            'sparsity': self._calculate_sparsity(data),
            'variance': self._calculate_variance(data)
        }
```

**Key Features:**

- **Data-Driven Analysis**: Analyzes dataset characteristics (size, sparsity, variance)
- **Domain-Specific Guidance**: Sales forecasting specific parameter recommendations
- **Adaptive Ranges**: Parameters scale based on data characteristics
- **Computational Efficiency**: Balances accuracy with training time

### 4. Model Training Orchestrator

**Design Pattern:** Factory Pattern with Strategy Selection

```python
class ModelFactory:
    def create_model(self, model_type, params):
        if model_type == "original":
            return OriginalTargetModel(params)
        elif model_type == "log_transformed":
            return LogTransformedModel(params)
        elif model_type == "ai_pruned":
            return AIPrunedModel(params)

class ModelOrchestrator:
    def __init__(self, factory, evaluator):
        self.factory = factory
        self.evaluator = evaluator

    def train_and_evaluate(self, data, features):
        models = {}

        # Train multiple model variants
        for model_type in ["original", "log_transformed", "ai_pruned"]:
            model = self.factory.create_model(model_type, params)
            models[model_type] = self.evaluator.evaluate(model, data)

        # Select best model
        return self.select_best_model(models)
```

## Data Flow Architecture

### 1. Data Ingestion Flow

```
CSV Upload → Validation → Column Mapping → Standardization → Feature Engineering
```

**Key Components:**

- **Validation**: File format, data quality checks
- **Mapping**: AI-powered or fallback column identification
- **Standardization**: Convert to standard format
- **Engineering**: Generate comprehensive feature set

### 2. Model Training Flow

```
Feature Set → Data Analysis → Hyperparameter Optimization → Model Training → Evaluation
```

**Key Components:**

- **Data Analysis**: **Comprehensive dataset characterization**
- **Hyperparameter Optimization**: **Data-driven parameter tuning**
- **Model Training**: Multiple model variants
- **Evaluation**: Performance comparison and selection

### 3. Prediction Flow

```
Input Data → Feature Generation → Model Prediction → Post-processing → Output
```

**Key Components:**

- **Feature Generation**: Real-time feature engineering
- **Model Prediction**: Use best trained model
- **Post-processing**: Format results for display

## AI Integration Architecture

### 1. Prompt Engineering Strategy

**Enhanced Prompt Design:**

```python
class PromptEngineer:
    def create_hyperparameter_prompt(self, data_stats, features):
        return f"""
        You are an expert data scientist specializing in sales forecasting with LightGBM.

        **DATA ANALYSIS:**
        - Dataset size: {data_stats['size']} rows, {len(features)} features
        - Sales statistics: {data_stats['sales_stats']}
        - Sparsity: {data_stats['sparsity']}
        - Variance: {data_stats['variance']}

        **SALES FORECASTING CONTEXT:**
        - Time series regression with seasonality and trends
        - High variance data with potential outliers
        - Balance between pattern capture and overfitting

        **HYPERPARAMETER GUIDANCE:**
        [Detailed parameter-specific guidance based on data characteristics]

        **OUTPUT FORMAT:**
        Return ONLY a valid Python dictionary for Optuna optimization.
        """
```

**Key Improvements:**

- **Data-Driven Context**: Uses actual data statistics
- **Domain Knowledge**: Sales forecasting specific guidance
- **Parameter Relationships**: Explains parameter dependencies
- **Clear Output Format**: Structured response requirements

### 2. AI Response Processing

```python
class AIResponseProcessor:
    def process_hyperparameter_response(self, response):
        try:
            # Extract dictionary from response
            code = self.extract_code(response)

            # Validate parameter ranges
            validated_ranges = self.validate_ranges(code)

            return validated_ranges
        except Exception as e:
            return self.get_fallback_ranges()
```

### 3. Fallback Mechanisms

```python
class FallbackManager:
    def __init__(self):
        self.fallbacks = {
            'column_mapping': SmartColumnMapping(),
            'hyperparameters': DefaultHyperparameters(),
            'feature_selection': AllFeaturesSelection()
        }

    def get_fallback(self, operation):
        return self.fallbacks.get(operation)
```

## Performance Optimization

### 1. Computational Efficiency

**Strategies:**

- **Early Stopping**: Prevents overfitting and reduces training time
- **Feature Selection**: Reduces dimensionality and improves speed
- **Batch Processing**: Efficient handling of large datasets
- **Memory Management**: Optimized data structures and cleanup

### 2. Model Performance

**Optimization Techniques:**

- **Hyperparameter Tuning**: **Data-driven optimization with domain guidance**
- **Cross-Validation**: Robust performance estimation
- **Ensemble Methods**: Combine multiple model predictions
- **Regularization**: Prevent overfitting while maintaining accuracy

### 3. Scalability Considerations

**Architecture Decisions:**

- **Modular Design**: Easy to scale individual components
- **Stateless Operations**: Enables horizontal scaling
- **Caching**: Store intermediate results for reuse
- **Async Processing**: Non-blocking operations for better UX

## Security and Reliability

### 1. Error Handling

**Comprehensive Error Management:**

```python
class ErrorHandler:
    def handle_ai_failure(self, operation, error):
        # Log error details
        self.logger.error(f"AI {operation} failed: {error}")

        # Use fallback mechanism
        fallback = self.fallback_manager.get_fallback(operation)
        return fallback.execute()

    def handle_data_error(self, error):
        # Validate and clean data
        return self.data_validator.fix_data(error)
```

### 2. Data Validation

**Multi-Level Validation:**

- **Input Validation**: File format, data types, required columns
- **Business Logic Validation**: Sales data consistency, date ranges
- **Model Validation**: Feature quality, prediction sanity checks

### 3. API Security

**Security Measures:**

- **API Key Management**: Secure storage and rotation
- **Rate Limiting**: Prevent abuse and manage costs
- **Input Sanitization**: Prevent injection attacks
- **Error Masking**: Don't expose sensitive information

## Monitoring and Observability

### 1. Performance Monitoring

**Key Metrics:**

- **Training Time**: Model training duration
- **Prediction Latency**: Time to generate predictions
- **Memory Usage**: Resource consumption tracking
- **Accuracy Metrics**: Model performance over time

### 2. Error Tracking

**Error Categories:**

- **Data Errors**: Invalid input, missing values
- **AI Errors**: API failures, parsing errors
- **Model Errors**: Training failures, prediction errors
- **System Errors**: Memory, network, configuration issues

### 3. User Experience Monitoring

**UX Metrics:**

- **Success Rate**: Percentage of successful operations
- **Response Time**: Time to complete user requests
- **Error Recovery**: Ability to recover from failures
- **User Satisfaction**: Feedback and usage patterns

## Future Architecture Considerations

### 1. Scalability Enhancements

**Planned Improvements:**

- **Microservices Architecture**: Decompose into smaller services
- **Containerization**: Docker containers for easy deployment
- **Load Balancing**: Distribute load across multiple instances
- **Database Integration**: Persistent storage for models and data

### 2. Advanced AI Integration

**Future Capabilities:**

- **Multi-Model AI**: Combine multiple AI services
- **Real-time Learning**: Continuous model updates
- **Automated Feature Engineering**: AI-driven feature creation
- **Explainable AI**: Model interpretability and insights

### 3. Enterprise Features

**Business Requirements:**

- **Multi-tenancy**: Support multiple organizations
- **Role-based Access**: User permissions and security
- **Audit Logging**: Track all operations and changes
- **Integration APIs**: Connect with existing business systems

---

This architecture provides a robust, scalable, and maintainable foundation for the sales forecasting application, with **advanced AI integration and data-driven optimization** at its core.
