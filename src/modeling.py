import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, auc, mean_squared_error, 
    mean_absolute_error, r2_score  # Added r2_score here
)
import lightgbm as lgb
import joblib
import shap

class ModelTrainer:
    """
    Handles the training of machine learning models for loan risk prediction.
    """
    
    def __init__(self, model_config, feature_metadata):
        """
        Initialize the ModelTrainer with configuration.
        
        Args:
            model_config (dict): Configuration for model training
            feature_metadata (dict): Metadata about features (names, types)
        """
        self.model_config = model_config
        self.feature_metadata = feature_metadata
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train, y_train):
        """
        Train a model on the provided data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            object: Trained model pipeline
        """
        self.logger.info("Starting model training")
        
        try:
            # Get feature lists from metadata
            numeric_features = self.feature_metadata.get('numeric_features', [])
            categorical_features = self.feature_metadata.get('categorical_features', [])
            
            # Ensure features exist in the training data
            numeric_features = [f for f in numeric_features if f in X_train.columns]
            categorical_features = [f for f in categorical_features if f in X_train.columns]
            
            self.logger.info(f"Training with {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
            
            # Create preprocessing pipeline
            preprocessor = self._create_preprocessor(numeric_features, categorical_features)
            
            # Determine model type based on target
            model_type = self.model_config.get('type', 'lightgbm').lower()
            
            # Create model
            if model_type == 'lightgbm':
                # Check if we need classification or regression
                model = self._create_lightgbm_model(y_train)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # Discretize target for classification if needed
            y_train_processed = self._process_target(y_train)
            
            # Train the model
            self.logger.info("Fitting model pipeline")
            pipeline.fit(X_train, y_train_processed)
            
            # Log feature importance if available
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                self._log_feature_importance(pipeline, X_train)
            
            self.logger.info("Model training completed successfully")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}", exc_info=True)
            raise
    
    def _create_preprocessor(self, numeric_features, categorical_features):
        """
        Create a preprocessing pipeline for the features.
        
        Args:
            numeric_features (list): List of numeric feature names
            categorical_features (list): List of categorical feature names
            
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        # Create transformers for different feature types
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers in a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop columns not specified
        )
        
        return preprocessor
    

    def _create_lightgbm_model(self, y_train):
        """
        Create a LightGBM model with appropriate parameters.
        
        Args:
            y_train (pd.Series): Training target for determining model type
            
        Returns:
            object: Configured LightGBM model
        """
        # Get hyperparameters from config, with defaults
        params = self.model_config.get('hyperparameters', {})
        
        # Determine if we need classification or regression
        # Based on the number of unique values and their distribution
        unique_values = y_train.nunique()
        is_classification = unique_values <= 10  # Adjust this threshold as needed
        
        if is_classification:
            self.logger.info("Using LightGBM Classifier (discrete target)")
            model = lgb.LGBMClassifier(
                objective='multiclass' if unique_values > 2 else 'binary',
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 20),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                reg_alpha=params.get('reg_alpha', 0.0),
                reg_lambda=params.get('reg_lambda', 0.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
        else:
            self.logger.info("Using LightGBM Regressor (continuous target)")
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 20),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                reg_alpha=params.get('reg_alpha', 0.0),
                reg_lambda=params.get('reg_lambda', 0.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
        
        return model
    
    
    def _create_model(self):
        """
        Create a model instance with the configured hyperparameters.
        
        Returns:
            object: Model instance
        """
        model_type = self.model_config.get('type', 'lightgbm')
        
        if model_type.lower() == 'lightgbm':
            # Get hyperparameters from config, with defaults
            params = self.model_config.get('hyperparameters', {})
            
            # Create LightGBM model
            model = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 20),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                reg_alpha=params.get('reg_alpha', 0.0),
                reg_lambda=params.get('reg_lambda', 0.0),
                random_state=params.get('random_state', 42),
                verbose=-1
            )
            
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _log_feature_importance(self, pipeline, X):
        """
        Log feature importance from the trained model.
        
        Args:
            pipeline (Pipeline): Trained model pipeline
            X (pd.DataFrame): Feature data used for training
        """
        model = pipeline.named_steps['classifier']
        
        # Get feature names after preprocessing
        # This is complex due to one-hot encoding
        feature_names = self._get_feature_names(pipeline, X)
        
        if len(feature_names) == len(model.feature_importances_):
            # Create DataFrame of feature importances
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            self.logger.info(f"Top 10 features by importance:\n{importance_df.head(10)}")
        else:
            self.logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match "
                f"feature importance length ({len(model.feature_importances_)})"
            )
    
    def _get_feature_names(self, pipeline, X):
        """
        Extract feature names after preprocessing.
        
        Args:
            pipeline (Pipeline): Trained model pipeline
            X (pd.DataFrame): Feature data
            
        Returns:
            list: Feature names after preprocessing
        """
        # This is a simplified version - actual implementation may need adjustments
        # based on scikit-learn version and preprocessing steps
        feature_names = []
        
        try:
            # Try to get feature names from the preprocessor
            preprocessor = pipeline.named_steps['preprocessor']
            
            # Loop through each transformer
            for name, transformer, features in preprocessor.transformers_:
                if name == 'num':
                    # For numeric features, names remain the same
                    feature_names.extend(features)
                elif name == 'cat' and len(features) > 0:
                    # For categorical features, get the one-hot encoded names
                    encoder = transformer.named_steps['onehot']
                    
                    # Handle different scikit-learn versions
                    if hasattr(encoder, 'get_feature_names_out'):
                        encoded_names = encoder.get_feature_names_out(features)
                    elif hasattr(encoder, 'get_feature_names'):
                        encoded_names = encoder.get_feature_names(features)
                    else:
                        # Fallback if methods not available
                        encoded_names = [f"{f}_encoded" for f in features]
                    
                    feature_names.extend(encoded_names)
        except Exception as e:
            self.logger.warning(f"Error getting feature names: {str(e)}")
            # Fallback to generic names
            feature_names = [f"feature_{i}" for i in range(len(X.columns))]
        
        return feature_names
    
    def _process_target(self, y_train):
        """
        Process target variable for model training.
        
        Args:
            y_train (pd.Series): Original target values
            
        Returns:
            pd.Series: Processed target values
        """
        # Check number of unique values
        unique_values = y_train.nunique()
        
        # If continuous, create risk buckets
        if unique_values > 10:
            # Create risk buckets or categorize
            y_processed = pd.cut(y_train, bins=5, labels=False)
            self.logger.info("Converted continuous target to categorical buckets")
            return y_processed
        
        return y_train


class ModelEvaluator:
    """
    Evaluates trained models and compares performance.
    """
    
    def __init__(self, eval_config):
        """
        Initialize the ModelEvaluator with configuration.
        
        Args:
            eval_config (dict): Configuration for model evaluation
        """
        self.eval_config = eval_config
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, model, X_test, y_test):
        """
        Evaluate a model on test data.
        
        Args:
            model (object): Trained model pipeline
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Evaluating model on test data")
        
        try:
            # Ensure all required missing indicator columns exist
            X_test_processed = self._ensure_missing_columns(X_test)
            
            # Determine model type
            final_estimator = model.named_steps['classifier'] if 'classifier' in model.named_steps else model.named_steps['regressor']
            
            # Get predictions
            y_pred = model.predict(X_test_processed)
            
            # Discretize continuous predictions and true values
            risk_thresholds = self.eval_config.get('risk_thresholds', {
                'low_risk_max': 0.3,
                'high_risk_min': 0.7
            })
            
            def discretize_risk(values):
                """
                Convert continuous risk scores to categorical risk levels
                """
                discrete_values = np.zeros_like(values, dtype=int)
                discrete_values[values <= risk_thresholds['low_risk_max']] = 0  # Low Risk
                discrete_values[(values > risk_thresholds['low_risk_max']) & 
                                (values <= risk_thresholds['high_risk_min'])] = 1  # Medium Risk
                discrete_values[values > risk_thresholds['high_risk_min']] = 2  # High Risk
                return discrete_values
            
            y_pred_discrete = discretize_risk(y_pred)
            y_test_discrete = discretize_risk(y_test)
            
            # Regression/Continuous model metrics
            results = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Classification metrics on discretized values
            results.update({
                'accuracy': accuracy_score(y_test_discrete, y_pred_discrete),
                'precision_macro': precision_score(y_test_discrete, y_pred_discrete, average='macro'),
                'recall_macro': recall_score(y_test_discrete, y_pred_discrete, average='macro'),
                'f1_macro': f1_score(y_test_discrete, y_pred_discrete, average='macro'),
                'confusion_matrix': confusion_matrix(y_test_discrete, y_pred_discrete).tolist(),
                'classification_report': classification_report(y_test_discrete, y_pred_discrete, output_dict=True)
            })
            
            # Percentile-based analysis
            try:
                y_pred_percentiles = np.array([stats.percentileofscore(y_test, pred) for pred in y_pred]) / 100
                
                results.update({
                    'mean_prediction': float(y_pred.mean()),
                    'std_prediction': float(y_pred.std()),
                    'percentile_scores': {
                        'mean': float(y_pred_percentiles.mean()),
                        'min': float(y_pred_percentiles.min()),
                        'max': float(y_pred_percentiles.max()),
                        'std': float(y_pred_percentiles.std())
                    }
                })
            except Exception as percentile_err:
                self.logger.warning(f"Could not compute percentile scores: {percentile_err}")
                results.update({
                    'mean_prediction': float(y_pred.mean()),
                    'std_prediction': float(y_pred.std())
                })
            
            # Risk categorization
            results['risk_categories'] = {
                'low_risk': float((y_pred_discrete == 0).mean() * 100),
                'medium_risk': float((y_pred_discrete == 1).mean() * 100),
                'high_risk': float((y_pred_discrete == 2).mean() * 100)
            }
            
            self.logger.info("Model evaluation completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            raise

    def _ensure_missing_columns(self, X_test):
        """
        Ensure all missing indicator columns exist in the test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            
        Returns:
            pd.DataFrame: Test features with added missing indicator columns
        """
        # List of columns that should have missing indicators
        missing_indicator_columns = [
            'days_to_origination', 'nPaidOff', 'clearfraudscore', 
            'application_year', 'application_month', 'application_day', 
            'application_dayofweek', 'application_hour', 
            'loanAmount', 'apr', 'leadCost'
        ]
        
        # Make a copy of the input dataframe
        X_test_processed = X_test.copy()
        
        # Add missing columns with default value 0
        for col in missing_indicator_columns:
            missing_col = f"{col}_missing"
            if missing_col not in X_test_processed.columns:
                X_test_processed[missing_col] = 0
        
        return X_test_processed
    
    def compare_models(self, new_model, production_model, X_test, y_test):
        """
        Compare a new model against the production model.
        
        Args:
            new_model (object): New trained model
            production_model (object): Current production model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Comparison results
        """
        self.logger.info("Comparing new model against production model")
        
        try:
            # Evaluate both models
            new_results = self.evaluate(new_model, X_test, y_test)
            prod_results = self.evaluate(production_model, X_test, y_test)
            
            # Comparison of metrics
            comparison = {
                'metric_diffs': {
                    # Regression metrics
                    'mse_diff': prod_results.get('mse', 0) - new_results.get('mse', 0),  # Lower is better
                    'mae_diff': prod_results.get('mae', 0) - new_results.get('mae', 0),  # Lower is better
                    'r2_diff': new_results.get('r2', 0) - prod_results.get('r2', 0),
                    
                    # Classification metrics
                    'accuracy_diff': new_results.get('accuracy', 0) - prod_results.get('accuracy', 0),
                    'precision_macro_diff': new_results.get('precision_macro', 0) - prod_results.get('precision_macro', 0),
                    'recall_macro_diff': new_results.get('recall_macro', 0) - prod_results.get('recall_macro', 0),
                    'f1_macro_diff': new_results.get('f1_macro', 0) - prod_results.get('f1_macro', 0)
                },
                'new_model': new_results,
                'production_model': prod_results
            }
            
            # Determine if new model is better
            improvement_threshold = self.eval_config.get('improvement_threshold', 0.01)
            is_better = (
                comparison['metric_diffs']['r2_diff'] > improvement_threshold and
                comparison['metric_diffs']['accuracy_diff'] > 0
            )
            comparison['is_better'] = is_better
            
            if is_better:
                self.logger.info("New model outperforms production model")
            else:
                self.logger.info("New model does not significantly outperform production model")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}", exc_info=True)
            raise
    
    def generate_report(self, model, X_test, y_test, evaluation_results, output_path):
        """
        Generate an HTML evaluation report with visualizations.
        
        Args:
            model (object): Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            evaluation_results (dict): Evaluation metrics
            output_path (str): Path to save the report
            
        Returns:
            bool: True if report was generated successfully
        """
        self.logger.info(f"Generating evaluation report to {output_path}")
        
        try:
            # Determine model type
            final_estimator = model.named_steps['classifier'] if 'classifier' in model.named_steps else model.named_steps['regressor']
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate plots
            self._create_prediction_plots(y_test, y_pred, output_path)
            
            # Generate feature importance plot if possible
            self._create_feature_importance_plot(model, X_test, output_path)
            
            # Prepare risk-related content
            risk_thresholds = self.eval_config.get('risk_thresholds', {
                'low_risk_max': 0.3,
                'high_risk_min': 0.7
            })
            
            def discretize_risk(values):
                """Convert continuous values to risk categories"""
                discrete_values = np.zeros_like(values, dtype=int)
                discrete_values[values <= risk_thresholds['low_risk_max']] = 0  # Low Risk
                discrete_values[(values > risk_thresholds['low_risk_max']) & 
                                (values <= risk_thresholds['high_risk_min'])] = 1  # Medium Risk
                discrete_values[values > risk_thresholds['high_risk_min']] = 2  # High Risk
                return discrete_values
            
            y_pred_discrete = discretize_risk(y_pred)
            y_test_discrete = discretize_risk(y_test)
            
            # Create HTML report
            html_content = f"""
            <html>
            <head>
                <title>Model Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; }}
                    .good {{ color: green; }}
                    .bad {{ color: red; }}
                    img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>Model Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td class="metric">Mean Squared Error (MSE)</td><td>{evaluation_results.get('mse', 'N/A'):.4f}</td></tr>
                    <tr><td class="metric">Mean Absolute Error (MAE)</td><td>{evaluation_results.get('mae', 'N/A'):.4f}</td></tr>
                    <tr><td class="metric">R² Score</td><td>{evaluation_results.get('r2', 'N/A'):.4f}</td></tr>
                    <tr><td class="metric">Accuracy</td><td>{evaluation_results.get('accuracy', 'N/A'):.4f}</td></tr>
                    <tr><td class="metric">Macro Precision</td><td>{evaluation_results.get('precision_macro', 'N/A'):.4f}</td></tr>
                    <tr><td class="metric">Macro Recall</td><td>{evaluation_results.get('recall_macro', 'N/A'):.4f}</td></tr>
                    <tr><td class="metric">Macro F1 Score</td><td>{evaluation_results.get('f1_macro', 'N/A'):.4f}</td></tr>
                </table>
                
                <h2>Risk Distribution</h2>
                <table>
                    <tr><th>Risk Category</th><th>Predicted (%)</th><th>Actual (%)</th></tr>
                    <tr>
                        <td>Low Risk</td>
                        <td>{evaluation_results['risk_categories'].get('low_risk', 'N/A'):.2f}%</td>
                        <td>{float((y_test_discrete == 0).mean() * 100):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Medium Risk</td>
                        <td>{evaluation_results['risk_categories'].get('medium_risk', 'N/A'):.2f}%</td>
                        <td>{float((y_test_discrete == 1).mean() * 100):.2f}%</td>
                    </tr>
                    <tr>
                        <td>High Risk</td>
                        <td>{evaluation_results['risk_categories'].get('high_risk', 'N/A'):.2f}%</td>
                        <td>{float((y_test_discrete == 2).mean() * 100):.2f}%</td>
                    </tr>
                </table>
                
                <h2>Visualizations</h2>
                <h3>Predicted vs Actual Risk Scores</h3>
                <img src="predicted_vs_actual.png" alt="Predicted vs Actual Risk Scores">
                
                <h3>Risk Score Distribution</h3>
                <img src="risk_score_distribution.png" alt="Risk Score Distribution">
                
                <h3>Feature Importance</h3>
                <img src="feature_importance.png" alt="Feature Importance">
            </body>
            </html>
            """
            
            # Write HTML to file
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Evaluation report generated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}", exc_info=True)
            return False
    
    def _create_prediction_plots(self, y_test, y_pred, output_dir):
        """
        Create prediction-related visualizations.
        
        Args:
            y_test (np.array): True target values
            y_pred (np.array): Predicted target values
            output_dir (str): Directory to save plots
        """
        # Create output directory if it doesn't exist
        base_dir = os.path.dirname(output_dir)
        
        # 1. Predicted vs Actual Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Risk Scores')
        plt.ylabel('Predicted Risk Scores')
        plt.title('Predicted vs Actual Risk Scores')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'predicted_vs_actual.png'))
        plt.close()
        
        # 2. Risk Score Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_test, bins=30, alpha=0.5, label='Actual')
        plt.hist(y_pred, bins=30, alpha=0.5, label='Predicted')
        plt.xlabel('Risk Scores')
        plt.ylabel('Frequency')
        plt.title('Distribution of Actual vs Predicted Risk Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'risk_score_distribution.png'))
        plt.close()

    def _create_feature_importance_plot(self, model, X_test, output_dir):
        """
        Create a feature importance plot based on the model's feature importances.
        
        Args:
            model (object): Trained model
            X_test (pd.DataFrame): Test features
            output_dir (str): Directory to save the plot
        """
        try:
            # Extract the classifier from the pipeline
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
            elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                classifier = model.named_steps['regressor']
            else:
                classifier = model
            
            # Check if model has feature_importances_
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                
                # Get feature names
                if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                    # Attempt to get feature names from preprocessor
                    preprocessor = model.named_steps['preprocessor']
                    feature_names = []
                    
                    # Handle numeric features
                    if hasattr(preprocessor, 'transformers_'):
                        for name, _, features in preprocessor.transformers_:
                            if name == 'num':
                                feature_names.extend(features)
                            elif name == 'cat':
                                # For categorical features, get one-hot encoded names
                                encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                                if hasattr(encoder, 'get_feature_names_out'):
                                    cat_feature_names = encoder.get_feature_names_out(features)
                                    feature_names.extend(cat_feature_names)
                else:
                    # Fallback to X_test columns
                    feature_names = X_test.columns.tolist()
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                top_n = min(20, len(importances))
                
                # Plot feature importances
                plt.figure(figsize=(10, 8))
                plt.title("Top Feature Importances")
                plt.barh(range(top_n), importances[indices[:top_n]])
                plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
                plt.xlabel("Relative Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(output_dir), 'feature_importance.png'))
                plt.close()
                
                self.logger.info("Feature importance plot created successfully")
            else:
                self.logger.warning("Model does not have feature_importances_ attribute")
                
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")

    def _create_feature_importance_plot(self, model, X_test, output_dir):
        """
        Create a feature importance plot based on the model's feature importances.
        
        Args:
            model (object): Trained model
            X_test (pd.DataFrame): Test features for reference
            output_dir (str): Directory to save the plot
        """
        try:
            # Extract the classifier from the pipeline
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
            else:
                classifier = model
            
            # Check if model has feature_importances_
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Get feature names if available
                if hasattr(classifier, 'feature_name_'):
                    feature_names = classifier.feature_name_
                elif hasattr(X_test, 'columns'):
                    feature_names = X_test.columns
                else:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                # Plot top 20 features or all features if less
                n_features = min(20, len(importances))
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(n_features), importances[indices[:n_features]], align='center')
                
                # Add feature names to y-axis, ensuring we don't go out of bounds
                labels = []
                for i in indices[:n_features]:
                    if i < len(feature_names):
                        labels.append(str(feature_names[i]))
                    else:
                        labels.append(f"feature_{i}")
                
                plt.yticks(range(n_features), labels)
                plt.xlabel('Relative Importance')
                plt.title('Feature Importance')
                plt.tight_layout()
                
                base_dir = os.path.dirname(output_dir)
                plt.savefig(os.path.join(base_dir, 'feature_importance.png'))
                plt.close()
                
                self.logger.info("Feature importance plot created successfully")
            else:
                self.logger.warning("Model does not have feature_importances_ attribute, cannot create feature importance plot")
                
        except Exception as e:
            self.logger.error(f"Error creating feature importance plot: {str(e)}")