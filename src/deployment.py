import os
import logging
import json
import shutil
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy data types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class ModelDeployer:
    """
    Handles the deployment of trained models to production.
    """
    
    def __init__(self, deployment_config):
        """
        Initialize the ModelDeployer with configuration.
        
        Args:
            deployment_config (dict): Configuration for model deployment
        """
        self.deployment_config = deployment_config
        self.logger = logging.getLogger(__name__)
    
    def should_deploy(self, evaluation_results):
        """
        Determine if a model should be deployed based on evaluation results.
        
        Args:
            evaluation_results (dict): Results from model evaluation
            
        Returns:
            bool: True if the model should be deployed
        """
        self.logger.info("Checking if model meets deployment criteria")
        
        # Get deployment thresholds from config
        thresholds = self.deployment_config.get('thresholds', {})
        
        # Default thresholds
        min_accuracy = thresholds.get('min_accuracy', 0.7)
        min_roc_auc = thresholds.get('min_roc_auc', 0.75)
        min_precision = thresholds.get('min_precision', 0.6)
        min_r2 = thresholds.get('min_r2', 0.5)  # Added R² threshold for regression
        
        # First check if there is an existing production model
        # If not, deploy the new model if it meets minimum criteria
        production_path = self.deployment_config.get('production_model_path', 'models/production')
        model_file = os.path.join(production_path, 'model.joblib')
        
        if not os.path.exists(model_file):
            self.logger.info("No existing production model found. Will deploy if minimum criteria are met.")
            # Skip comparison step
        # If there is an existing model and comparison results available, check for improvement
        elif 'comparison' in evaluation_results:
            primary_metric = self.deployment_config.get('primary_metric', 'roc_auc')
            improvement_threshold = thresholds.get('improvement_threshold', 0.01)
            
            comparison = evaluation_results['comparison']
            metric_diff = comparison['metric_diffs'].get(f'{primary_metric}_diff', 0)
            
            # If new model isn't better by the threshold, don't deploy
            if metric_diff < improvement_threshold:
                self.logger.info(f"New model does not show sufficient improvement in {primary_metric} ({metric_diff:.4f} < {improvement_threshold})")
                return False
        
        # Check absolute performance thresholds
        # First check if this is a regression model (has R² but no ROC AUC)
        is_regression = 'r2' in evaluation_results and 'roc_auc' not in evaluation_results
        
        if is_regression:
            # Check regression metrics
            if 'accuracy' in evaluation_results and evaluation_results['accuracy'] < min_accuracy:
                self.logger.info(f"Model does not meet minimum accuracy threshold ({evaluation_results['accuracy']:.4f} < {min_accuracy})")
                return False
            
            if evaluation_results['r2'] < min_r2:
                self.logger.info(f"Model does not meet minimum R² threshold ({evaluation_results['r2']:.4f} < {min_r2})")
                return False
            
            # For regression models, we might not have precision
            if 'precision_macro' in evaluation_results and evaluation_results['precision_macro'] < min_precision:
                self.logger.info(f"Model does not meet minimum precision threshold ({evaluation_results['precision_macro']:.4f} < {min_precision})")
                return False
        else:
            # Check classification metrics
            if evaluation_results['accuracy'] < min_accuracy:
                self.logger.info(f"Model does not meet minimum accuracy threshold ({evaluation_results['accuracy']:.4f} < {min_accuracy})")
                return False
            
            if 'roc_auc' in evaluation_results and evaluation_results['roc_auc'] < min_roc_auc:
                self.logger.info(f"Model does not meet minimum ROC AUC threshold ({evaluation_results['roc_auc']:.4f} < {min_roc_auc})")
                return False
            
            if evaluation_results['precision'] < min_precision:
                self.logger.info(f"Model does not meet minimum precision threshold ({evaluation_results['precision']:.4f} < {min_precision})")
                return False
        
        # All criteria passed
        self.logger.info("Model meets all deployment criteria")
        return True
    
    def deploy(self, model, model_metadata):
        """
        Deploy a model to production.
        
        Args:
            model (object): Trained model to deploy
            model_metadata (dict): Metadata about the model
            
        Returns:
            dict: Deployment information
        """
        self.logger.info("Deploying model to production")
        
        try:
            # Get production model path from config
            production_path = self.deployment_config.get('production_model_path', 'models/production')
            
            # Create production directory if it doesn't exist
            os.makedirs(production_path, exist_ok=True)
            
            # Generate model version
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save current production model as backup if it exists
            model_file = os.path.join(production_path, 'model.joblib')
            if os.path.exists(model_file):
                backup_dir = os.path.join(production_path, 'backups')
                os.makedirs(backup_dir, exist_ok=True)
                backup_file = os.path.join(backup_dir, f'model_{version}.joblib')
                shutil.copy2(model_file, backup_file)
                self.logger.info(f"Backed up previous production model to {backup_file}")
            else:
                self.logger.info("First deployment: no existing model to back up")
            
            # Deploy new model to production
            joblib.dump(model, model_file)
            
            # Update model metadata
            model_metadata['deployment_date'] = datetime.now().isoformat()
            model_metadata['model_version'] = version
            
            # Save metadata
            metadata_file = os.path.join(production_path, 'model_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(model_metadata, f, indent=2, cls=NumpyEncoder)
            
            # Create deployment info
            deployment_info = {
                'model_version': version,
                'deployment_date': datetime.now().isoformat(),
                'model_path': model_file,
                'metadata_path': metadata_file,
                'is_first_deployment': not os.path.exists(os.path.join(production_path, 'backups'))
            }
            
            # Save deployment registry
            self._update_deployment_registry(deployment_info)
            
            self.logger.info(f"Model successfully deployed to production with version {version}")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {str(e)}", exc_info=True)
            raise
    
    def _update_deployment_registry(self, deployment_info):
        """
        Update the deployment registry with new deployment info.
        
        Args:
            deployment_info (dict): Information about the deployment
        """
        registry_path = self.deployment_config.get('registry_path', 'models/registry.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Load existing registry if it exists
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'deployments': []}
        
        # Add new deployment
        registry['deployments'].append(deployment_info)
        registry['current_version'] = deployment_info['model_version']
        
        # Save updated registry
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        self.logger.info(f"Deployment registry updated at {registry_path}")
    
    def serve_prediction(self, data):
        """
        Serve predictions using the deployed model.
        
        Args:
            data (pd.DataFrame or dict): Data to predict on
            
        Returns:
            dict: Prediction results
        """
        self.logger.info("Serving prediction request")
        
        try:
            # Get production model path
            production_path = self.deployment_config.get('production_model_path', 'models/production')
            model_file = os.path.join(production_path, 'model.joblib')
            
            # Check if model exists
            if not os.path.exists(model_file):
                self.logger.error(f"Production model not found at {model_file}")
                raise FileNotFoundError(f"Production model not found at {model_file}. Please deploy a model first.")
            
            # Load the model
            model = joblib.load(model_file)
            
            # Convert input to DataFrame if it's a dict
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Determine if this is a regression model by looking at the model or metadata
            is_regression = False
            
            # 1. First check metadata for is_regression flag
            metadata_file = os.path.join(production_path, 'model_metadata.json')
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    is_regression = metadata.get('is_regression', False)
                except Exception:
                    pass
            
            # 2. If not determined by metadata, check if the model has predict_proba
            if not is_regression:
                # Check if final estimator has predict_proba (classification) or not (regression)
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    # It's a pipeline with a classifier
                    is_regression = not hasattr(model.named_steps['classifier'], 'predict_proba')
                elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
                    # It's a pipeline with a regressor
                    is_regression = True
                else:
                    # It's a direct model, check if it has predict_proba
                    is_regression = not hasattr(model, 'predict_proba')
            
            # Make prediction based on model type
            predictions = model.predict(data)
            
            if is_regression:
                # For regression models
                risk_thresholds = {
                    'low_risk_max': 0.3,
                    'high_risk_min': 0.7
                }
                
                # Create response
                prediction_results = {
                    'predictions': predictions.tolist(),
                    'threshold': self.deployment_config.get('prediction_threshold', 0.5),
                    # Add risk categories based on thresholds
                    'risk_categories': [
                        'high_risk' if p > risk_thresholds['high_risk_min'] else 
                        'medium_risk' if p > risk_thresholds['low_risk_max'] else 
                        'low_risk' for p in predictions
                    ],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # For classification models
                try:
                    # Try to get probabilities if available
                    probabilities = model.predict_proba(data)[:, 1]
                    prediction_results = {
                        'predictions': probabilities.tolist(),
                        'threshold': self.deployment_config.get('prediction_threshold', 0.5),
                        'decision': (probabilities >= self.deployment_config.get('prediction_threshold', 0.5)).tolist(),
                        'timestamp': datetime.now().isoformat()
                    }
                except (AttributeError, IndexError):
                    # If predict_proba fails, use binary predictions
                    prediction_results = {
                        'predictions': predictions.tolist(),
                        'threshold': self.deployment_config.get('prediction_threshold', 0.5),
                        'decision': predictions.tolist(),
                        'timestamp': datetime.now().isoformat()
                    }
            
            self.logger.info(f"Prediction served successfully")
            return prediction_results
            
        except Exception as e:
            self.logger.error(f"Error serving prediction: {str(e)}", exc_info=True)
            raise
    
    def has_production_model(self):
        """
        Check if there is a model currently deployed in production.
        
        Returns:
            bool: True if a production model exists, False otherwise
        """
        production_path = self.deployment_config.get('production_model_path', 'models/production')
        model_file = os.path.join(production_path, 'model.joblib')
        return os.path.exists(model_file)
    
    def deploy_if_no_model_exists(self, model, model_metadata):
        """
        Deploy a model to production only if no model currently exists.
        
        Args:
            model (object): Trained model to deploy
            model_metadata (dict): Metadata about the model
            
        Returns:
            dict or None: Deployment information if deployed, None if a model already exists
        """
        if not self.has_production_model():
            self.logger.info("No production model found. Proceeding with deployment.")
            return self.deploy(model, model_metadata)
        else:
            self.logger.info("Production model already exists. Skipping deployment.")
            return None