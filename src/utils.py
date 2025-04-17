import os
import logging
from datetime import datetime
import yaml
import json

def setup_logging(log_dir='logs'):
    """
    Set up logging configuration for the ML system.
    
    Args:
        log_dir (str): Directory to store log files
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log filename with timestamp
    log_filename = os.path.join(log_dir, f'ml_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Create logger instance
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized, saving logs to {log_filename}")

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration as dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {str(e)}")
        raise

def save_json(data, filepath):
    """
    Save data as JSON file.
    
    Args:
        data (dict): Data to save
        filepath (str): Path to save the file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save data
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving JSON to {filepath}: {str(e)}")
        raise

def load_json(filepath):
    """
    Load data from JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        dict: Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {filepath}: {str(e)}")
        raise

def create_directory_structure():
    """
    Create the directory structure required by the ML system.
    """
    # Define required directories
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'models/training',
        'models/production',
        'models/production/backups',
        'logs',
        'data/monitoring',
        'data/monitoring/reference_data',
        'data/monitoring/reports',
        'config'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logging.info("Directory structure created successfully")

def generate_default_config():
    """
    Generate a default configuration file.
    
    Returns:
        str: Path to the generated config file
    """
    # Default configuration
    default_config = {
        'data_sources': {
            'loan_data': {
                'path': 'data/raw/loan.csv',
                'type': 'csv'
            },
            'clarity_data': {
                'path': 'data/raw/clarity_underwriting_variables.csv',
                'type': 'csv'
            },
            'payment_data': {
                'path': 'data/raw/payment.csv',
                'type': 'csv'
            }
        },
        'data_storage': {
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'features_dir': 'data/features',
            'models_dir': 'models/training',
            'evaluation_dir': 'models/evaluation',
            'deployment_dir': 'models/production',
            'monitoring_dir': 'data/monitoring'
        },
        'feature_engineering': {
            'create_interactions': True,
            'polynomial_features': False,
            'polynomial_features_list': ['loanAmount', 'apr'],
            'categorical_encoding': 'onehot',
            'drop_low_variance': True,
            'variance_threshold': 0.01
        },
        'target_column': 'target',
        'training': {
            'test_size': 0.2,
            'random_seed': 42,
            'cv_folds': 5,
            'stratify': True
        },
        'model': {
            'type': 'lightgbm',
            'hyperparameters': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': 42
            }
        },
        'evaluation': {
            'primary_metric': 'roc_auc',
            'threshold': 0.5,
            'generate_shap': True
        },
        'deployment': {
            'production_model_path': 'models/production',
            'registry_path': 'models/registry.json',
            'thresholds': {
                'min_accuracy': 0.7,
                'min_roc_auc': 0.75,
                'min_precision': 0.6,
                'improvement_threshold': 0.01
            },
            'prediction_threshold': 0.5
        },
        'monitoring': {
            'frequency': 'daily',
            'checks': ['data_drift', 'performance_degradation', 'concept_drift'],
            'reference_data_path': 'data/monitoring/reference_data',
            'save_reference_data': True,
            'thresholds': {
                'drift_threshold': 0.1,
                'categorical_drift_threshold': 0.1,
                'ks_threshold': 0.1,
                'performance': {
                    'min_accuracy': 0.7,
                    'min_f1': 0.6,
                    'relative_degradation_threshold': -0.1
                }
            }
        },
        'logging': {
            'log_dir': 'logs',
            'log_level': 'INFO'
        }
    }
    
    # Create config directory if it doesn't exist
    os.makedirs('config', exist_ok=True)
    
    # Save default config
    config_path = 'config/pipeline_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    logging.info(f"Default configuration generated at {config_path}")
    return config_path