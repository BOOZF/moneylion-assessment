#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit application for MoneyLion Loan Risk Prediction System

This application provides a user interface for:
1. Predicting risk of new loan applications
2. Visualizing model performance and data insights
3. Running the automated ML pipeline components
"""

import json
import os
import sys
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
from sklearn.model_selection import train_test_split

# Add the project root to the path to import custom modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import pipeline components
from src.data import DataIngestion, DataProcessor
from src.modeling import ModelTrainer, ModelEvaluator
from src.deployment import ModelDeployer
from src.monitoring import ModelMonitor
from src.utils import setup_logging, load_config

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="MoneyLion Loan Risk Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
CONFIG_PATH = "config/pipeline_config.yaml"

@st.cache_data
def load_app_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

config = load_app_config()

# Define production model path early to avoid undefined variable errors
production_model_path = config.get('deployment', {}).get('production_model_path', 'models/production')

# Pipeline functions (as discussed in the previous response)
def ingest_data(config, run_id):
    """
    Ingest data using the DataIngestion class from src.data.
    
    Args:
        config (dict): Pipeline configuration
        run_id (str): Run identifier for tracking
    
    Returns:
        dict: Dictionary containing the ingested data DataFrames
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion")
    
    # Initialize data ingestion
    data_ingestion = DataIngestion(
        config['data_sources'],
        config['data_storage']
    )
    
    # Ingest loan application data
    loan_data = data_ingestion.ingest_loan_data()
    
    # Ingest clarity/fraud data
    clarity_data = data_ingestion.ingest_clarity_data()
    
    # Ingest payment history data
    payment_data = data_ingestion.ingest_payment_data()
    
    # Save the ingested data with the run ID for traceability
    output_dir = os.path.join(config['data_storage']['processed_data_dir'], run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving ingested data to {output_dir}")
    loan_data.to_parquet(os.path.join(output_dir, 'loan_data.parquet'))
    clarity_data.to_parquet(os.path.join(output_dir, 'clarity_data.parquet'))
    payment_data.to_parquet(os.path.join(output_dir, 'payment_data.parquet'))
    
    logger.info(f"Data ingestion completed and saved to {output_dir}")
    
    return {'loan_data': loan_data, 'clarity_data': clarity_data, 'payment_data': payment_data}

def process_data(config, run_id):
    """
    Process and prepare data for model training using DataProcessor.
    
    Args:
        config (dict): Pipeline configuration
        run_id (str): Run identifier for tracking
    
    Returns:
        dict: Dictionary containing processed feature data
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data processing")
    
    # Load the ingested data
    input_dir = os.path.join(config['data_storage']['processed_data_dir'], run_id)
    
    # Load data
    logger.info(f"Loading data from {input_dir}")
    loan_data = pd.read_parquet(os.path.join(input_dir, 'loan_data.parquet'))
    clarity_data = pd.read_parquet(os.path.join(input_dir, 'clarity_data.parquet'))
    payment_data = pd.read_parquet(os.path.join(input_dir, 'payment_data.parquet'))
    
    # Initialize data processor
    data_processor = DataProcessor(config['feature_engineering'])
    
    # Process data
    logger.info("Transforming raw data into model features")
    processed_data = data_processor.process(loan_data, clarity_data, payment_data)
    
    # Split into training and testing sets
    logger.info("Splitting data into training and test sets")
    X = processed_data.drop(columns=[config['target_column']])
    y = processed_data[config['target_column']]
    
    # Determine stratification approach for continuous target
    stratify_param = None
    if config['training'].get('stratify', False):
        try:
            # Create risk buckets for stratification
            y_buckets = pd.qcut(y, q=5, labels=False, duplicates='drop')
            
            # Ensure each bucket has at least 2 samples
            bucket_counts = y_buckets.value_counts()
            logger.info("Risk score bucket distribution:")
            logger.info(bucket_counts)
            
            # If buckets are valid, use for stratification
            if (bucket_counts >= 2).all():
                stratify_param = y_buckets
                logger.info("Using risk score buckets for stratification")
            else:
                logger.warning("Not enough samples in risk buckets. Falling back to non-stratified split.")
        except Exception as e:
            logger.warning(f"Error creating risk buckets: {str(e)}. Using non-stratified split.")
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['training']['test_size'],
        random_state=config['training']['random_seed'],
        stratify=stratify_param
    )
    
    # Save processed data
    output_dir = os.path.join(config['data_storage']['features_dir'], run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    X_train.to_parquet(os.path.join(output_dir, 'X_train.parquet'))
    X_test.to_parquet(os.path.join(output_dir, 'X_test.parquet'))
    pd.DataFrame(y_train, columns=[config['target_column']]).to_parquet(os.path.join(output_dir, 'y_train.parquet'))
    pd.DataFrame(y_test, columns=[config['target_column']]).to_parquet(os.path.join(output_dir, 'y_test.parquet'))
    
    # Save feature metadata
    feature_metadata = {
        'feature_names': X.columns.tolist(),
        'categorical_features': data_processor.categorical_features,
        'numeric_features': data_processor.numeric_features,
        'target_distribution': {
            'mean': float(y.mean()),
            'median': float(y.median()),
            'min': float(y.min()),
            'max': float(y.max()),
            'std': float(y.std())
        },
        'preprocessing_date': datetime.now().isoformat()
    }
    
    joblib.dump(feature_metadata, os.path.join(output_dir, 'feature_metadata.joblib'))
    
    logger.info(f"Data processing completed and saved to {output_dir}")
    
    return {
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train, 
        'y_test': y_test,
        'feature_metadata': feature_metadata
    }

def train_model(config, run_id):
    """
    Train the loan risk prediction model using processed data.
    
    Args:
        config (dict): Pipeline configuration
        run_id (str): Run identifier for tracking
    
    Returns:
        dict: Dictionary containing the trained model and metadata
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training")
    
    # Load processed data
    features_dir = os.path.join(config['data_storage']['features_dir'], run_id)
    
    # Load data
    X_train = pd.read_parquet(os.path.join(features_dir, 'X_train.parquet'))
    y_train_df = pd.read_parquet(os.path.join(features_dir, 'y_train.parquet'))
    y_train = y_train_df[config['target_column']]
    feature_metadata = joblib.load(os.path.join(features_dir, 'feature_metadata.joblib'))
    
    # Initialize model trainer
    model_config = config['model'].copy()
    model_config['task'] = 'regression'  # Explicitly set for risk scoring
    
    model_trainer = ModelTrainer(
        model_config,
        feature_metadata
    )
    
    # Train model
    model = model_trainer.train(X_train, y_train)
    
    # Save model
    models_dir = os.path.join(config['data_storage']['models_dir'], run_id)
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Prepare and save model metadata
    model_metadata = {
        'model_type': model_config['type'],
        'task': 'regression',
        'is_regression': True,
        'hyperparameters': model_config.get('hyperparameters', {}),
        'feature_names': feature_metadata['feature_names'],
        'training_date': datetime.now().isoformat(),
        'run_id': run_id
    }
    
    # Extract feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_names = model_trainer._get_feature_names(model, X_train)
        indices = np.argsort(model.feature_importances_)[::-1]
        
        model_metadata['feature_importance'] = {
            'features': [feature_names[i] for i in indices],
            'values': model.feature_importances_[indices].tolist()
        }
    
    joblib.dump(model_metadata, os.path.join(models_dir, 'model_metadata.joblib'))
    
    logger.info(f"Model training completed and saved to {model_path}")
    
    return {'model': model, 'model_metadata': model_metadata}

def evaluate_model(config, run_id):
    """
    Evaluate the trained model on test data.
    
    Args:
        config (dict): Pipeline configuration
        run_id (str): Run identifier for tracking
    
    Returns:
        dict: Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation")
    
    # Load model
    models_dir = os.path.join(config['data_storage']['models_dir'], run_id)
    model_path = os.path.join(models_dir, 'model.joblib')
    model = joblib.load(model_path)
    
    # Load test data
    features_dir = os.path.join(config['data_storage']['features_dir'], run_id)
    X_test = pd.read_parquet(os.path.join(features_dir, 'X_test.parquet'))
    y_test_df = pd.read_parquet(os.path.join(features_dir, 'y_test.parquet'))
    y_test = y_test_df[config['target_column']]
    
    # Prepare evaluation config for regression
    eval_config = config['evaluation'].copy()
    
    # Initialize model evaluator
    model_evaluator = ModelEvaluator(eval_config)
    
    # Evaluate model
    evaluation_results = model_evaluator.evaluate(model, X_test, y_test)
    
    # Create evaluation directory
    eval_dir = os.path.join(config['data_storage']['evaluation_dir'], run_id)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save evaluation results
    joblib.dump(evaluation_results, os.path.join(eval_dir, 'evaluation_results.joblib'))
    
    # Attempt to generate report
    report_path = os.path.join(eval_dir, 'evaluation_report.html')
    try:
        model_evaluator.generate_report(
            model=model,
            X_test=X_test,
            y_test=y_test,
            evaluation_results=evaluation_results,
            output_path=report_path
        )
    except Exception as e:
        logger.error(f"Error generating evaluation report: {str(e)}")
    
    logger.info(f"Model evaluation completed and saved to {eval_dir}")
    
    return {'evaluation_results': evaluation_results}

def deploy_model(config, run_id, force_deploy=False):
    """
    Deploy the model to production if it meets quality criteria.
    
    Args:
        config (dict): Pipeline configuration
        run_id (str): Run identifier for tracking
        force_deploy (bool): If True, deploy model regardless of criteria
    
    Returns:
        dict: Dictionary containing deployment results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model deployment process")
    
    # Load evaluation results
    eval_dir = os.path.join(config['data_storage']['evaluation_dir'], run_id)
    evaluation_results = joblib.load(os.path.join(eval_dir, 'evaluation_results.joblib'))
    
    # Load model and metadata
    models_dir = os.path.join(config['data_storage']['models_dir'], run_id)
    model_path = os.path.join(models_dir, 'model.joblib')
    model = joblib.load(model_path)
    model_metadata = joblib.load(os.path.join(models_dir, 'model_metadata.joblib'))
    
    # Initialize model deployer
    model_deployer = ModelDeployer(config['deployment'])
    
    # Deployment decision logic
    if force_deploy:
        logger.info("Force deploy flag is set. Deploying model regardless of criteria.")
        deployment_info = model_deployer.deploy(model, model_metadata)
    elif not model_deployer.has_production_model():
        # No production model exists, automatically deploy
        logger.info("No existing production model found. Automatically deploying current model.")
        deployment_info = model_deployer.deploy(model, model_metadata)
    else:
        # Check if model meets deployment criteria
        if model_deployer.should_deploy(evaluation_results):
            # Deploy model
            logger.info("Model meets quality criteria, proceeding with deployment")
            deployment_info = model_deployer.deploy(model, model_metadata)
        else:
            logger.info("Model did not meet deployment criteria. Skipping deployment.")
            return {'deployment_status': 'skipped', 'reason': 'Did not meet deployment criteria'}
    
    # Save deployment info
    deploy_dir = os.path.join(config['data_storage']['deployment_dir'], run_id)
    os.makedirs(deploy_dir, exist_ok=True)
    
    joblib.dump(deployment_info, os.path.join(deploy_dir, 'deployment_info.joblib'))
    
    logger.info(f"Model deployment completed successfully. Model {run_id} is now in production.")
    return {'deployment_status': 'success', 'deployment_info': deployment_info}

def monitor_model(config, run_id):
    """
    Set up monitoring for the deployed model.
    
    Args:
        config (dict): Pipeline configuration
        run_id (str): Run identifier for tracking
    
    Returns:
        dict: Dictionary containing monitoring configuration info
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model monitoring setup")
    
    # Initialize model monitor
    model_monitor = ModelMonitor(config['monitoring'])
    
    # Set up monitoring
    monitoring_info = model_monitor.setup_monitoring(run_id)
    
    # Save monitoring configuration
    monitor_dir = os.path.join(config['data_storage']['monitoring_dir'], run_id)
    os.makedirs(monitor_dir, exist_ok=True)
    
    joblib.dump(monitoring_info, os.path.join(monitor_dir, 'monitoring_info.joblib'))
    
    logger.info(f"Model monitoring setup completed")
    
    return {'monitoring_info': monitoring_info}

# Existing helper functions from the previous implementation
def load_model_and_metadata():
    """
    Load the production model and its metadata with robust error handling.
    
    Returns:
        tuple: (model, metadata) if found, otherwise (None, None)
    """
    try:
        model_path = os.path.join(production_model_path, 'model.joblib')
        
        # Check if model exists
        if not os.path.exists(model_path):
            logging.warning(f"Production model not found at {model_path}")
            return None, None
            
        # Try to load the model
        model = joblib.load(model_path)
        
        # Try to load metadata from different potential sources
        metadata = None
        metadata_paths = [
            os.path.join(production_model_path, 'model_metadata.joblib'),
            os.path.join(production_model_path, 'model_metadata.json')
        ]
        
        for path in metadata_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.json'):
                        with open(path, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = joblib.load(path)
                    logging.info(f"Model metadata loaded from {path}")
                    break
                except Exception as e:
                    logging.error(f"Error loading model metadata from {path}: {str(e)}")
        
        # If metadata still not found, create minimal metadata from model
        if metadata is None:
            logging.warning("Model metadata not found, creating basic metadata")
            metadata = {
                'model_type': 'lightgbm',
                'is_regression': True,
                'training_date': datetime.now().isoformat(),
                'model_version': 'unknown'
            }
            
            # Try to extract feature metadata from the model
            if hasattr(model, 'feature_importances_'):
                metadata['feature_importance'] = {
                    'values': model.feature_importances_.tolist() 
                }
        
        return model, metadata
        
    except Exception as e:
        logging.error(f"Error loading model and metadata: {str(e)}", exc_info=True)
        return None, None

# Caching decorator for production model
@st.cache_resource
def load_production_model():
    """
    Load the production model with caching.
    
    Returns:
        object: Loaded model or None if not found
    """
    model, _ = load_model_and_metadata()
    return model

# Preprocessing and prediction functions
def preprocess_input(loan_data, clarity_df=None, payment_df=None):
    """
    Preprocess input data to match the format expected by the model.
    
    Args:
        loan_data (pd.DataFrame): Loan application data
        clarity_df (pd.DataFrame): Optional clarity data
        payment_df (pd.DataFrame): Optional payment data
            
    Returns:
        pd.DataFrame: Data with all required features
    """
    try:
        # Initialize data processor
        data_processor = DataProcessor(config['feature_engineering'])
        
        # Add missing columns identified in the error message
        required_loan_columns = [
            'hasCF', 'anon_ssn', 'isFunded', 'originallyScheduledPaymentAmount'
        ]
        
        for col in required_loan_columns:
            if col not in loan_data.columns:
                # Set default values based on the column type
                if col == 'hasCF':
                    # Boolean indicating if clarity fraud data exists
                    loan_data[col] = True if clarity_df is not None and not clarity_df.empty else False
                elif col == 'anon_ssn':
                    # Anonymous SSN - set to a placeholder
                    loan_data[col] = 'anonymous-' + loan_data['loanId'].astype(str)
                elif col == 'isFunded':
                    # Whether loan is funded - set to same as 'approved'
                    loan_data[col] = loan_data['approved'] if 'approved' in loan_data.columns else True
                elif col == 'originallyScheduledPaymentAmount':
                    # Payment amount - estimate from loan amount (divide by 12 for monthly payment estimate)
                    loan_data[col] = loan_data['loanAmount'] / 12 if 'loanAmount' in loan_data.columns else 0
        
        # If clarity_df is None or empty, create a minimal one with required fields
        if clarity_df is None or clarity_df.empty:
            if 'clarityFraudId' in loan_data.columns:
                fraud_ids = loan_data['clarityFraudId'].unique()
                clarity_df = pd.DataFrame({
                    'underwritingid': fraud_ids,
                    'clearfraudscore': [500] * len(fraud_ids)  # Default score
                })
        
        # If payment_df is None, create an empty one with expected structure
        if payment_df is None:
            payment_df = pd.DataFrame(columns=[
                'loanId', 'installmentIndex', 'paymentAmount', 
                'paymentStatus', 'principal', 'fees'
            ])
            
            # Add minimal payment data for each loan
            if 'loanId' in loan_data.columns:
                loan_ids = loan_data['loanId'].unique()
                payment_records = []
                
                for loan_id in loan_ids:
                    # Create a dummy successful payment
                    payment_records.append({
                        'loanId': loan_id,
                        'installmentIndex': 1,
                        'paymentAmount': loan_data.loc[loan_data['loanId'] == loan_id, 'loanAmount'].values[0] / 12,  # Estimate monthly payment
                        'paymentStatus': 'Checked',  # Successful payment
                        'principal': loan_data.loc[loan_data['loanId'] == loan_id, 'loanAmount'].values[0] / 15,  # Estimated principal component
                        'fees': loan_data.loc[loan_data['loanId'] == loan_id, 'loanAmount'].values[0] / 60  # Estimated fees component
                    })
                
                if payment_records:
                    payment_df = pd.DataFrame(payment_records)
        
        # Process the data using the DataProcessor
        processed_df = data_processor.process(loan_data, clarity_df, payment_df)
        
        # Check for any missing required columns and add them with default values
        required_columns = [
            'clearfraudscore', 'clearfraudscore_squared', 'days_to_origination',
            'paymentAmount_mean', 'paymentAmount_sum', 'paymentAmount_count',
            'principal_mean', 'principal_sum', 'fees_mean', 'fees_sum', 
            'payment_success_rate', 'fpStatus'
        ]
        
        for col in required_columns:
            if col not in processed_df.columns:
                if col == 'clearfraudscore' and 'clearfraudscore' in loan_data.columns:
                    processed_df[col] = loan_data['clearfraudscore']
                elif col == 'clearfraudscore_squared' and 'clearfraudscore' in loan_data.columns:
                    processed_df[col] = loan_data['clearfraudscore'] ** 2
                elif col == 'days_to_origination':
                    processed_df[col] = 1  # Default value
                elif col in ['paymentAmount_mean', 'principal_mean', 'fees_mean']:
                    processed_df[col] = 0.0  # Default to zero
                elif col in ['paymentAmount_sum', 'principal_sum', 'fees_sum', 'paymentAmount_count']:
                    processed_df[col] = 0  # Default to zero
                elif col == 'payment_success_rate':
                    processed_df[col] = 1.0  # Default to perfect payment history
                elif col == 'fpStatus':
                    processed_df[col] = 'None'  # Default status
        
        return processed_df
    except Exception as e:
        logging.error(f"Error preprocessing input: {str(e)}", exc_info=True)
        return None

def handle_loan_prediction_submission(loan_form_data):
    """
    Process the loan form data and make a prediction.
    
    Args:
        loan_form_data (dict): Form input data from the user
        
    Returns:
        dict: Prediction results or error message
    """
    try:
        # Create a sample dataframe with input values
        # Converting to the expected format
        pay_freq_map = {"Weekly": "W", "Biweekly": "B", "Semimonthly": "S", "Monthly": "M"}
        loan_id = f"PRED_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        input_data = pd.DataFrame({
            'loanId': [loan_id],
            'loanAmount': [loan_form_data['loan_amount']],
            'apr': [loan_form_data['apr']],
            'payFrequency': [pay_freq_map.get(loan_form_data['pay_frequency'], "W")],
            'state': [loan_form_data['state']],
            'leadType': [loan_form_data['lead_type']],
            'nPaidOff': [loan_form_data['n_paid_off']],
            'leadCost': [loan_form_data['lead_cost']],
            # Add other required fields with default values
            'approved': [True],
            'originated': [True],
            'applicationDate': [datetime.now().strftime('%Y-%m-%d')],
            'clarityFraudId': [f"CF_{loan_form_data['clear_fraud_score']}"],
            'loanStatus': ["New Loan"],
            'fpStatus': ["None"]
        })
        
        # Create clarity data based on the provided fraud score
        clarity_df = pd.DataFrame({
            'underwritingid': [f"CF_{loan_form_data['clear_fraud_score']}"],
            'clearfraudscore': [loan_form_data['clear_fraud_score']]
        })
        
        # Create minimal payment data
        payment_df = pd.DataFrame({
            'loanId': [loan_id],
            'installmentIndex': [1],
            'paymentAmount': [loan_form_data['loan_amount'] / 12],  # Estimate monthly payment
            'paymentStatus': ['Checked'],  # Successful payment
            'principal': [loan_form_data['loan_amount'] / 15],  # Estimated principal component
            'fees': [loan_form_data['loan_amount'] / 60]  # Estimated fees component
        })
        
        # Use the existing preprocess_input function to handle missing columns
        processed_df = preprocess_input(input_data, clarity_df, payment_df)
        
        if processed_df is None:
            return {"error": "Failed to preprocess input data"}
        
        # Remove target column if present
        if 'target' in processed_df.columns:
            X_input = processed_df.drop('target', axis=1)
        else:
            X_input = processed_df
        
        # Define risk thresholds 
        low_risk_max = 0.3
        high_risk_min = 0.7
        
        # Load the production model
        model, _ = load_model_and_metadata()
        
        if model is None:
            return {"error": "Production model not found"}
        
        # Make prediction
        pred_value = float(model.predict(X_input)[0])
        
        # Determine risk level based on thresholds
        if pred_value <= low_risk_max:
            risk_level = "Low Risk"
        elif pred_value >= high_risk_min:
            risk_level = "High Risk"
        else:
            risk_level = "Medium Risk"
        
        return {
            "success": True,
            "risk_score": pred_value,
            "risk_category": risk_level
        }
    
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}", exc_info=True)
        return {"error": f"Error making prediction: {str(e)}"}

def normalize_risk_score_for_display(risk_score):
    """
    Normalize risk scores for visualization purposes.
    
    The model appears to output scores outside the 0-1 range,
    so this function maps them to a 0-1 scale for gauge display.
    
    Args:
        risk_score (float): The original risk score from the model
        
    Returns:
        float: Normalized score between 0 and 1
    """
    # Based on the sample data, we've seen scores from around -0.01 to 4.0
    # Adjust these min/max values based on your model's output range
    min_score = -1.0
    max_score = 5.0
    
    # Normalize to 0-1 range
    normalized = (risk_score - min_score) / (max_score - min_score)
    
    # Clamp between 0 and 1
    return max(0, min(normalized, 1))

# Continue with the rest of the Streamlit app code from the previous implementation
# (The navigation, sidebar, and page rendering code remains the same)

if __name__ == "__main__":
    pass
