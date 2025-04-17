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
                    'clearfraudscore': [300] * len(fraud_ids)  # Changed from 500 to 300 for more balanced risk
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
                    # Create a dummy payment with some risk indicators
                    payment_records.append({
                        'loanId': loan_id,
                        'installmentIndex': 1,
                        'paymentAmount': loan_data.loc[loan_data['loanId'] == loan_id, 'loanAmount'].values[0] / 12,
                        'paymentStatus': 'Late',  # Changed from 'Checked' to 'Late' to indicate some risk
                        'principal': loan_data.loc[loan_data['loanId'] == loan_id, 'loanAmount'].values[0] / 15,
                        'fees': loan_data.loc[loan_data['loanId'] == loan_id, 'loanAmount'].values[0] / 30  # Increased fees
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
        
        # Log the processed features and their values
        logging.info("Processed input features:")
        for col in processed_df.columns:
            logging.info(f"{col}: {processed_df[col].values[0]}")
        
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
        
        # Log the processed features and their values
        logging.info("Processed input features:")
        for col in processed_df.columns:
            logging.info(f"{col}: {processed_df[col].values[0]}")
        
        # Remove target column if present
        if 'target' in processed_df.columns:
            X_input = processed_df.drop('target', axis=1)
        else:
            X_input = processed_df
        
        # Define risk thresholds 
        low_risk_max = 0.4  # Increased from 0.3
        high_risk_min = 0.6  # Decreased from 0.7
        
        # Load the production model
        model, _ = load_model_and_metadata()
        
        if model is None:
            return {"error": "Production model not found"}
        
        # Make prediction
        pred_value = float(model.predict(X_input)[0])
        
        # Log the prediction value for debugging
        logging.info(f"Raw prediction value: {pred_value}")
        
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

# Define sidebar navigation
st.sidebar.title("MoneyLion ML System")
st.sidebar.image("image/Logo.png", width=200)

# Navigation options
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Data Exploration", "Predict Risk", "Model Performance", "Run ML Pipeline"]
)

# Button to reload configuration
if st.sidebar.button("Reload Configuration"):
    st.cache_data.clear()
    config = load_app_config()
    st.sidebar.success("Configuration reloaded!")

# Show current model version
st.sidebar.markdown("---")

# Get model metadata safely
_, model_metadata = load_model_and_metadata()

if model_metadata:
    model_version = model_metadata.get('model_version', 'Unknown')
    training_date = model_metadata.get('training_date', 'Unknown')
    model_type = model_metadata.get('model_type', 'Unknown')
    is_regression = model_metadata.get('is_regression', True)
    
    st.sidebar.markdown(f"**Production Model**")
    st.sidebar.markdown(f"Version: {model_version}")
    st.sidebar.markdown(f"Trained: {training_date}")
    st.sidebar.markdown(f"Type: {model_type} {'Regression' if is_regression else 'Classification'}")
else:
    st.sidebar.warning("No production model found")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 MoneyLion ML Team")

# Overview page
if page == "Overview":
    st.title("Loan Risk Prediction System")
    
    st.markdown("""
    This application provides tools to assess the risk of loan applications using a machine learning model.
    
    ## Key Features
    
    - **Predict Risk**: Evaluate the risk of a new loan application
    - **Data Exploration**: Visualize and analyze loan data
    - **Model Performance**: Review current model metrics and performance
    - **Run ML Pipeline**: Execute the automated machine learning pipeline
    
    ## Risk Score Interpretation
    
    The system produces a continuous risk score from 0 to 1:
    - **Low Risk** (0.0 - 0.3): Lower likelihood of default
    - **Medium Risk** (0.3 - 0.7): Moderate likelihood of default
    - **High Risk** (0.7 - 1.0): Higher likelihood of default
    
    ## Model Information
    
    The system uses a **LightGBM** model trained on historical loan data including:
    - Application details
    - Payment history
    - Fraud detection variables
    """)
    
    # Display model info if available
    model = load_production_model()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Architecture")
        st.image("image/architecture.png", 
                 caption="ML System Architecture Example")
    
    with col2:
        st.subheader("Current Configuration")
        st.json(config)

# Data Exploration page
elif page == "Data Exploration":
    st.title("Data Exploration")
    
    # Load and process sample data
    with st.spinner("Processing data for exploration..."):
        try:
            # Configure data sources
            data_source_config = {
                'loan_data': {'path': config.get('data_sources', {}).get('loan_data', {}).get('path', 'data/raw/loan.csv')},
                'clarity_data': {'path': config.get('data_sources', {}).get('clarity_data', {}).get('path', 'data/raw/clarity_underwriting_variables.csv')},
                'payment_data': {'path': config.get('data_sources', {}).get('payment_data', {}).get('path', 'data/raw/payment.csv')}
            }
            
            data_storage_config = {
                'processed_data_dir': config.get('data_storage', {}).get('processed_data_dir', 'data/processed'),
                'features_dir': config.get('data_storage', {}).get('features_dir', 'data/features')
            }
            
            # Initialize data ingestion
            data_ingestion = DataIngestion(data_source_config, data_storage_config)
            
            # Load data
            loan_df = data_ingestion.ingest_loan_data()
            clarity_df = data_ingestion.ingest_clarity_data()
            payment_df = data_ingestion.ingest_payment_data()
            
            # Configure feature engineering
            feature_config = {
                'create_interactions': True,
                'polynomial_features': True,
                'polynomial_features_list': ['loanAmount', 'apr', 'clearfraudscore'],
                'categorical_encoding': 'target',
                'drop_low_variance': True,
                'variance_threshold': 0.01
            }
            
            # Initialize data processor
            data_processor = DataProcessor(feature_config)
            
            # Process data
            processed_df = data_processor.process(loan_df, clarity_df, payment_df)
            
            # Create tabs for viewing different data
            tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Processed Data", "Feature Information", "Visualizations"])
            
            with tab1:
                st.subheader("Raw Loan Data Sample")
                st.dataframe(loan_df.head(10))
                st.text(f"Loan data shape: {loan_df.shape}")
                
                st.subheader("Clarity Data Sample")
                st.dataframe(clarity_df.head(10))
                st.text(f"Clarity data shape: {clarity_df.shape}")
                
                st.subheader("Payment Data Sample")
                st.dataframe(payment_df.head(10))
                st.text(f"Payment data shape: {payment_df.shape}")
            
            with tab2:
                st.subheader("Processed Data")
                
                # Option to view specific columns
                all_columns = processed_df.columns.tolist()
                
                # Group columns for easier selection
                column_groups = {
                    "All Columns": all_columns,
                    "Original Features": [col for col in all_columns if not (
                        col.endswith('_squared') or 
                        col.endswith('_mean') or 
                        col.endswith('_sum') or 
                        col.endswith('_count') or
                        col.endswith('_target_encoded')
                    )],
                    "Derived Features": [col for col in all_columns if (
                        col.endswith('_squared') or 
                        col.endswith('_mean') or 
                        col.endswith('_sum') or 
                        col.endswith('_count') or
                        col.endswith('_target_encoded')
                    )],
                    "Important Features": ['loanAmount', 'apr', 'clearfraudscore', 'clearfraudscore_squared',
                                            'paymentAmount_mean', 'payment_success_rate', 'target']
                }
                
                # Let user choose which column group to view
                selected_group = st.selectbox("Select column group to view:", list(column_groups.keys()))
                columns_to_show = column_groups[selected_group]
                
                # Allow filtering by column name
                if st.checkbox("Filter columns by name"):
                    filter_text = st.text_input("Filter columns containing:", "")
                    if filter_text:
                        columns_to_show = [col for col in all_columns if filter_text.lower() in col.lower()]
                
                # Show the filtered processed data
                if columns_to_show:
                    st.dataframe(processed_df[columns_to_show].head(20))
                    
                    # Option to see more rows
                    num_rows = st.slider("Number of rows to show:", 5, 100, 20)
                    if num_rows != 20:
                        st.dataframe(processed_df[columns_to_show].head(num_rows))
                else:
                    st.warning("No columns match your filter.")
                
                # Option to see full data
                if st.checkbox("Show all processed data (may be large)"):
                    st.dataframe(processed_df)
            
            with tab3:
                st.subheader("Feature Information")
                
                # Show feature types
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Numeric Features:**")
                    st.write(f"Count: {len(data_processor.numeric_features)}")
                    st.write(data_processor.numeric_features)
                
                with col2:
                    st.markdown("**Categorical Features:**")
                    st.write(f"Count: {len(data_processor.categorical_features)}")
                    st.write(data_processor.categorical_features)
                
                # Feature statistics
                st.subheader("Feature Statistics")
                st.dataframe(processed_df.describe())
                
                # Target statistics
                if 'target' in processed_df.columns:
                    st.subheader("Target Variable Statistics")
                    st.dataframe(processed_df['target'].describe())
            
            with tab4:
                st.subheader("Data Visualizations")
                
                # Target distribution
                if 'target' in processed_df.columns:
                    st.markdown("**Risk Score Distribution**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    processed_df['target'].hist(bins=30, ax=ax)
                    ax.set_title('Distribution of Risk Scores')
                    ax.set_xlabel('Risk Score')
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
                
                # Feature correlations with target
                st.markdown("**Feature Correlations with Target**")
                
                if 'target' in processed_df.columns:
                    numeric_features = processed_df.select_dtypes(include=['float64', 'int64']).columns
                    target_correlations = processed_df[numeric_features].corrwith(processed_df['target']).sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    target_correlations.head(15).plot(kind='bar', ax=ax)
                    ax.set_title('Top 15 Features by Correlation with Risk Score')
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Correlation')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("**Top 10 Features by Correlation with Risk Score:**")
                    st.dataframe(target_correlations.head(10))
                
                # Correlation matrix
                if st.checkbox("Show correlation matrix heatmap"):
                    st.markdown("**Correlation Matrix**")
                    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
                    
                    # Limit to top correlated features for readability
                    if len(numeric_cols) > 20:
                        # If target exists, use features most correlated with target
                        if 'target' in processed_df.columns:
                            selected_cols = list(target_correlations.head(15).index)
                            if 'target' not in selected_cols:
                                selected_cols.append('target')
                        else:
                            selected_cols = list(numeric_cols[:15])
                    else:
                        selected_cols = numeric_cols
                    
                    corr_matrix = processed_df[selected_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                                fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title('Feature Correlation Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)

# Predict Risk page
elif page == "Predict Risk":
    st.title("Predict Loan Risk")
    
    # Load production model
    model = load_production_model()
    
    if model is None:
        st.error("No production model available. Please train and deploy a model first.")
    else:
        st.info("Enter loan application details to predict risk score.")
        
        # Create form for input
        with st.form("loan_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                loan_amount = st.number_input("Loan Amount ($)", min_value=100, max_value=50000, value=1000)
                apr = st.number_input("APR (%)", min_value=1.0, max_value=1000.0, value=25.0)
                pay_frequency = st.selectbox("Payment Frequency", ["Weekly", "Biweekly", "Semimonthly", "Monthly"])
                
            with col2:
                state = st.selectbox("State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                                              "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                                              "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                                              "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                                              "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
                lead_type = st.selectbox("Lead Type", ["organic", "lead", "bvMandatory", "express", "california", 
                                                     "rc_returning", "prescreen", "repeat", "instant-offer"])
                n_paid_off = st.number_input("Previous Paid Off Loans", min_value=0, max_value=10, value=0)
                
            with col3:
                clear_fraud_score = st.slider("Clear Fraud Score", min_value=0, max_value=1000, value=500)
                lead_cost = st.number_input("Lead Cost ($)", min_value=0.0, max_value=100.0, value=5.0)
                
            submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            # Collect form data
            loan_form_data = {
                'loan_amount': loan_amount,
                'apr': apr,
                'pay_frequency': pay_frequency,
                'state': state,
                'lead_type': lead_type,
                'n_paid_off': n_paid_off,
                'clear_fraud_score': clear_fraud_score,
                'lead_cost': lead_cost
            }
            
            # Call the prediction handler
            result = handle_loan_prediction_submission(loan_form_data)
            
            if "error" in result:
                st.error(result["error"])
            else:
                # Display the prediction result
                st.markdown("## Risk Assessment Result")
                
                # Determine color based on risk category
                if result["risk_category"] == "Low Risk":
                    color = "green"
                elif result["risk_category"] == "High Risk":
                    color = "red"
                else:
                    color = "orange"
                
                # Create columns for better layout
                result_col1, result_col2 = st.columns([1, 3])
                
                with result_col1:
                    st.markdown(f"### Risk Score")
                    st.markdown(f"<h1 style='color:{color};'>{result['risk_score']:.4f}</h1>", unsafe_allow_html=True)
                    st.markdown(f"**Category:** {result['risk_category']}")
                
                with result_col2:
                    # Create a gauge chart - normalize the score for display purposes
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Get thresholds from config
                    risk_thresholds = config.get('deployment', {}).get('risk_categorization', {})
                    low_risk_max = risk_thresholds.get('low_risk_max', 0.3)
                    high_risk_min = risk_thresholds.get('high_risk_min', 0.7)
                    
                    # Create a horizontal bar
                    ax.barh(y=0, width=1, color='lightgrey', height=0.5)
                    
                    # Create segment colors
                    ax.barh(y=0, width=low_risk_max, color='green', height=0.5)
                    ax.barh(y=0, width=high_risk_min-low_risk_max, left=low_risk_max, color='orange', height=0.5)
                    ax.barh(y=0, width=1-high_risk_min, left=high_risk_min, color='red', height=0.5)
                    
                    # Normalize the risk score for display on the gauge (0-1 range)
                    normalized_score = normalize_risk_score_for_display(result['risk_score'])
                    
                    # Show the marker for current score
                    ax.scatter(x=normalized_score, y=0, color='black', s=150, zorder=10)
                    
                    # Add labels
                    ax.text(low_risk_max/2, -0.3, "Low Risk", ha='center')
                    ax.text((low_risk_max+high_risk_min)/2, -0.3, "Medium Risk", ha='center')
                    ax.text((high_risk_min+1)/2, -0.3, "High Risk", ha='center')
                    
                    # Add annotation with actual score
                    ax.annotate(f"Score: {result['risk_score']:.2f}", 
                               xy=(normalized_score, 0),
                               xytext=(normalized_score, 0.3),
                               arrowprops=dict(arrowstyle="->", color='black'),
                               ha='center')
                    
                    # Set limits and remove axes
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    ax.set_title("Risk Score Scale", pad=20)
                    
                    st.pyplot(fig)
                
                # Show interpretation
                st.markdown("### Interpretation")
                
                if result["risk_category"] == "Low Risk":
                    st.markdown("""
                    - This application shows a **lower likelihood of default**
                    - Recommended for approval with standard terms
                    - Regular monitoring is sufficient
                    """)
                elif result["risk_category"] == "Medium Risk":
                    st.markdown("""
                    - This application shows a **moderate likelihood of default**
                    - Consider approval with adjusted terms (higher APR or lower amount)
                    - Enhanced monitoring recommended
                    """)
                else:
                    st.markdown("""
                    - This application shows a **higher likelihood of default**
                    - Recommend declining or requiring substantial additional verification
                    - If approved, implement strict monitoring and collection procedures
                    """)
                
                # Add explanation about risk score range
                st.info("""
                **Note about risk scores:** 
                This model produces risk scores that may fall outside the standard 0-1 range. 
                Scores below 0.3 are considered Low Risk, while scores above 0.7 are considered High Risk. 
                The visualization above normalizes these scores for display purposes.
                """)

# Model Performance page
elif page == "Model Performance":
    st.title("Model Performance")
    
    # Check if model exists
    model = load_production_model()
    
    if model is None:
        st.error("No production model available. Please train and deploy a model first.")
    else:
        try:
            # Try to load evaluation results
            eval_dir = os.path.join(config.get('data_storage', {}).get('evaluation_dir', 'models/evaluation'))
            
            # Find the latest evaluation if exists
            if os.path.exists(eval_dir):
                runs = [d for d in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, d))]
                
                if runs:
                    latest_run = sorted(runs)[-1]
                    eval_path = os.path.join(eval_dir, latest_run, 'evaluation_results.joblib')
                    
                    if os.path.exists(eval_path):
                        evaluation_results = joblib.load(eval_path)
                        
                        # Create tabs for different evaluation views
                        tab1, tab2, tab3 = st.tabs(["Main Metrics", "Visualizations", "Feature Importance"])
                        
                        with tab1:
                            st.subheader("Key Performance Metrics")
                            
                            # Create metrics display
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    label="R² Score",
                                    value=f"{evaluation_results.get('r2', 0):.4f}"
                                )
                                st.metric(
                                    label="Mean Absolute Error (MAE)",
                                    value=f"{evaluation_results.get('mae', 0):.4f}"
                                )
                            
                            with col2:
                                st.metric(
                                    label="Accuracy (Discretized)",
                                    value=f"{evaluation_results.get('accuracy', 0):.4f}"
                                )
                                st.metric(
                                    label="Mean Squared Error (MSE)",
                                    value=f"{evaluation_results.get('mse', 0):.4f}"
                                )
                            
                            with col3:
                                st.metric(
                                    label="Macro F1 Score",
                                    value=f"{evaluation_results.get('f1_macro', 0):.4f}"
                                )
                                st.metric(
                                    label="Macro Precision",
                                    value=f"{evaluation_results.get('precision_macro', 0):.4f}"
                                )
                            
                            # Display risk category distribution
                            st.subheader("Risk Category Distribution")
                            
                            if 'risk_categories' in evaluation_results:
                                risk_data = evaluation_results['risk_categories']
                                
                                # Create pie chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                labels = ['Low Risk', 'Medium Risk', 'High Risk']
                                sizes = [
                                    risk_data.get('low_risk', 0),
                                    risk_data.get('medium_risk', 0),
                                    risk_data.get('high_risk', 0)
                                ]
                                colors = ['green', 'orange', 'red']
                                
                                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')
                                ax.set_title("Distribution of Risk Categories in Test Set")
                                
                                st.pyplot(fig)
                            
                        with tab2:
                            st.subheader("Performance Visualizations")
                            
                            # Check if visualization files exist
                            vis_dir = os.path.join(eval_dir, latest_run)
                            pred_vs_actual_path = os.path.join(vis_dir, 'predicted_vs_actual.png')
                            dist_path = os.path.join(vis_dir, 'risk_score_distribution.png')
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if os.path.exists(pred_vs_actual_path):
                                    st.image(pred_vs_actual_path, caption="Predicted vs Actual Risk Scores")
                                else:
                                    st.warning("Predicted vs Actual plot not available")
                            
                            with col2:
                                if os.path.exists(dist_path):
                                    st.image(dist_path, caption="Risk Score Distribution")
                                else:
                                    st.warning("Risk score distribution plot not available")
                        
                        with tab3:
                            st.subheader("Feature Importance")
                            
                            # Load model metadata to get feature importance
                            _, model_metadata = load_model_and_metadata()
                            
                            if model_metadata and 'feature_importance' in model_metadata and model_metadata['feature_importance'] is not None:
                                feature_imp = model_metadata['feature_importance']
                                
                                # Plot feature importance
                                if 'features' in feature_imp and 'values' in feature_imp:
                                    features = feature_imp['features']
                                    values = feature_imp['values']
                                    
                                    # Get top 15 features
                                    if len(features) > 15:
                                        features = features[:15]
                                        values = values[:15]
                                    
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    y_pos = np.arange(len(features))
                                    
                                    ax.barh(y_pos, values, align='center')
                                    ax.set_yticks(y_pos)
                                    ax.set_yticklabels(features)
                                    ax.invert_yaxis()  # labels read top-to-bottom
                                    ax.set_title('Feature Importance')
                                    ax.set_xlabel('Importance')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.warning("Feature importance data not available in expected format")
                            else:
                                st.warning("Feature importance not available in model metadata")
                    else:
                        st.warning("No evaluation results found for the latest run")
                else:
                    st.warning("No evaluation runs found")
            else:
                st.warning("Evaluation directory does not exist")
                
        except Exception as e:
            st.error(f"Error loading model performance data: {str(e)}")
            st.exception(e)

# Run ML Pipeline page
elif page == "Run ML Pipeline":
    st.title("Run Automated ML Pipeline")
    
    st.markdown("""
    This page allows you to run components of the automated machine learning pipeline.
    
    **Note:** Running the full pipeline may take several minutes depending on the dataset size.
    """)
    
    # Create pipeline execution form
    with st.form("pipeline_form"):
        # Mode selection
        pipeline_mode = st.selectbox(
            "Pipeline Mode",
            options=[
                "full", "ingest", "process", "train", 
                "evaluate", "deploy", "monitor"
            ],
            help="Select which components of the pipeline to run"
        )
        
        # Run ID
        run_id = st.text_input(
            "Run ID (optional)",
            value=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Unique identifier for this pipeline run"
        )
        
        # Force deploy option
        force_deploy = st.checkbox(
            "Force Deploy Model",
            value=False,
            help="Deploy model regardless of performance metrics"
        )
        
        # Submit button
        submitted = st.form_submit_button("Run Pipeline")
    
    if submitted:
        # Build command
        cmd = [sys.executable, "main.py", "--mode", pipeline_mode]
        
        if run_id:
            cmd.extend(["--run-id", run_id])
        
        if force_deploy:
            cmd.append("--force-deploy")
        
        # Create progress bar
        progress = st.progress(0)
        
        # Create placeholder for log output
        log_output = st.empty()
        log_output.text("Starting pipeline execution...\n\n")
        
        # Run the command
        try:
            # Run pipeline as a subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Collect output
            log_text = ""
            progress_value = 0
            progress_step = 0.1
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    log_text += output
                    log_output.text(log_text)
                    
                    # Update progress
                    if progress_value < 0.9:
                        progress_value += progress_step
                        progress.progress(progress_value)
            
            # Get return code
            return_code = process.poll()
            
            # Update final progress
            progress.progress(1.0)
            
            if return_code == 0:
                st.success(f"Pipeline completed successfully in {pipeline_mode} mode!")
                
                # Show additional details based on mode
                if pipeline_mode in ["evaluate", "full"]:
                    st.info("Evaluation completed. Check the 'Model Performance' page for results.")
                
                if pipeline_mode in ["deploy", "full"]:
                    st.info("Model deployed successfully. Reload the application to see the latest model.")
            else:
                st.error(f"Pipeline execution failed with return code {return_code}")
        
        except Exception as e:
            st.error(f"Error executing pipeline: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    pass
