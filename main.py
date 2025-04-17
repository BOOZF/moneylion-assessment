import os
import sys
import logging
from datetime import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.data import DataIngestion, DataProcessor
from src.modeling import ModelTrainer, ModelEvaluator
from src.deployment import ModelDeployer
from src.monitoring import ModelMonitor
from src.utils import setup_logging

def main():
    """
    Main entry point for the automated machine learning pipeline.
    
    This function parses command line arguments, loads configuration,
    and runs the specified pipeline components.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Loan Risk Prediction Automated ML System')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--mode', type=str, 
                        choices=['full', 'ingest', 'process', 'train', 'evaluate', 'deploy', 'monitor'],
                        default='full', 
                        help='Pipeline mode to run')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID to use for model evaluation or deployment')
    parser.add_argument('--force-deploy', action='store_true', 
                        help='Force model deployment regardless of quality criteria')
    
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up logging
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting automated ML pipeline in {args.mode} mode")
    logger.info(f"Using configuration from {args.config}")
    
    try:
        # Create run ID for this pipeline execution if not provided
        run_id = args.run_id if args.run_id else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Run ID: {run_id}")
        
        # Execute pipeline based on mode
        if args.mode in ['full', 'ingest']:
            ingest_data(config, run_id)
        
        if args.mode in ['full', 'process']:
            process_data(config, run_id)
        
        if args.mode in ['full', 'train']:
            train_model(config, run_id)
        
        if args.mode in ['full', 'evaluate']:
            evaluate_model(config, run_id)
        
        if args.mode in ['full', 'deploy']:
            deploy_model(config, run_id, force_deploy=args.force_deploy)
        
        if args.mode in ['full', 'monitor']:
            monitor_model(config, run_id)
        
        logger.info(f"Pipeline completed successfully in {args.mode} mode")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


def ingest_data(config, run_id):
    """
    Ingest data from various sources into the ML pipeline.
    
    This function loads loan application data, payment history, and fraud detection data
    from source files and performs initial validation.
    
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
    logger.info("Ingesting loan application data")
    loan_data = data_ingestion.ingest_loan_data()
    
    # Ingest clarity/fraud data
    logger.info("Ingesting fraud detection data (Clarity)")
    clarity_data = data_ingestion.ingest_clarity_data()
    
    # Ingest payment history data
    logger.info("Ingesting payment history data")
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
    Process and prepare data for model training.
    
    This function transforms raw data into features suitable for model training,
    including feature engineering, cleaning, and train/test splitting.
    
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
    
    # If ingested data doesn't exist for this run, use the latest available
    if not os.path.exists(input_dir):
        runs = [d for d in os.listdir(config['data_storage']['processed_data_dir']) 
                if os.path.isdir(os.path.join(config['data_storage']['processed_data_dir'], d))]
        if not runs:
            raise ValueError("No ingested data found. Please run data ingestion first.")
        latest_run = sorted(runs)[-1]
        input_dir = os.path.join(config['data_storage']['processed_data_dir'], latest_run)
        logger.info(f"No ingested data found for run {run_id}. Using data from {latest_run}")
    
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
        # Create risk buckets for stratification with continuous target
        try:
            # Create risk buckets (usually 5 is a good number)
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
    
    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Test set size: {X_test.shape[0]} samples")
    
    # Log target distribution
    logger.info("Training set target distribution:")
    logger.info(y_train.describe())
    logger.info("Test set target distribution:")
    logger.info(y_test.describe())
    
    # Save processed data
    output_dir = os.path.join(config['data_storage']['features_dir'], run_id)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}")
    X_train.to_parquet(os.path.join(output_dir, 'X_train.parquet'))
    X_test.to_parquet(os.path.join(output_dir, 'X_test.parquet'))
    
    # Convert Series to DataFrame for saving to parquet
    pd.DataFrame(y_train, columns=[config['target_column']]).to_parquet(os.path.join(output_dir, 'y_train.parquet'))
    pd.DataFrame(y_test, columns=[config['target_column']]).to_parquet(os.path.join(output_dir, 'y_test.parquet'))
    
    # Save feature names and metadata
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
    
    This function trains a LightGBM model to predict loan risk scores 
    using the processed data from the previous step.
    
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
    
    # If processed data doesn't exist for this run, use the latest available
    if not os.path.exists(features_dir):
        runs = [d for d in os.listdir(config['data_storage']['features_dir']) 
                if os.path.isdir(os.path.join(config['data_storage']['features_dir'], d))]
        if not runs:
            raise ValueError("No processed data found. Please run data processing first.")
        latest_run = sorted(runs)[-1]
        features_dir = os.path.join(config['data_storage']['features_dir'], latest_run)
        logger.info(f"No processed data found for run {run_id}. Using data from {latest_run}")
    
    # Load data
    logger.info(f"Loading processed data from {features_dir}")
    X_train = pd.read_parquet(os.path.join(features_dir, 'X_train.parquet'))
    y_train_df = pd.read_parquet(os.path.join(features_dir, 'y_train.parquet'))
    y_train = y_train_df[config['target_column']]
    feature_metadata = joblib.load(os.path.join(features_dir, 'feature_metadata.joblib'))
    
    # Initialize model trainer
    logger.info("Initializing model trainer with LightGBM")
    
    # Update model config for regression if not already set
    model_config = config['model'].copy()
    if 'task' in model_config and model_config['task'] == 'regression':
        logger.info("Training regression model for continuous risk scoring")
    else:
        logger.info("Setting model task to regression for continuous risk scoring")
        model_config['task'] = 'regression'
        # Ensure hyperparameters are appropriate for regression
        if 'hyperparameters' in model_config:
            model_config['hyperparameters']['objective'] = 'regression'
    
    model_trainer = ModelTrainer(
        model_config,
        feature_metadata
    )
    
    # Train model
    logger.info(f"Training model with {X_train.shape[0]} samples and {X_train.shape[1]} features")
    model = model_trainer.train(X_train, y_train)
    
    # Save model
    models_dir = os.path.join(config['data_storage']['models_dir'], run_id)
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Extract and save feature importance if available
    feature_importance = None
    if hasattr(model, 'named_steps'):
        # Check if it's a regression or classification model
        if 'regressor' in model.named_steps:
            estimator = model.named_steps['regressor']
        elif 'classifier' in model.named_steps:
            estimator = model.named_steps['classifier']
        else:
            estimator = None
            
        if estimator is not None and hasattr(estimator, 'feature_importances_'):
            # Get feature names and importances
            try:
                # Get feature names from the pipeline if available
                feature_names = model_trainer._get_feature_names(model, X_train)
                
                # Sort by importance
                importances = estimator.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Create sorted lists
                sorted_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in indices]
                sorted_importances = [importances[i] for i in indices]
                
                feature_importance = {
                    'features': sorted_features,
                    'values': sorted_importances
                }
            except Exception as e:
                logger.warning(f"Could not extract feature names: {str(e)}")
                feature_importance = {
                    'feature_indices': list(range(len(estimator.feature_importances_))),
                    'values': estimator.feature_importances_.tolist()
                }
    
    # Save model metadata
    model_metadata = {
        'model_type': model_config['type'],
        'task': model_config.get('task', 'regression'),  # Default to regression
        'is_regression': True,  # Explicitly flag as regression model
        'hyperparameters': model_config.get('hyperparameters', {}),
        'feature_names': feature_metadata['feature_names'],
        'categorical_features': feature_metadata['categorical_features'],
        'numeric_features': feature_metadata['numeric_features'],
        'feature_importance': feature_importance,
        'training_date': datetime.now().isoformat(),
        'run_id': run_id,
        'target_distribution': feature_metadata.get('target_distribution', {})
    }
    
    joblib.dump(model_metadata, os.path.join(models_dir, 'model_metadata.joblib'))
    
    logger.info(f"Model training completed and saved to {model_path}")
    
    return {'model': model, 'model_metadata': model_metadata}


def evaluate_model(config, run_id):
    """
    Evaluate the trained model on test data.
    
    This function assesses the model's performance using regression metrics
    and generates evaluation reports.
    
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
    
    # If model doesn't exist for this run, use the latest available
    if not os.path.exists(models_dir):
        runs = [d for d in os.listdir(config['data_storage']['models_dir']) 
                if os.path.isdir(os.path.join(config['data_storage']['models_dir'], d))]
        if not runs:
            raise ValueError("No model found. Please run model training first.")
        latest_run = sorted(runs)[-1]
        models_dir = os.path.join(config['data_storage']['models_dir'], latest_run)
        logger.info(f"No model found for run {run_id}. Using model from {latest_run}")
    
    logger.info(f"Loading model from {models_dir}")
    model_path = os.path.join(models_dir, 'model.joblib')
    model = joblib.load(model_path)
    model_metadata = joblib.load(os.path.join(models_dir, 'model_metadata.joblib'))
    
    # Load test data
    features_dir = os.path.join(config['data_storage']['features_dir'], run_id)
    if not os.path.exists(features_dir):
        runs = [d for d in os.listdir(config['data_storage']['features_dir']) 
                if os.path.isdir(os.path.join(config['data_storage']['features_dir'], d))]
        if not runs:
            raise ValueError("No processed data found. Please run data processing first.")
        latest_run = sorted(runs)[-1]
        features_dir = os.path.join(config['data_storage']['features_dir'], latest_run)
        logger.info(f"No processed data found for run {run_id}. Using data from {latest_run}")
    
    logger.info(f"Loading test data from {features_dir}")
    X_test = pd.read_parquet(os.path.join(features_dir, 'X_test.parquet'))
    y_test_df = pd.read_parquet(os.path.join(features_dir, 'y_test.parquet'))
    y_test = y_test_df[config['target_column']]
    
    # Prepare evaluation config for regression
    eval_config = config['evaluation'].copy()
    
    # Add risk thresholds if available
    if 'risk_categorization' in config.get('deployment', {}):
        risk_thresholds = config['deployment']['risk_categorization']
        eval_config['risk_thresholds'] = {
            'low_risk_max': risk_thresholds.get('low_risk_max', 0.3),
            'high_risk_min': risk_thresholds.get('high_risk_min', 0.7)
        }
    
    # Initialize model evaluator
    model_evaluator = ModelEvaluator(eval_config)
    
    # Evaluate model
    logger.info(f"Evaluating regression model on {len(y_test)} test samples")
    evaluation_results = model_evaluator.evaluate(model, X_test, y_test)
    
    # Load current production model for comparison (if exists)
    production_model_path = os.path.join(config['deployment']['production_model_path'], 'model.joblib')
    if os.path.exists(production_model_path):
        logger.info("Comparing with current production model")
        production_model = joblib.load(production_model_path)
        
        # Compare new model with production model
        comparison_results = model_evaluator.compare_models(
            new_model=model,
            production_model=production_model,
            X_test=X_test,
            y_test=y_test
        )
        
        evaluation_results['comparison'] = comparison_results
        
        # Log comparison results
        if comparison_results.get('is_better', False):
            logger.info("New model outperforms production model!")
        else:
            logger.info("New model does not significantly outperform production model")
    
    # Create evaluation directory
    eval_dir = os.path.join(config['data_storage']['evaluation_dir'], run_id)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Save evaluation results
    joblib.dump(evaluation_results, os.path.join(eval_dir, 'evaluation_results.joblib'))
    
    # Create evaluation report
    logger.info("Generating evaluation report")
    report_path = os.path.join(eval_dir, 'evaluation_report.html')
    model_evaluator.generate_report(
        model=model,
        X_test=X_test,
        y_test=y_test,
        evaluation_results=evaluation_results,
        output_path=report_path
    )
    
    # Also save a summary as text
    with open(os.path.join(eval_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(f"Model Evaluation Summary for {run_id}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add regression metrics first
        f.write("Regression Metrics:\n")
        f.write(f"MSE: {evaluation_results.get('mse', 0):.6f}\n")
        f.write(f"MAE: {evaluation_results.get('mae', 0):.6f}\n")
        f.write(f"R²: {evaluation_results.get('r2', 0):.4f}\n\n")
        
        # Add classification metrics from discretized predictions
        f.write("Discretized Classification Metrics:\n")
        f.write(f"Accuracy: {evaluation_results.get('accuracy', 0):.4f}\n")
        f.write(f"Macro Precision: {evaluation_results.get('precision_macro', 0):.4f}\n")
        f.write(f"Macro Recall: {evaluation_results.get('recall_macro', 0):.4f}\n")
        f.write(f"Macro F1 Score: {evaluation_results.get('f1_macro', 0):.4f}\n\n")
        
        # Add risk category distribution
        if 'risk_categories' in evaluation_results:
            f.write("Risk Category Distribution:\n")
            f.write(f"Low Risk: {evaluation_results['risk_categories'].get('low_risk', 0):.2f}%\n")
            f.write(f"Medium Risk: {evaluation_results['risk_categories'].get('medium_risk', 0):.2f}%\n")
            f.write(f"High Risk: {evaluation_results['risk_categories'].get('high_risk', 0):.2f}%\n\n")
        
        if 'comparison' in evaluation_results:
            f.write("\nComparison with Production Model:\n")
            metric_diffs = evaluation_results['comparison'].get('metric_diffs', {})
            f.write(f"MSE Change: {metric_diffs.get('mse_diff', 0):+.6f}\n")
            f.write(f"R² Change: {metric_diffs.get('r2_diff', 0):+.4f}\n")
            f.write(f"Accuracy Change: {metric_diffs.get('accuracy_diff', 0)*100:+.2f}%\n")
            f.write(f"Is Better: {evaluation_results['comparison'].get('is_better', False)}\n")
    
    logger.info(f"Model evaluation completed and saved to {eval_dir}")
    
    return {'evaluation_results': evaluation_results}


def deploy_model(config, run_id, force_deploy=False):
    """
    Deploy the model to production if it meets quality criteria.
    
    This function checks if the model meets deployment criteria and
    if so, moves it to the production environment. If no production 
    model exists, it will automatically store the current model.
    
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
    
    # If evaluation doesn't exist for this run, use the latest available
    if not os.path.exists(eval_dir):
        runs = [d for d in os.listdir(config['data_storage']['evaluation_dir']) 
                if os.path.isdir(os.path.join(config['data_storage']['evaluation_dir'], d))]
        if not runs:
            raise ValueError("No evaluation results found. Please run model evaluation first.")
        latest_run = sorted(runs)[-1]
        eval_dir = os.path.join(config['data_storage']['evaluation_dir'], latest_run)
        logger.info(f"No evaluation results found for run {run_id}. Using results from {latest_run}")
        run_id = latest_run
    
    logger.info(f"Loading evaluation results from {eval_dir}")
    evaluation_results = joblib.load(os.path.join(eval_dir, 'evaluation_results.joblib'))
    
    # Load model
    models_dir = os.path.join(config['data_storage']['models_dir'], run_id)
    if not os.path.exists(models_dir):
        runs = [d for d in os.listdir(config['data_storage']['models_dir']) 
                if os.path.isdir(os.path.join(config['data_storage']['models_dir'], d))]
        latest_run = sorted(runs)[-1]
        models_dir = os.path.join(config['data_storage']['models_dir'], latest_run)
        logger.info(f"No model found for run {run_id}. Using model from {latest_run}")
        run_id = latest_run
    
    logger.info(f"Loading model from {models_dir}")
    model_path = os.path.join(models_dir, 'model.joblib')
    model = joblib.load(model_path)
    model_metadata = joblib.load(os.path.join(models_dir, 'model_metadata.joblib'))
    
    # Ensure model metadata has regression flag
    if 'is_regression' not in model_metadata:
        logger.info("Adding regression flag to model metadata")
        model_metadata['is_regression'] = True
    
    # Initialize model deployer
    model_deployer = ModelDeployer(config['deployment'])
    
    # Check if a production model exists
    production_model_exists = model_deployer.has_production_model()
    
    # Deployment decision logic
    if force_deploy:
        logger.info("Force deploy flag is set. Deploying model regardless of criteria.")
        deployment_info = model_deployer.deploy(model, model_metadata)
    elif not production_model_exists:
        # No production model exists, automatically deploy
        logger.info("No existing production model found. Automatically deploying current model.")
        deployment_info = model_deployer.deploy(model, model_metadata)
    else:
        # Check if model meets deployment criteria
        logger.info("Checking if model meets deployment quality criteria")
        if model_deployer.should_deploy(evaluation_results):
            # Deploy model
            logger.info("Model meets quality criteria, proceeding with deployment")
            deployment_info = model_deployer.deploy(model, model_metadata)
        else:
            logger.info("Model did not meet deployment criteria. Keeping existing production model.")
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
    
    This function configures monitoring to track the model's
    performance and detect issues like data drift.
    
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
    logger.info(f"Setting up monitoring checks for run {run_id}")
    monitoring_info = model_monitor.setup_monitoring(run_id)
    
    # Save monitoring configuration
    monitor_dir = os.path.join(config['data_storage']['monitoring_dir'], run_id)
    os.makedirs(monitor_dir, exist_ok=True)
    
    joblib.dump(monitoring_info, os.path.join(monitor_dir, 'monitoring_info.joblib'))
    
    # Create monitoring dashboard if enabled
    if config['monitoring'].get('create_dashboard', False):
        logger.info("Generating monitoring dashboard")
        dashboard_path = os.path.join(monitor_dir, 'monitoring_dashboard.html')
        model_monitor.generate_monitoring_report(run_id)
    
    logger.info(f"Model monitoring setup completed")
    
    return {'monitoring_info': monitoring_info}


if __name__ == "__main__":
    main()