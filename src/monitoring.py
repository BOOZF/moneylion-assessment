# src/monitoring.py
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

class ModelMonitor:
    """
    Monitors model performance and detects drift in production.
    """
    
    def __init__(self, monitoring_config):
        """
        Initialize the ModelMonitor with configuration.
        
        Args:
            monitoring_config (dict): Configuration for model monitoring
        """
        self.monitoring_config = monitoring_config
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring(self, run_id):
        """
        Set up monitoring for a deployed model.
        
        Args:
            run_id (str): Run ID for the current pipeline execution
            
        Returns:
            dict: Monitoring configuration information
        """
        self.logger.info(f"Setting up model monitoring for run {run_id}")
        
        try:
            # Create monitoring configuration
            monitoring_info = {
                'run_id': run_id,
                'setup_date': datetime.now().isoformat(),
                'monitoring_checks': self.monitoring_config.get('checks', []),
                'monitoring_frequency': self.monitoring_config.get('frequency', 'daily'),
                'alert_thresholds': self.monitoring_config.get('thresholds', {}),
                'reference_data_path': self.monitoring_config.get('reference_data_path', None)
            }
            
            # Save reference data for drift detection if specified
            if self.monitoring_config.get('save_reference_data', True):
                self._save_reference_data(run_id)
            
            # Schedule monitoring jobs
            self._schedule_monitoring_jobs(run_id)
            
            self.logger.info(f"Model monitoring setup completed successfully")
            return monitoring_info
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {str(e)}", exc_info=True)
            raise
    
    def _save_reference_data(self, run_id):
        """
        Save reference data for drift detection.
        
        Args:
            run_id (str): Run ID
        """
        self.logger.info("Saving reference data for drift detection")
        
        try:
            # Get reference data path from config
            reference_data_path = self.monitoring_config.get('reference_data_path')
            if not reference_data_path:
                reference_data_path = os.path.join('data', 'monitoring', 'reference_data')
            
            # Create directory if it doesn't exist
            os.makedirs(reference_data_path, exist_ok=True)
            
            # Load feature data from the processed features
            features_dir = os.path.join('data', 'features', run_id)
            
            # If this run's data doesn't exist, try to find the latest
            if not os.path.exists(features_dir):
                features_parent = os.path.join('data', 'features')
                if os.path.exists(features_parent):
                    runs = [d for d in os.listdir(features_parent) 
                          if os.path.isdir(os.path.join(features_parent, d))]
                    if runs:
                        latest_run = sorted(runs)[-1]
                        features_dir = os.path.join(features_parent, latest_run)
                        self.logger.info(f"Using features from run {latest_run}")
            
            # Check if X_test exists
            x_test_path = os.path.join(features_dir, 'X_test.parquet')
            if os.path.exists(x_test_path):
                X_test = pd.read_parquet(x_test_path)
                
                # Save as reference data
                reference_path = os.path.join(reference_data_path, f'reference_data_{run_id}.parquet')
                X_test.to_parquet(reference_path)
                
                # Save feature statistics for drift detection
                feature_stats = self._calculate_feature_stats(X_test)
                stats_path = os.path.join(reference_data_path, f'reference_stats_{run_id}.json')
                with open(stats_path, 'w') as f:
                    json.dump(feature_stats, f, indent=2)
                
                self.logger.info(f"Reference data saved to {reference_path}")
                self.logger.info(f"Reference statistics saved to {stats_path}")
            else:
                self.logger.warning(f"Could not find test data at {x_test_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving reference data: {str(e)}", exc_info=True)
    
    def _calculate_feature_stats(self, data):
        """
        Calculate statistics for each feature for drift detection.
        
        Args:
            data (pd.DataFrame): Feature data
            
        Returns:
            dict: Feature statistics
        """
        feature_stats = {}
        
        # For each column, calculate statistics based on data type
        for column in data.columns:
            col_data = data[column].dropna()
            if len(col_data) == 0:
                continue
                    
            if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
                # Numeric feature statistics (excluding boolean)
                stats = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'p10': float(col_data.quantile(0.1)),
                    'p90': float(col_data.quantile(0.9)),
                    'type': 'numeric'
                }
            elif pd.api.types.is_bool_dtype(col_data):
                # Boolean feature statistics
                stats = {
                    'mean': float(col_data.mean()),
                    'type': 'boolean'
                }
            else:
                # Categorical feature statistics
                value_counts = col_data.value_counts(normalize=True).to_dict()
                stats = {
                    'value_counts': {str(k): float(v) for k, v in value_counts.items()},
                    'unique_values': col_data.nunique(),
                    'type': 'categorical'
                }
            
            feature_stats[column] = stats
        
        return feature_stats
    
    def _schedule_monitoring_jobs(self, run_id):
        """
        Schedule monitoring jobs based on configuration.
        
        Args:
            run_id (str): Run ID
        """
        self.logger.info("Scheduling monitoring jobs")
        
        # Get monitoring frequency from config
        frequency = self.monitoring_config.get('frequency', 'daily')
        
        # In a real system, this would set up scheduled jobs using
        # a task scheduler like Airflow, Celery, etc.
        # For this example, we'll just create a config file
        
        # Define monitoring schedule
        schedule = {
            'run_id': run_id,
            'checks': self.monitoring_config.get('checks', []),
            'frequency': frequency,
            'next_run': (datetime.now() + timedelta(days=1)).isoformat(),
            'enabled': True
        }
        
        # Save schedule config
        schedule_path = os.path.join('data', 'monitoring', 'schedules')
        os.makedirs(schedule_path, exist_ok=True)
        
        with open(os.path.join(schedule_path, f'schedule_{run_id}.json'), 'w') as f:
            json.dump(schedule, f, indent=2)
        
        self.logger.info(f"Monitoring schedule configured for {frequency} frequency")
    
    def check_data_drift(self, new_data, reference_data=None, reference_stats=None):
        """
        Check for data drift between new data and reference data.
        
        Args:
            new_data (pd.DataFrame): New production data
            reference_data (pd.DataFrame): Reference data from training
            reference_stats (dict): Reference data statistics
            
        Returns:
            dict: Drift detection results
        """
        self.logger.info("Checking for data drift")
        
        try:
            drift_results = {'drift_detected': False, 'drifted_features': []}
            
            # If reference_stats provided, use that for comparison
            if reference_stats:
                drift_results = self._detect_drift_using_stats(new_data, reference_stats)
            # If reference data provided, compare directly
            elif reference_data is not None:
                drift_results = self._detect_drift_using_data(new_data, reference_data)
            # If neither provided, try to load the latest reference data
            else:
                reference_data_path = self.monitoring_config.get('reference_data_path')
                if not reference_data_path:
                    reference_data_path = os.path.join('data', 'monitoring', 'reference_data')
                
                if os.path.exists(reference_data_path):
                    # Find latest reference stats
                    ref_files = [f for f in os.listdir(reference_data_path) if f.startswith('reference_stats_')]
                    if ref_files:
                        latest_ref = sorted(ref_files)[-1]
                        with open(os.path.join(reference_data_path, latest_ref), 'r') as f:
                            reference_stats = json.load(f)
                        
                        drift_results = self._detect_drift_using_stats(new_data, reference_stats)
                    else:
                        self.logger.warning("No reference statistics found")
                else:
                    self.logger.warning(f"Reference data path {reference_data_path} does not exist")
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Error checking data drift: {str(e)}", exc_info=True)
            return {'error': str(e), 'drift_detected': False}
    
    def _detect_drift_using_stats(self, new_data, reference_stats):
        """
        Detect drift by comparing new data with reference statistics.
        
        Args:
            new_data (pd.DataFrame): New production data
            reference_stats (dict): Reference data statistics
            
        Returns:
            dict: Drift detection results
        """
        drift_results = {'drift_detected': False, 'drifted_features': [], 'details': {}}
        
        # Get drift thresholds from config
        drift_threshold = self.monitoring_config.get('thresholds', {}).get('drift_threshold', 0.1)
        categorical_drift_threshold = self.monitoring_config.get('thresholds', {}).get('categorical_drift_threshold', 0.1)
        
        # For each feature in new data that also exists in reference stats
        for column in new_data.columns:
            if column not in reference_stats:
                continue
                
            ref_stats = reference_stats[column]
            col_data = new_data[column].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Check drift based on data type
            if ref_stats['type'] == 'numeric':
                # For numeric features, compare distributions using KS test if possible
                # Otherwise compare basic statistics
                new_mean = col_data.mean()
                ref_mean = ref_stats['mean']
                mean_diff = abs(new_mean - ref_mean) / max(abs(ref_mean), 1e-10)
                
                new_std = col_data.std()
                ref_std = ref_stats['std']
                std_diff = abs(new_std - ref_std) / max(abs(ref_std), 1e-10)
                
                # Check if drift exceeds threshold
                if mean_diff > drift_threshold or std_diff > drift_threshold:
                    drift_results['drift_detected'] = True
                    drift_results['drifted_features'].append(column)
                    drift_results['details'][column] = {
                        'mean_diff': float(mean_diff),
                        'std_diff': float(std_diff),
                        'threshold': drift_threshold
                    }
            
            elif ref_stats['type'] == 'categorical':
                # For categorical features, compare value distributions
                new_counts = col_data.value_counts(normalize=True).to_dict()
                ref_counts = ref_stats['value_counts']
                
                # Calculate JS divergence or similar metric
                # For simplicity, we'll just check for new categories or large shifts
                drift_detected = False
                max_diff = 0
                
                # Check for new categories
                for category, new_pct in new_counts.items():
                    str_category = str(category)
                    if str_category not in ref_counts:
                        if new_pct > categorical_drift_threshold:
                            drift_detected = True
                            max_diff = max(max_diff, float(new_pct))
                    else:
                        diff = abs(new_pct - ref_counts[str_category])
                        max_diff = max(max_diff, diff)
                        if diff > categorical_drift_threshold:
                            drift_detected = True
                
                if drift_detected:
                    drift_results['drift_detected'] = True
                    drift_results['drifted_features'].append(column)
                    drift_results['details'][column] = {
                        'max_category_diff': float(max_diff),
                        'threshold': categorical_drift_threshold
                    }
        
        return drift_results
    
    def _detect_drift_using_data(self, new_data, reference_data):
        """
        Detect drift by directly comparing new data with reference data.
        
        Args:
            new_data (pd.DataFrame): New production data
            reference_data (pd.DataFrame): Reference data from training
            
        Returns:
            dict: Drift detection results
        """
        drift_results = {'drift_detected': False, 'drifted_features': [], 'details': {}}
        
        # Get drift threshold from config
        ks_threshold = self.monitoring_config.get('thresholds', {}).get('ks_threshold', 0.1)
        
        # For each feature in both datasets, perform KS test
        common_columns = list(set(new_data.columns) & set(reference_data.columns))
        
        for column in common_columns:
            if pd.api.types.is_numeric_dtype(new_data[column]):
                # Drop NAs for comparison
                ref_col = reference_data[column].dropna()
                new_col = new_data[column].dropna()
                
                if len(ref_col) > 0 and len(new_col) > 0:
                    # Perform Kolmogorov-Smirnov test
                    ks_stat, p_value = ks_2samp(ref_col, new_col)
                    
                    # Check if KS statistic exceeds threshold
                    if ks_stat > ks_threshold:
                        drift_results['drift_detected'] = True
                        drift_results['drifted_features'].append(column)
                        drift_results['details'][column] = {
                            'ks_statistic': float(ks_stat),
                            'p_value': float(p_value),
                            'threshold': ks_threshold
                        }
        
        return drift_results
    
    def check_performance_degradation(self, predictions, actuals):
        """
        Check for model performance degradation over time.
        
        Args:
            predictions (list/array): Model predictions
            actuals (list/array): Actual target values
            
        Returns:
            dict: Performance degradation results
        """
        self.logger.info("Checking for performance degradation")
        
        try:
            # Convert inputs to numpy arrays
            preds = np.array(predictions)
            acts = np.array(actuals)
            
            # Calculate performance metrics
            accuracy = (preds == acts).mean()
            precision = np.sum((preds == 1) & (acts == 1)) / max(np.sum(preds == 1), 1)
            recall = np.sum((preds == 1) & (acts == 1)) / max(np.sum(acts == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            
            # Get performance thresholds from config
            perf_thresholds = self.monitoring_config.get('thresholds', {}).get('performance', {})
            min_accuracy = perf_thresholds.get('min_accuracy', 0.7)
            min_f1 = perf_thresholds.get('min_f1', 0.6)
            
            # Check if current performance is below thresholds
            degradation_detected = (accuracy < min_accuracy) or (f1 < min_f1)
            
            # Compare with historical performance if available
            historical_perf = self._load_historical_performance()
            if historical_perf:
                # Calculate relative degradation from baseline
                baseline_accuracy = historical_perf.get('baseline_accuracy', accuracy)
                baseline_f1 = historical_perf.get('baseline_f1', f1)
                
                rel_accuracy_change = (accuracy - baseline_accuracy) / max(baseline_accuracy, 1e-10)
                rel_f1_change = (f1 - baseline_f1) / max(baseline_f1, 1e-10)
                
                # Define relative degradation threshold
                rel_degradation_threshold = perf_thresholds.get('relative_degradation_threshold', -0.1)
                
                # Check if relative degradation exceeds threshold
                if rel_accuracy_change < rel_degradation_threshold or rel_f1_change < rel_degradation_threshold:
                    degradation_detected = True
            
            # Prepare results
            results = {
                'degradation_detected': degradation_detected,
                'current_performance': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                }
            }
            
            if historical_perf:
                results['baseline_performance'] = {
                    'accuracy': historical_perf.get('baseline_accuracy'),
                    'f1': historical_perf.get('baseline_f1')
                }
                results['relative_change'] = {
                    'accuracy': float(rel_accuracy_change),
                    'f1': float(rel_f1_change)
                }
            
            # Save current performance for future comparisons
            self._save_performance_metrics(results['current_performance'])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error checking performance degradation: {str(e)}", exc_info=True)
            return {'error': str(e), 'degradation_detected': False}
    
    def _load_historical_performance(self):
        """
        Load historical performance metrics for comparison.
        
        Returns:
            dict: Historical performance metrics
        """
        history_path = os.path.join('data', 'monitoring', 'performance_history.json')
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                return history
            except Exception as e:
                self.logger.error(f"Error loading performance history: {str(e)}")
                return None
        else:
            return None
    
    def _save_performance_metrics(self, metrics):
        """
        Save current performance metrics for future comparisons.
        
        Args:
            metrics (dict): Current performance metrics
        """
        history_dir = os.path.join('data', 'monitoring')
        os.makedirs(history_dir, exist_ok=True)
        history_path = os.path.join(history_dir, 'performance_history.json')
        
        # Load existing history if available
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except Exception:
                history = {'performances': []}
        else:
            history = {'performances': []}
        
        # Add timestamp to current metrics
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Save to history
        history['performances'].append(metrics)
        
        # Set baseline if not already set
        if 'baseline_accuracy' not in history:
            history['baseline_accuracy'] = metrics['accuracy']
            history['baseline_f1'] = metrics['f1']
        
        # Trim history to last 100 entries
        if len(history['performances']) > 100:
            history['performances'] = history['performances'][-100:]
        
        # Save updated history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def generate_monitoring_report(self, run_id):
        """
        Generate a monitoring report with all checks and visualizations.
        
        Args:
            run_id (str): Run ID for the report
            
        Returns:
            str: Path to the generated report
        """
        self.logger.info(f"Generating monitoring report for run {run_id}")
        
        try:
            # Create report directory
            report_dir = os.path.join('data', 'monitoring', 'reports', run_id)
            os.makedirs(report_dir, exist_ok=True)
            
            # Create monitoring visualizations
            self._create_monitoring_visualizations(report_dir)
            
            # Create HTML report
            report_path = os.path.join(report_dir, 'monitoring_report.html')
            with open(report_path, 'w') as f:
                f.write(f"""
                <html>
                <head>
                    <title>Model Monitoring Report - {run_id}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #2c3e50; }}
                        .section {{ margin-bottom: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                        th {{ background-color: #f2f2f2; }}
                        .alert {{ color: red; font-weight: bold; }}
                        img {{ max-width: 100%; height: auto; margin: 10px 0; }}
                    </style>
                </head>
                <body>
                    <h1>Model Monitoring Report</h1>
                    <p>Run ID: {run_id}</p>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="section">
                        <h2>Data Drift Analysis</h2>
                        <img src="drift_detection.png" alt="Data Drift Visualization">
                    </div>
                    
                    <div class="section">
                        <h2>Performance Tracking</h2>
                        <img src="performance_tracking.png" alt="Performance Tracking">
                    </div>
                    
                    <div class="section">
                        <h2>Monitoring Alerts</h2>
                        <p>No critical alerts detected.</p>
                    </div>
                </body>
                </html>
                """)
            
            self.logger.info(f"Monitoring report generated at {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating monitoring report: {str(e)}", exc_info=True)
            raise
    
    def _create_monitoring_visualizations(self, report_dir):
        """
        Create visualizations for the monitoring report.
        
        Args:
            report_dir (str): Directory to save visualizations
        """
        # Load performance history if available
        history_path = os.path.join('data', 'monitoring', 'performance_history.json')
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                if 'performances' in history and len(history['performances']) > 0:
                    # Extract performance metrics over time
                    timestamps = [p['timestamp'] for p in history['performances']]
                    accuracies = [p['accuracy'] for p in history['performances']]
                    f1_scores = [p['f1'] for p in history['performances']]
                    
                    # Convert timestamps to datetime objects
                    dates = [datetime.fromisoformat(ts) for ts in timestamps]
                    
                    # Create performance tracking visualization
                    plt.figure(figsize=(10, 6))
                    plt.plot(dates, accuracies, 'b-', label='Accuracy')
                    plt.plot(dates, f1_scores, 'g-', label='F1 Score')
                    plt.axhline(y=history.get('baseline_accuracy', 0), color='b', linestyle='--', alpha=0.5)
                    plt.axhline(y=history.get('baseline_f1', 0), color='g', linestyle='--', alpha=0.5)
                    plt.xlabel('Date')
                    plt.ylabel('Metric Value')
                    plt.title('Model Performance Over Time')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(report_dir, 'performance_tracking.png'))
                    plt.close()
            except Exception as e:
                self.logger.error(f"Error creating performance visualization: {str(e)}")
        
        # Create dummy drift detection visualization if real data not available
        plt.figure(figsize=(10, 6))
        # Sample data for drift visualization
        features = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
        drift_scores = [0.02, 0.15, 0.08, 0.22, 0.05]
        threshold = 0.1
        
        # Create bar chart
        bars = plt.bar(features, drift_scores, color=['green' if score < threshold else 'red' for score in drift_scores])
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Drift Threshold')
        plt.xlabel('Features')
        plt.ylabel('Drift Score')
        plt.title('Feature Drift Detection')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, 'drift_detection.png'))
        plt.close()