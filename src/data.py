import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataIngestion:
    """
    Handles the ingestion of data from various sources into the ML pipeline.
    """
    
    def __init__(self, data_source_config, data_storage_config):
        """
        Initialize the DataIngestion with configuration.
        
        Args:
            data_source_config (dict): Configuration for data sources
            data_storage_config (dict): Configuration for data storage
        """
        self.data_source_config = data_source_config
        self.data_storage_config = data_storage_config
        self.logger = logging.getLogger(__name__)
    
    def ingest_loan_data(self):
        """
        Ingest loan application data with advanced filtering.
        
        Returns:
            pd.DataFrame: The filtered and processed loan data
        """
        self.logger.info("Ingesting and filtering loan data")
        
        try:
            # Load from source defined in config
            loan_source = self.data_source_config.get('loan_data', {})
            source_path = loan_source.get('path', 'data/loan.csv')
            
            # Check if file exists
            if not os.path.exists(source_path):
                self.logger.error(f"Loan data file not found at {source_path}")
                raise FileNotFoundError(f"Loan data file not found at {source_path}")
            
            # Load the data based on file extension
            if source_path.endswith('.csv'):
                loan_df = pd.read_csv(source_path)
            elif source_path.endswith('.parquet'):
                loan_df = pd.read_parquet(source_path)
            else:
                self.logger.error(f"Unsupported file format for {source_path}")
                raise ValueError(f"Unsupported file format for {source_path}")
            
            # Advanced Filtering
            # 1. Filter only approved loans
            approved_loans = loan_df[loan_df['approved'] == True]
            
            # 2. Check loan status distribution for approved loans
            status_dist = approved_loans['loanStatus'].value_counts(normalize=True)
            self.logger.info("Loan Status Distribution for Approved Loans:")
            self.logger.info(status_dist)
            
            # Optional: Further filter based on loan status if needed
            # For example, you might want to focus on specific statuses like 'Paid Off'
            # approved_loans = approved_loans[approved_loans['loanStatus'].isin(['Paid Off'])]
            
            # Validate the filtered data
            self._validate_filtered_loan_data(approved_loans)
            
            self.logger.info(f"Successfully filtered loan data. Remaining rows: {approved_loans.shape[0]}")
            return approved_loans
            
        except Exception as e:
            self.logger.error(f"Error ingesting loan data: {str(e)}", exc_info=True)
            raise
    
    def ingest_clarity_data(self):
        """
        Ingest clarity/fraud check data.
        
        Returns:
            pd.DataFrame: The ingested clarity data
        """
        self.logger.info("Ingesting clarity data")
        
        try:
            # Load from source defined in config
            clarity_source = self.data_source_config.get('clarity_data', {})
            source_path = clarity_source.get('path', 'data/clarity_underwriting_variables.csv')
            
            # Check if file exists
            if not os.path.exists(source_path):
                self.logger.warning(f"Clarity data file not found at {source_path}. Creating empty DataFrame.")
                # Return an empty DataFrame with expected structure
                return pd.DataFrame(columns=['underwritingid', 'clearfraudscore'])
            
            # Load the data based on file extension
            if source_path.endswith('.csv'):
                clarity_df = pd.read_csv(source_path, low_memory=False)
            elif source_path.endswith('.parquet'):
                clarity_df = pd.read_parquet(source_path)
            else:
                self.logger.error(f"Unsupported file format for {source_path}")
                raise ValueError(f"Unsupported file format for {source_path}")
            
            # Apply basic validation
            self._validate_clarity_data(clarity_df)
            
            self.logger.info(f"Successfully ingested clarity data with {clarity_df.shape[0]} rows and {clarity_df.shape[1]} columns")
            return clarity_df
            
        except Exception as e:
            self.logger.error(f"Error ingesting clarity data: {str(e)}", exc_info=True)
            raise
    
    def ingest_payment_data(self):
        """
        Ingest payment history data.
        
        Returns:
            pd.DataFrame: The ingested payment data
        """
        self.logger.info("Ingesting payment data")
        
        try:
            # Load from source defined in config
            payment_source = self.data_source_config.get('payment_data', {})
            source_path = payment_source.get('path', 'data/payment.csv')
            
            # Check if file exists
            if not os.path.exists(source_path):
                self.logger.warning(f"Payment data file not found at {source_path}. Creating empty DataFrame.")
                # Return an empty DataFrame with expected structure
                return pd.DataFrame(columns=['loanId', 'installmentIndex', 'paymentAmount', 'paymentStatus'])
            
            # Load the data based on file extension
            if source_path.endswith('.csv'):
                payment_df = pd.read_csv(source_path)
            elif source_path.endswith('.parquet'):
                payment_df = pd.read_parquet(source_path)
            else:
                self.logger.error(f"Unsupported file format for {source_path}")
                raise ValueError(f"Unsupported file format for {source_path}")
            
            # Apply basic validation
            self._validate_payment_data(payment_df)
            
            self.logger.info(f"Successfully ingested payment data with {payment_df.shape[0]} rows and {payment_df.shape[1]} columns")
            return payment_df
            
        except Exception as e:
            self.logger.error(f"Error ingesting payment data: {str(e)}", exc_info=True)
            raise
    
    def _validate_filtered_loan_data(self, df):
        """
        Validate the structure and content of filtered loan data.
        
        Args:
            df (pd.DataFrame): Filtered loan data to validate
        """
        # Check for any remaining required columns
        required_columns = ['loanId', 'clarityFraudId', 'apr', 'loanAmount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"Filtered loan data is missing required columns: {missing_columns}")
        
        # Check for duplicate loan IDs
        if df['loanId'].duplicated().any():
            self.logger.warning(f"Filtered loan data contains {df['loanId'].duplicated().sum()} duplicate loan IDs")
        
        # Check for missing clarityFraudId
        missing_fraud_ids = df['clarityFraudId'].isna().sum()
        if missing_fraud_ids > 0:
            self.logger.warning(f"{missing_fraud_ids} rows have missing clarityFraudId")
        
        # Additional data quality checks could be added here
    
    def _validate_clarity_data(self, df):
        """
        Validate the structure and content of clarity data.
        
        Args:
            df (pd.DataFrame): Clarity data to validate
        """
        # Check if the key identifier column exists
        if 'underwritingid' not in df.columns:
            self.logger.warning("Clarity data is missing 'underwritingid' column")
        
        # Additional data quality checks could be added here
    
    def _validate_payment_data(self, df):
        """
        Validate the structure and content of payment data.
        
        Args:
            df (pd.DataFrame): Payment data to validate
        """
        # Check required columns
        required_columns = ['loanId', 'paymentAmount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"Payment data is missing required columns: {missing_columns}")
        
        # Additional data quality checks could be added here


class DataProcessor:
    """
    Processes raw data to create features for model training.
    """
    
    def __init__(self, feature_config):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            feature_config (dict): Configuration for feature engineering
        """
        self.feature_config = feature_config
        self.logger = logging.getLogger(__name__)
        self.numeric_features = []
        self.categorical_features = []
    
    def __init__(self, feature_config):
        """
        Initialize the DataProcessor with configuration.
        
        Args:
            feature_config (dict): Configuration for feature engineering
        """
        self.feature_config = feature_config
        self.logger = logging.getLogger(__name__)
        self.numeric_features = []
        self.categorical_features = []
    
    def process(self, loan_df, clarity_df, payment_df):
        """
        Process the raw data to create features for model training.
        
        Args:
            loan_df (pd.DataFrame): Approved loan application data
            clarity_df (pd.DataFrame): Clarity/fraud check data
            payment_df (pd.DataFrame): Payment history data
            
        Returns:
            pd.DataFrame: Processed data ready for model training
        """
        self.logger.info("Processing data for feature engineering")
        
        try:
            # 1. Enhance data with fraud score from clarity data
            processed_df = self._join_fraud_score(loan_df, clarity_df)
            
            # 2. Add payment features
            processed_df = self._add_payment_features(processed_df, payment_df)
            
            # 3. Preprocess dates
            processed_df = self._preprocess_dates(processed_df)
            
            # 4. Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # 5. Feature engineering
            processed_df = self._engineer_features(processed_df)
            
            # 6. Define target variable
            processed_df = self._define_target(processed_df)
            
            self.logger.info(f"Data processing completed. Final dataset has {processed_df.shape[0]} rows and {processed_df.shape[1]} columns")
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}", exc_info=True)
            raise

    def _join_fraud_score(self, loan_df, clarity_df):
        """
        Join loan data with clarity fraud score.
        
        Args:
            loan_df (pd.DataFrame): Loan application data
            clarity_df (pd.DataFrame): Clarity/fraud check data
            
        Returns:
            pd.DataFrame: Joined dataframe with fraud score
        """
        self.logger.info("Joining loan data with clarity fraud score")
        
        # Check if clarity data is available and has data
        if clarity_df.empty:
            self.logger.warning("Clarity data is empty, skipping fraud score join")
            return loan_df
        
        # Check join conditions
        if 'clarityFraudId' not in loan_df.columns or 'underwritingid' not in clarity_df.columns:
            self.logger.warning("Missing join columns for clarity data")
            return loan_df
        
        # Perform the join
        try:
            # Left join to keep all loan records
            processed_df = loan_df.merge(
                clarity_df[['underwritingid', 'clearfraudscore']], 
                left_on='clarityFraudId', 
                right_on='underwritingid', 
                how='left'
            )
            
            # Log fraud score distribution
            if 'clearfraudscore' in processed_df.columns:
                fraud_score_dist = processed_df['clearfraudscore'].describe()
                self.logger.info("Fraud Score Distribution:")
                self.logger.info(fraud_score_dist)
                
                # Add to numeric features
                self.numeric_features.append('clearfraudscore')
            
            return processed_df
            
        except Exception as e:
            self.logger.error(f"Error joining fraud score: {str(e)}")
            return loan_df
    
    # Rest of the methods remain mostly the same as in the original implementation
    # (methods like _preprocess_dates, _add_payment_features, etc.)
    
    def _preprocess_dates(self, df):
        """
        Process date columns and extract useful features.
        
        Args:
            df (pd.DataFrame): DataFrame with date columns
            
        Returns:
            pd.DataFrame: DataFrame with processed date features
        """
        self.logger.info("Preprocessing date columns")
        
        # Convert date strings to datetime objects
        date_columns = ['applicationDate', 'originatedDate']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Extract features from applicationDate
        if 'applicationDate' in df.columns:
            df['application_year'] = df['applicationDate'].dt.year
            df['application_month'] = df['applicationDate'].dt.month
            df['application_day'] = df['applicationDate'].dt.day
            df['application_dayofweek'] = df['applicationDate'].dt.dayofweek
            df['application_hour'] = df['applicationDate'].dt.hour
            
            # Add these to numeric features list
            self.numeric_features.extend([
                'application_year', 'application_month', 'application_day',
                'application_dayofweek', 'application_hour'
            ])
        
        # Calculate time to origination for originated loans
        if 'applicationDate' in df.columns and 'originatedDate' in df.columns:
            df['days_to_origination'] = np.nan
            mask = ~df['originatedDate'].isna()
            if any(mask):
                df.loc[mask, 'days_to_origination'] = (
                    df.loc[mask, 'originatedDate'] - df.loc[mask, 'applicationDate']
                ).dt.total_seconds() / (24 * 60 * 60)
                
                self.numeric_features.append('days_to_origination')
        
        return df
        
    def _add_payment_features(self, processed_df, payment_df):
        """
        Add payment history features to the loan data.
        
        Args:
            processed_df (pd.DataFrame): Processed loan data
            payment_df (pd.DataFrame): Payment history data
            
        Returns:
            pd.DataFrame: DataFrame with added payment features
        """
        self.logger.info("Adding payment history features")
        
        # Check if payment data is available and has data
        if payment_df.empty:
            self.logger.warning("Payment data is empty, skipping payment features")
            return processed_df
        
        # Check if we have the needed columns
        required_cols = ['loanId', 'paymentAmount', 'paymentStatus']
        missing_cols = [col for col in required_cols if col not in payment_df.columns]
        
        if missing_cols:
            self.logger.warning(f"Payment data is missing columns: {missing_cols}, skipping payment features")
            return processed_df
        
        try:
            # Aggregate payment data at the loan level
            payment_stats = payment_df.groupby('loanId').agg({
                'paymentAmount': ['mean', 'sum', 'count'],
                'principal': ['mean', 'sum'] if 'principal' in payment_df.columns else [],
                'fees': ['mean', 'sum'] if 'fees' in payment_df.columns else [],
                'paymentStatus': lambda x: (x == 'Checked').mean() if 'paymentStatus' in payment_df.columns else 0
            }).reset_index()
            
            # Flatten the column names
            payment_stats.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] 
                for col in payment_stats.columns
            ]
            
            # Rename the paymentStatus lambda column
            if 'paymentStatus_<lambda>' in payment_stats.columns:
                payment_stats.rename(columns={'paymentStatus_<lambda>': 'payment_success_rate'}, inplace=True)
            
            # Join with the processed data
            result = processed_df.merge(
                payment_stats,
                on='loanId',
                how='left'
            )
            
            # Add new columns to numeric features list
            new_numeric_features = [col for col in payment_stats.columns if col != 'loanId']
            self.numeric_features.extend(new_numeric_features)
            
            # Fill NAs for loans without payment data
            for col in new_numeric_features:
                if col in result.columns:
                    result[col] = result[col].fillna(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error adding payment features: {str(e)}", exc_info=True)
            # If there's an error, return the original dataframe
            return processed_df
        
    def _handle_missing_values(self, processed_df):
        """
        Handle missing values in the processed data.
        
        Args:
            processed_df (pd.DataFrame): Processed data with potential missing values
            
        Returns:
            pd.DataFrame: Data with handled missing values
        """
        self.logger.info("Handling missing values")
        
        # Make a copy to avoid modifying the original
        df = processed_df.copy()
        
        # List of columns that should have missing indicators
        missing_indicator_columns = [
            'days_to_origination', 'nPaidOff', 'clearfraudscore', 
            'application_year', 'application_month', 'application_day', 
            'application_dayofweek', 'application_hour', 
            'loanAmount', 'apr', 'leadCost'
        ]
        
        # 1. Handle numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            missing_pct = df[col].isna().mean() * 100
            
            if missing_pct > 0:
                self.logger.info(f"Column {col} has {missing_pct:.2f}% missing values")
                
                # Create missing indicator if the column is in our list
                if col in missing_indicator_columns:
                    missing_col = f"{col}_missing"
                    df[missing_col] = df[col].isna().astype(int)
                    
                    # Add to numeric features if not already present
                    if missing_col not in self.numeric_features:
                        self.numeric_features.append(missing_col)
                
                # Fill missing values
                if missing_pct < 10:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # 2. Ensure all expected missing indicator columns exist
        for col in missing_indicator_columns:
            missing_col = f"{col}_missing"
            if missing_col not in df.columns:
                df[missing_col] = 0
                if missing_col not in self.numeric_features:
                    self.numeric_features.append(missing_col)
        
        # 3. Handle categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            missing_pct = df[col].isna().mean() * 100
            
            if missing_pct > 0:
                self.logger.info(f"Column {col} has {missing_pct:.2f}% missing values")
                
                # Fill with 'Unknown' for categorical
                df[col] = df[col].fillna('Unknown')
                
                # If this is a new categorical column, add it to our list
                if col not in self.categorical_features and col != 'loanId':
                    self.categorical_features.append(col)
        
        # 4. Handle boolean features
        boolean_cols = df.select_dtypes(include=['bool']).columns
        for col in boolean_cols:
            missing_pct = df[col].isna().mean() * 100
            
            if missing_pct > 0:
                self.logger.info(f"Column {col} has {missing_pct:.2f}% missing values")
                
                # Fill with False for boolean
                df[col] = df[col].fillna(False)
        
        # 5. Drop original date columns
        date_cols = ['applicationDate', 'originatedDate']
        for col in date_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors='ignore')
        
        return df
    
    def _engineer_features(self, processed_df):
        """
        Engineer features for model training.
        
        Args:
            processed_df (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        self.logger.info("Engineering features")
        
        # Make a copy to avoid modifying the original
        df = processed_df.copy()
        
        # 1. Define base features if not already set
        if not self.numeric_features:
            self.numeric_features = [
                'loanAmount', 'apr', 'nPaidOff', 'leadCost'
            ]
        
        if not self.categorical_features:
            self.categorical_features = [
                'state', 'leadType', 'payFrequency'
            ]
        
        # Make sure all specified features exist in the dataframe
        self.numeric_features = [f for f in self.numeric_features if f in df.columns]
        self.categorical_features = [f for f in self.categorical_features if f in df.columns]
        
        # 2. Add interaction features if configured
        if self.feature_config.get('create_interactions', False):
            self.logger.info("Creating interaction features")
            
            # Add interactions between select numeric features
            # For example: loanAmount * apr
            if 'loanAmount' in df.columns and 'apr' in df.columns:
                df['loanAmount_x_apr'] = df['loanAmount'] * df['apr']
                self.numeric_features.append('loanAmount_x_apr')
            
            # Ratio features
            if 'leadCost' in df.columns and 'loanAmount' in df.columns and (df['loanAmount'] > 0).all():
                df['leadCost_to_loanAmount'] = df['leadCost'] / df['loanAmount']
                self.numeric_features.append('leadCost_to_loanAmount')
        
        # 3. Add polynomial features if configured
        if self.feature_config.get('polynomial_features', False):
            self.logger.info("Creating polynomial features")
            
            for feature in self.feature_config.get('polynomial_features_list', []):
                if feature in df.columns:
                    # Square term
                    df[f"{feature}_squared"] = df[feature] ** 2
                    self.numeric_features.append(f"{feature}_squared")
        
        # 4. Add categorical encodings if configured
        if self.feature_config.get('categorical_encoding', 'onehot') == 'target':
            self.logger.info("Creating target encoding for categorical features")
            
            # We'll need the target for this
            if 'originated' in df.columns:
                for feature in self.categorical_features:
                    # Calculate mean target value per category
                    encoding = df.groupby(feature)['originated'].mean()
                    
                    # Map back to the dataframe
                    df[f"{feature}_target_encoded"] = df[feature].map(encoding)
                    
                    # Add to numeric features list
                    self.numeric_features.append(f"{feature}_target_encoded")
        
        # 5. Feature selection - drop low variance or highly correlated features
        if self.feature_config.get('drop_low_variance', False):
            self.logger.info("Dropping low variance features")
            
            # For numeric features
            for feature in self.numeric_features[:]:  # Use a copy for iteration
                if feature in df.columns:
                    # Calculate variance
                    variance = df[feature].var()
                    
                    # If variance is below threshold, drop the feature
                    if variance < self.feature_config.get('variance_threshold', 0.01):
                        self.logger.info(f"Dropping low variance feature: {feature} (variance: {variance:.6f})")
                        self.numeric_features.remove(feature)
                        df.drop(columns=[feature], inplace=True)
        
        # Return the dataframe with engineered features
        return df
    
    def _define_target(self, processed_df):
        """
        Define a risk score as the target variable based on loan status percentages.
        
        Args:
            processed_df (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Data with risk score target variable
        """
        self.logger.info("Defining risk score target variable")
        
        # Detailed risk mapping based on loan statuses
        risk_mapping = {
            # High Risk (0.8 - 1.0)
            'Rejected': 0.95,  # Highest risk
            'Withdrawn Application': 0.85,
            'External Collection': 0.80,
            'Internal Collection': 0.75,
            
            # Moderate Risk (0.4 - 0.8)
            'Returned Item': 0.65,
            'Pending Rescind': 0.55,
            'Pending Application Fee': 0.50,
            'Pending Paid Off': 0.45,
            
            # Low Risk (0.0 - 0.4)
            'Settled Bankruptcy': 0.30,
            'Paid Off Loan': 0.20,
            'Settlement Paid Off': 0.15,
            'Settlement Pending Paid Off': 0.10,
            'Customer Voided New Loan': 0.05,
            'CSR Voided New Loan': 0.05,
            
            # Default for any unmapped status
            'New Loan': 0.40,  # Neutral risk
        }
        
        # Create target column with risk scores
        processed_df['target'] = processed_df['loanStatus'].map(risk_mapping)
        
        # Fill any unmapped statuses with median risk
        processed_df['target'] = processed_df['target'].fillna(0.50)
        
        # Log target distribution
        target_stats = processed_df['target'].describe()
        self.logger.info("Risk Score Distribution:")
        self.logger.info(target_stats)
        
        # Optional: Visualize risk score distribution
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_df['target'], bins=20, kde=True)
            plt.title('Distribution of Risk Scores')
            plt.xlabel('Risk Score')
            plt.ylabel('Frequency')
            plt.savefig('risk_score_distribution.png')
            plt.close()
            self.logger.info("Risk score distribution plot saved to risk_score_distribution.png")
        except Exception as e:
            self.logger.warning(f"Could not create risk score distribution plot: {str(e)}")
        
        return processed_df