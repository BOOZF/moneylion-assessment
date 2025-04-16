# Full pipeline
python main.py --config config/pipeline_config.yaml --mode full

# Individual stages
python main.py --mode ingest  # Data ingestion only
python main.py --mode train   # Model training only

# True Negatives (Top-left): 106,311 loans were correctly predicted as "Not Originated"
# False Positives (Top-right): 25 loans were incorrectly predicted as "Originated" when they were actually not
# False Negatives (Bottom-left): 30 loans were incorrectly predicted as "Not Originated" when they actually were
# True Positives (Bottom-right): 9,171 loans were correctly predicted as "Originated"