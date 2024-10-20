import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('./logs/model_evaluation.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_model(model_path: str):
    """Loads the trained model."""
    try:
        model = pickle.load(open(model_path, 'rb'))
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except pickle.PickleError as e:
        logger.error(f"Error loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the model: {e}")
        raise

def load_test_data(data_path: str) -> pd.DataFrame:
    """Loads the test dataset."""
    try:
        test_data = pd.read_csv(data_path)
        logger.info("Test data loaded successfully.")
        return test_data
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing the test data CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading test data: {e}")
        raise

def evaluate_model(clf, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates the model and calculates metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
        }
        logger.info("Model evaluation completed successfully.")
        return metrics
    except ValueError as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model evaluation: {e}")
        raise

def save_metrics(metrics: dict, output_path: str) -> None:
    """Saves the evaluation metrics as a JSON file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'metrics.json'), 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info("Metrics saved successfully to './metrics/metrics.json'.")
    except OSError as e:
        logger.error(f"Error creating directory or writing file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving metrics: {e}")
        raise

def main():
    try:
        # Load the model and test data
        clf = load_model('models/model.pkl')
        test_data = load_test_data('./data/features/test_bow.csv')

        # Prepare test data
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)

        # Save the metrics
        save_metrics(metrics, './metrics')

    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()
