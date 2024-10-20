import numpy as np
import pandas as pd
import pickle
import os
import yaml
import logging
from sklearn.ensemble import GradientBoostingClassifier

# Configure logging
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('././logs/model_building.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_data(data_path: str) -> pd.DataFrame:
    """Loads the training data."""
    try:
        train_data = pd.read_csv(data_path)
        logger.info("Successfully loaded training data.")
        return train_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def load_params(param_path: str) -> tuple:
    """Loads model parameters from params.yaml."""
    try:
        params = yaml.safe_load(open(param_path, 'r'))
        n_estimators = params['model_building']['n_estimators']
        learning_rate = params['model_building']['learning_rate']
        logger.info(f"Loaded parameters: n_estimators={n_estimators}, learning_rate={learning_rate}")
        return n_estimators, learning_rate
    except FileNotFoundError as e:
        logger.error(f"Params file not found: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing key in params.yaml: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing params.yaml: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, learning_rate: float) -> GradientBoostingClassifier:
    """Trains the GradientBoostingClassifier."""
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X_train, y_train)
        logger.info("Successfully trained the GradientBoostingClassifier model.")
        return clf
    except ValueError as e:
        logger.error(f"Model training failed: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        raise

def save_model(model, output_path: str) -> None:
    """Saves the trained model as a pickle file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model successfully saved to {output_path}")
    except OSError as e:
        logger.error(f"Error creating directory or writing file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the model: {e}")
        raise

def main():
    try:
        # Load training data
        train_data = load_data('././data/interim/train_bow.csv')

        # Load parameters from params.yaml
        n_estimators, learning_rate = load_params('././params.yaml')

        # Prepare training data
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train the model
        clf = train_model(X_train, y_train, n_estimators, learning_rate)

        # Save the trained model
        save_model(clf, '././models')

    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()