import numpy as np
import pandas as pd
import os
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('././logs/feature_engineering.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_data(train_path: str, test_path: str) -> tuple:
    """Loads the processed training and test data."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info("Successfully loaded processed train and test data.")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV files: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

def load_params(param_path: str) -> int:
    """Loads parameters from the params.yaml file."""
    try:
        params = yaml.safe_load(open(param_path, 'r'))
        max_features = params['feature_engineering']['max_features']
        logger.info(f"Loaded max_features: {max_features}")
        return max_features
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
        logger.error(f"An unexpected error occurred while loading parameters: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures no NaN values are present in the 'content' column."""
    try:
        df['content'].fillna('', inplace=True)  # Replace NaNs with empty strings
        df = df[df['content'].str.strip() != '']  # Remove rows with empty strings
        logger.info("Successfully cleaned the data.")
        return df
    except KeyError as e:
        logger.error(f"Missing 'content' column: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during data cleaning: {e}")
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Applies Bag of Words (CountVectorizer) to the data."""
    try:
        # Clean the data to ensure no NaN values are passed to CountVectorizer
        train_data = clean_data(train_data)
        test_data = clean_data(test_data)

        vectorizer = CountVectorizer(max_features=max_features)

        X_train_bow = vectorizer.fit_transform(train_data['content'].values)
        X_test_bow = vectorizer.transform(test_data['content'].values)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = train_data['sentiment'].values

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = test_data['sentiment'].values

        logger.info("Successfully applied Bag of Words.")
        return train_df, test_df
    except ValueError as e:
        logger.error(f"Invalid input to CountVectorizer: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while applying BOW: {e}")
        raise

def store_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_path: str) -> None:
    """Stores the transformed data in the specified path."""
    try:
        os.makedirs(output_path, exist_ok=True)
        train_df.to_csv(os.path.join(output_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(output_path, "test_bow.csv"), index=False)
        logger.info("Successfully stored BOW features.")
    except OSError as e:
        logger.error(f"Error creating directory or writing files: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while storing data: {e}")
        raise

def main() -> None:
    try:
        # Load data
        train_data, test_data = load_data('././data/processed/train_processed.csv', '././data/processed/test_processed.csv')

        # Load parameters
        max_features = load_params('././params.yaml')

        # Apply Bag of Words
        train_df, test_df = apply_bow(train_data, test_data, max_features)

        # Store data
        store_data(train_df, test_df, '././data/interim')
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        raise

if __name__ == '__main__':
    main()