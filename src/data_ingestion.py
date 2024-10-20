import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

# Configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('./logs/ingestion.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load parameters from params.yaml
def load_params(path: str) -> float:
    try:
        test_size = yaml.safe_load(open(path, 'r'))['data_ingestion']['test_size']
        logger.info('Test size retrieved successfully.')
        return test_size
    except FileNotFoundError:
        logger.error(f'The file {path} was not found.')
        raise
    except yaml.YAMLError:
        logger.error(f'Error parsing YAML file {path}.')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occurred in load_params: {e}')
        raise

# Download raw data
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        logger.info('Data read successfully from the provided URL.')
        return df
    except pd.errors.ParserError:
        logger.error(f'Error parsing the CSV file from {url}.')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occurred in read_data: {e}')
        raise

# Apply basic transformations
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.info('Data processed successfully.')
        return final_df
    except KeyError as e:
        logger.error(f'Missing column: {e}')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occurred in process_data: {e}')
        raise

# Split data into train and test sets
def train_test(final_df: pd.DataFrame, test_size: float) -> tuple:
    try:
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        logger.info('Data split into train and test sets successfully.')
        return train_data, test_data
    except ValueError as e:
        logger.error(f'Invalid test_size value: {e}')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occurred in train_test: {e}')
        raise

# Store data in data/raw directory
def store_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        data_path = os.path.join("data", "raw")
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info('Data stored successfully.')
    except OSError as e:
        logger.error(f'Error creating directory or writing files: {e}')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occurred in store_data: {e}')
        raise

def main() -> None:
    try:
        test_size = load_params('./params.yaml')
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = train_test(final_df, test_size)
        store_data(train_data, test_data)
    except Exception as e:
        logger.error(f'An unexpected error occurred in main: {e}')
        raise

if __name__ == '__main__':
    main()
