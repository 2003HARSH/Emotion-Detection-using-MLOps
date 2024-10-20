import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logger = logging.getLogger('text_preprocessing')
logger.setLevel('DEBUG')

file_handler = logging.FileHandler('./logs/preprocessing.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Download necessary NLTK resources
try:
    nltk.download('wordnet')
    nltk.download('stopwords')
    logger.info("Successfully downloaded required NLTK packages.")
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {e}")
    raise

# Load raw data
def load_data(train_path: str, test_path: str) -> tuple:
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        logger.info("Train and test data loaded successfully.")
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

# Define text transformation functions with exception handling
def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        raise

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        text = [word for word in text.split() if word not in stop_words]
        return " ".join(text)
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        raise

def removing_numbers(text: str) -> str:
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        raise

def lower_case(text: str) -> str:
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logger.error(f"Error converting to lower case: {e}")
        raise

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        raise

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.loc[i, 'text'].split()) < 3:
                df.loc[i, 'text'] = np.nan
        logger.info("Small sentences removed successfully.")
    except Exception as e:
        logger.error(f"Error removing small sentences: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.info("Text normalization completed successfully.")
        return df
    except KeyError as e:
        logger.error(f"Missing column in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in text normalization: {e}")
        raise

# Store processed data
def store_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_path: str) -> None:
    try:
        os.makedirs(output_path, exist_ok=True)
        train_data.to_csv(os.path.join(output_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(output_path, "test_processed.csv"), index=False)
        logger.info("Processed data stored successfully.")
    except OSError as e:
        logger.error(f"Error creating directory or writing files: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in store_data: {e}")
        raise

# Main function to execute the workflow
def main() -> None:
    try:
        train_data, test_data = load_data('./data/raw/train.csv', './data/raw/test.csv')
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)
        store_data(train_processed, test_processed, './data/processed')
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        raise

if __name__ == '__main__':
    main()
