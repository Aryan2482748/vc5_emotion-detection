import numpy as np
import pandas as pd
import yaml
import os
import logging
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

pd.set_option('future.no_silent_downcasting', True)

def load_params(params_path: str) -> Dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def download_data(url: str) -> pd.DataFrame:
    """Download dataset from a URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Data downloaded from {url}")
        return df
    except Exception as e:
        logging.error(f"Failed to download data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame: drop columns, filter, and encode labels."""
    try:
        df = df.drop(columns=['tweet_id'])
        logging.info("Dropped 'tweet_id' column")
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        logging.info("Filtered for 'happiness' and 'sadness' sentiments")
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info("Encoded sentiments: 'happiness'->1, 'sadness'->0")
        return final_df
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split into train and test sets with test_size={test_size}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
    """Save train and test DataFrames to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        logging.info(f"Train and test data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        df = download_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = split_data(final_df, test_size)
        save_data(train_data, test_data, 'data/raw')
        logging.info("Data ingestion pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Data ingestion pipeline failed: {e}")

if __name__ == "__main__":
    main()