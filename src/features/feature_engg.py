import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Tuple
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path).dropna(subset=["content"])
        test_data = pd.read_csv(test_path).dropna(subset=["content"])
        logging.info("Loaded processed train and test data.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def vectorize_text(X_train: np.ndarray, X_test: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Vectorize text data using Bag-of-Words."""
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        logging.info("Text data vectorized using Bag-of-Words.")
        return X_train_bow, X_test_bow, vectorizer
    except Exception as e:
        logging.error(f"Vectorization failed: {e}")
        raise

def save_features(X_train_bow: np.ndarray, y_train: np.ndarray, X_test_bow: np.ndarray, y_test: np.ndarray, output_dir: str) -> None:
    """Save feature-engineered train and test data to CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['sentiment'] = y_train
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['sentiment'] = y_test
        train_df.to_csv(os.path.join(output_dir, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_bow.csv"), index=False)
        logging.info(f"Feature data saved to {output_dir}")
    except Exception as e:
        logging.error(f"Failed to save feature data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate feature engineering."""
    try:
        params = load_params("params.yaml")
        max_features = params['feature_engg']['max_features']
        train_data, test_data = load_data("data/processed/train.csv", "data/processed/test.csv")
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        X_train_bow, X_test_bow, _ = vectorize_text(X_train, X_test, max_features)
        save_features(X_train_bow, y_train, X_test_bow, y_test, "data/interim")
        logging.info("Feature engineering pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Feature engineering pipeline failed: {e}")

if __name__ == "__main__":
    main()