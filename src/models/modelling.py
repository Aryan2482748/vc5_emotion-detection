import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from typing import Any, Dict
import yaml
import os
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def load_params(params_path: str) -> Dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Parameters loaded from {params_path}")
        return params
    except Exception as e:
        logging.error(f"Failed to load parameters: {e}")
        raise

def load_train_data(train_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from CSV file."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.drop(columns=['sentiment']).values
        y_train = train_data['sentiment'].values
        logging.info(f"Training data loaded from {train_path}")
        return X_train, y_train
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        logging.info("RandomForestClassifier model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the trained model to disk using pickle."""
    try:
        with open(model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model training and saving."""
    try:
        params = load_params("params.yaml")
        n_estimators = params['modelling']['n_estimators']
        max_depth = params['modelling']['max_depth']
        X_train, y_train = load_train_data("data/interim/train_tfidf.csv")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model training pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model training pipeline failed: {e}")

if __name__ == "__main__":
    main()