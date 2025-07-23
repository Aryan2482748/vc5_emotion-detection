import logging
import pickle
import pandas as pd
import json
import os
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score  # , roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def load_model(model_path: str) -> Any:
    """Load a trained model from disk."""
    try:
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def load_test_data(test_path: str) -> pd.DataFrame:
    """Load test data from CSV file."""
    try:
        test_data = pd.read_csv(test_path)
        logging.info(f"Test data loaded from {test_path}")
        return test_data
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        raise

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate the model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        # roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            # "roc_auc": roc_auc
        }
        logging.info("Model evaluation completed.")
        return metrics
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")
        raise

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        logging.info(f"Evaluation metrics saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save evaluation metrics: {e}")
        raise

def main() -> None:
    """Main function to orchestrate model evaluation."""
    try:
        model = load_model("models/random_forest_model.pkl")
        test_data = load_test_data("data/interim/test_tfidf.csv")
        X_test = test_data.drop(columns=['sentiment']).values
        y_test = test_data['sentiment'].values
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/evaluation_metrics.json")
        logging.info("Model evaluation pipeline completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation pipeline failed: {e}")

if __name__ == "__main__":
    main()