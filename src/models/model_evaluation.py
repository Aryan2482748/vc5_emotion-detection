from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import pickle
import pandas as pd
import json

with open("models/random_forest_model.pkl","rb") as model_file:
    model=pickle.load(model_file)

test_data = pd.read_csv("data/interim/test_bow.csv")
X_test = test_data.drop(columns=['sentiment']).values
y_test = test_data['sentiment'].values  


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
# roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

# Save evaluation metrics to a JSON file for later analysis
metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    # "roc_auc": roc_auc
}

with open("reports/evaluation_metrics.json", "w") as metrics_file:
    json.dump(metrics, metrics_file, indent=4)