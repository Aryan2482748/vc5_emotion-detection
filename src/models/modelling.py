import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

train_data=pd.read_csv("data/interim/train_bow.csv")
X_train=train_data.drop(columns=['sentiment']).values
y_train=train_data['sentiment'].values

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
# Save the trained model to disk using pickle
with open("models/random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
