import xgboost
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print('Loading data ...')
training_data = pd.read_csv('../../data/numerai_training_data.csv')
evaluation_data = pd.read_csv('../../data/numerai_tournament_data.csv')

features_df = pd.read_csv('../../analyse/feature-selection/feature_selection.csv', index_col=0)
features_df = features_df.loc[features_df['COUNT'] > 8]
selected_features = features_df.index.tolist()
print(selected_features)

# Label Encoding
print('Label encoding data ...')
label_encoder = LabelEncoder()
label_encoder.fit([0, 0.25, 0.5, 0.75, 1])
X_train = training_data[selected_features].copy()
y_train = training_data["target"].copy()
X_valid = evaluation_data[selected_features].copy()

y_train = label_encoder.transform(y_train)
for f in selected_features:
    X_train[f] = label_encoder.transform(X_train[f])
    X_valid[f] = label_encoder.transform(X_valid[f])

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier as RFC
print('Model Training ...')
model = XGBClassifier(max_depth=10, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1, use_label_encoder=False)
#model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1)
#model = RFC(n_estimators=100, max_depth=30, random_state=1, n_jobs=10)
model.fit(X_train, y_train)
print('Generate Prediction ...')
training_data["prediction"] = model.predict(X_train)
evaluation_data["prediction"] = model.predict(X_valid)

training_pred = pd.DataFrame({"id": training_data['id'], "prediction": label_encoder.inverse_transform(model.predict(X_train)), "label": training_data["target"]})
eval_pred = pd.DataFrame({"id": evaluation_data['id'], "prediction": label_encoder.inverse_transform(model.predict(X_valid))})

training_pred.to_csv('training_prediction_classif.csv', index=False)
eval_pred.to_csv('evaluation_prediction_classif.csv', index=False)