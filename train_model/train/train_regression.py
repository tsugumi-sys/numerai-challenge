import xgboost
import pandas as pd
import numpy as np

print('Loading data ...')
training_data = pd.read_csv('../../data/numerai_training_data.csv')
evaluation_data = pd.read_csv('../../data/numerai_tournament_data.csv')

features_df = pd.read_csv('../../analyse/feature-selection/feature_selection.csv', index_col=0)
features_df = features_df.loc[features_df['COUNT'] > 8]
selected_features = features_df.index.tolist()
print(selected_features)


X_train = training_data[selected_features].copy()
y_train = training_data["target"].copy()
X_valid = evaluation_data[selected_features].copy()


from xgboost import XGBRegressor
print('Model Training ...')

model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1)

model.fit(X_train, y_train)
print('Generate Prediction ...')
training_data["prediction"] = model.predict(X_train)
evaluation_data["prediction"] = model.predict(X_valid)

training_pred = pd.DataFrame({"id": training_data['id'], "prediction": model.predict(X_train), "label": training_data["target"]})
eval_pred = pd.DataFrame({"id": evaluation_data['id'], "prediction": model.predict(X_valid)})

training_pred.to_csv('training_prediction_regressoion.csv', index=False)
eval_pred.to_csv('evaluation_prediction_regression.csv', index=False)