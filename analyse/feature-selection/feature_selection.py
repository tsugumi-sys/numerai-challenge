import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier as ABC, ExtraTreesClassifier as ETC, GradientBoostingClassifier as GBC
from sklearn.feature_selection import SelectFromModel

# Load Train Data with random state 1
def load_data(random_state=1):
    # Load & Sampling Data
    data = pd.read_csv('../../data/numerai_training_data.csv', index_col='id')
    sample_data = data.sample(frac=0.3, random_state=random_state)
    feature_names = [f for f in data.columns if "feature" in f]
    X = sample_data[feature_names].copy()
    y = sample_data["target"].copy()

    # Check NaN
    print("NaN in X: ", X.isna().sum().sum())
    print('NaN in y: ', y.isna().sum().sum())

    # Label Encoding Data
    label_encoder = LabelEncoder()
    label_encoder.fit([0, 0.25, 0.5, 0.75, 1])
    y = label_encoder.transform(y)
    for col in feature_names:
        X[col] = label_encoder.transform(X[col])


    return X, y, feature_names

# Select Feature based on mi score
# Parameter
# ========================
# X: train data
# y: label data
#
# Return
# ========================
# list of str (freature name)
def mi_based_select(X, y):
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores.loc[mi_scores > 0.01].index.tolist()

# Select Feature based on Model Selection
# Parameter
# ========================
# X: train data
# y: label data
#
# Return
# ========================
# list of str (freature name)
def model_based_select(X, y, feature_names):
    models = {
    "DecisionTreeC": DTC(max_depth=15, random_state=1, max_leaf_nodes=50),
    "RandomForestC": RFC(n_estimators=100, max_depth=15, random_state=1),
    "AdaBoostC": ABC(n_estimators=100, random_state=1),
    "ExtraTreesC": ETC(n_estimators=100, max_depth=15, random_state=1),
    "GradientBoostingC": GBC(n_estimators=50, max_depth=15, random_state=1)
    }

    # Make Feature Goups
    feature_groups = {
        g: [c for c in feature_names if c.startswith(f"feature_{g}")]
        for g in ["intelligence", "wisdom", "charisma", "dexterity", "strength", "constitution"]
    }

    group_names = list(feature_groups.keys())
    res_features = []
    for key in group_names:
        print('-'*40, key, '-'*40)
        features = feature_groups[key]
        feature_counts = dict.fromkeys(features, 0)
        f_names = np.array(features)
        for model_key in models.keys():
            m = models[model_key].fit(X[features], y)
            selector = SelectFromModel(estimator=m, prefit=True)
            selected_features = f_names[selector.get_support()]
            for f in selected_features:
                feature_counts[f] += 1
        df = pd.DataFrame({"COUNT": feature_counts.values()}, index=feature_counts.keys())
        df = df.sort_values(by="COUNT", ascending=False)
        if key in ["strength", "constitution"]:
            threshold = 5
        else:
            threshold = 4
        res_features += df.loc[df["COUNT"] >= threshold].index.tolist()
    return res_features


def feature_select():
    _, _, feature_names = load_data()
    feature_counts = dict.fromkeys(feature_names, 0)
    for i in range(1, 11):
        random_state = i
        X, y, feature_names = load_data(random_state)
        mi_based_result = mi_based_select(X, y)
        model_based_result = model_based_select(X, y, feature_names)
        res_list = mi_based_result + model_based_result
        res_list = list(dict.fromkeys(res_list))
        for f in res_list:
            feature_counts[f] += 1
    print(feature_counts)
    df = pd.DataFrame({"COUNT": feature_counts.values()}, index=feature_counts.keys())
    df = df.sort_values(by="COUNT", ascending=False)
    df.to_csv('feature_selection.csv')

    return feature_counts

if __name__ == '__main__':
    feature_select()