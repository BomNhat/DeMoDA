from datetime import date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("Dataset.csv")
test = pd.read_csv("Dataset.csv")
data = pd.read_csv("Dataset.csv")
print(train.head(5).T)
train, test = train_test_split(data, test_size=1/3, random_state=42)

target = train["NObeyesdad"]
train = train.drop("NObeyesdad", axis="columns")

target.value_counts()
print(target.value_counts())

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sorted_labels):
        self.sorted_labels = sorted_labels  # Thêm dòng này để gán giá trị cho thuộc tính sorted_labels
        self.label_to_numeric_mapping = None

    def fit(self, y):
        self.label_to_numeric_mapping = {label: numeric for numeric, label in enumerate(self.sorted_labels)}
        return self

    def transform(self, y):
        if self.label_to_numeric_mapping is None:
            raise ValueError("fit method must be called before transform")
        return y.map(self.label_to_numeric_mapping)

    def inverse_transform(self, y):
        if self.label_to_numeric_mapping is None:
            raise ValueError("fit method must be called before inverse_transform")
        return pd.Series(y).map({numeric: label for label, numeric in self.label_to_numeric_mapping.items()})

color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0B2447"]
cmap_custom = ListedColormap(color_list)

sorted_labels = ['Insufficient_Weight', 'Normal_Weight', 
 'Overweight_Level_I', 'Overweight_Level_II', 
 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

plt.figure(figsize=(8, 4))
ax = sns.countplot(x=target, order=sorted_labels, palette=color_list)

plt.title('Distribution of Obesity Risk Levels')
plt.xlabel('Obesity Risk Levels')
plt.ylabel('Count')

ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha='right', fontsize=8)
plt.tight_layout() 
plt.show()
target_encoder = CustomLabelEncoder(sorted_labels)

target_encoder.fit(target)
target_numeric = target_encoder.transform(target)
target_numeric
target_encoder.inverse_transform(target_numeric)

train.isnull().sum()
print(train.isnull().sum())
train.dtypes
print(train.dtypes)
categorical_features = train.columns[train.dtypes=="object"].tolist()
numeric_features = train.columns[train.dtypes!="object"].tolist()
train[categorical_features].nunique()
train[numeric_features].nunique()
print(train[numeric_features].nunique())


categorical_features = train.columns[train.dtypes == "object"].tolist()
numeric_features = train.columns[train.dtypes != "object"].tolist()
train[categorical_features].nunique()
train[numeric_features].nunique()
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0B2447"]   

def plot_count_pairs(train, test, feature, hue="set", order=None, palette=None):
    data_df = train.copy()
    data_df['set'] = 'train'
    data_df = pd.concat([data_df, test.copy()]).fillna('test')
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.countplot(x=feature, data=data_df, hue=hue, palette=color_list, order=order)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title(f"Paired train/test frequencies of {feature}")
    plt.show()
    
for feature in categorical_features:
    if feature in ["CAEC", "CALC"]:
        order = ["no", "Always", "Sometimes", "Frequently"]
    else:
        order = sorted(train[feature].unique())
    plot_count_pairs(train, test, feature=feature, order=order, palette=color_list)

plt.show()



import warnings
def plot_distribution_pairs(train, test, feature, hue="set", palette=None):
    data_df = train.copy()
    data_df['set'] = 'train'
    data_df = pd.concat([data_df, test.copy()]).fillna('test')
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, s in enumerate(data_df[hue].unique()):
        selection = data_df.loc[data_df[hue]==s, feature]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            g = sns.histplot(selection, color=palette[i], ax=ax, label=s)
    ax.set_title(f"Paired train/test distributions of {feature}")
    g.legend()
    plt.show()

def plot_distribution_pairs_boxplot(train, test, feature, hue="set", palette=None):
    data_df = train.copy()
    data_df['set'] = 'train'
    data_df = pd.concat([data_df, test.copy()]).fillna('test')
    data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, s in enumerate(data_df[hue].unique()):
        selection = data_df.loc[data_df[hue]==s, feature]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            g = sns.boxplot(x=hue, y=feature, data=data_df, palette=palette, ax=ax)
    ax.set_title(f"Paired train/test boxplots of {feature}")
    g.legend()
    plt.show()
    
for feature in numeric_features:
    plot_distribution_pairs(train, test, feature, palette=color_list)
    plot_distribution_pairs_boxplot(train, test, feature, palette=color_list)




encoder = OneHotEncoder(sparse_output=False)
encoder.fit(pd.concat([train[categorical_features], test[categorical_features]], axis=0))

train_encoded = encoder.fit_transform(train[categorical_features])
train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_features))
test_encoded = encoder.transform(test[categorical_features])
test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_features))
train_sum = train_encoded_df.sum(axis=0).astype(int)
test_sum = test_encoded_df.sum(axis=0).astype(int)

sum_df = pd.concat([train_sum, test_sum], axis=1, keys=['Train', 'Test'])
print(sum_df)


combine_columns = ['CALC_Always', 'CALC_Frequently']

train_encoded_df['CALC_Always|Frequently'] = train_encoded_df[combine_columns].sum(axis=1)
test_encoded_df['CALC_Always|Frequently'] = test_encoded_df[combine_columns].sum(axis=1)

train_encoded_df = train_encoded_df.drop(columns=combine_columns).set_index(train.index)
test_encoded_df = test_encoded_df.drop(columns=combine_columns).set_index(test.index)
levels = {"Always": 3, "Frequently": 2, "Sometimes": 1, "no": 0}
train["CALC_ord"] = train["CALC"].map(levels)
test["CALC_ord"] = test["CALC"].map(levels)
train["CAEC_ord"] = train["CAEC"].map(levels)
test["CAEC_ord"] = test["CAEC"].map(levels)
train = pd.concat([train.drop(categorical_features, axis=1), train_encoded_df], axis=1)
test = pd.concat([test.drop(categorical_features, axis=1), test_encoded_df], axis=1)


X =  pd.concat([train, test], axis=0)
y = [0] * len(train) + [1] * len(test)
X_train = train
y_train = target_encoder.transform(train["NObeyesdad"])
X_test = test


model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
cv_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1, method='predict_proba')

score = roc_auc_score(y_true=y, y_score=cv_preds[:,1])
print(f"roc-auc score: {score:0.3f}")

train['BMI'] = train['Weight'] / (train['Height'] ** 2)
test['BMI'] = test['Weight'] / (test['Height'] ** 2)
plot_distribution_pairs(train, test, feature="BMI", palette=color_list)
5