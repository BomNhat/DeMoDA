import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

train = pd.read_csv("Dataset.csv")
test = pd.read_csv("Dataset2.csv")
#column_names = train.columns
num_rows, num_columns = train.shape
print(f"Số lượng cột trong DataFrame là: {num_columns}")

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, sorted_labels):
        self.classes_ = sorted_labels
        self.label_to_numeric_mapping = None

    def fit(self, y):
        self.label_to_numeric_mapping = {label: numeric for numeric, label in enumerate(self.classes_)}
        return self

    def transform(self, y):
        if self.label_to_numeric_mapping is None:
            raise ValueError("fit method must be called before transform")
        return y.map(self.label_to_numeric_mapping)

    def inverse_transform(self, y):
        if self.label_to_numeric_mapping is None:
            raise ValueError("fit method must be called before inverse_transform")
        return pd.Series(y).map({numeric: label for label, numeric in self.label_to_numeric_mapping.items()})

# Đổi tên các nhãn của mức độ béo phì thành tiếng Việt
sorted_labels = ['Thiếu Cân - Insufficient_Weight', 'Cân Nặng Bình Thường - Normal_Weight', 
                 'Thừa Cân Mức 1 - Overweight_Level_I', 'Thừa Cân Mức 2 - Overweight_Level_II', 
                 'Béo Phì Mức 1 - Obesity_Type_I', 'Béo Phì Mức 2 - Obesity_Type_II', 'Béo Phì Mức 3 - Obesity_Type_III']

target_encoder = CustomLabelEncoder(sorted_labels)
target_encoder.fit(train["NObeyesdad"])
target_numeric = target_encoder.transform(train["NObeyesdad"])

train.isnull().sum()
train.dtypes

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

# Gọi hàm plot_count_pairs cho từng feature trong categorical_features
for feature in categorical_features:
    if feature in ["CAEC", "CALC"]:
        order = ["no", "Always", "Sometimes", "Frequently"]
    else:
        order = sorted(train[feature].unique())
    plot_count_pairs(train, test, feature=feature, order=order, palette=color_list)


    

plt.show()
