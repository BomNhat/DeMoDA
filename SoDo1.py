import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns



train = pd.read_csv("Dataset.csv")
test = pd.read_csv("Dataset.csv")
#column_names = train.columns
num_rows, num_columns = train.shape
print(f"Số lượng cột trong DataFrame là: {num_columns}")
#print(test.head(5))
target = train["NObeyesdad"]
train = train.drop("NObeyesdad", axis="columns")
#print(target.value_counts())
print(target.value_counts() / len(target))

color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0B2447"]
cmap_custom = ListedColormap(color_list)

sorted_labels = ['Insufficient_Weight', 'Normal_Weight', 
 'Overweight_Level_I', 'Overweight_Level_II', 
 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
plt.figure(figsize=(8, 4))
ax = sns.countplot(x=target, order=sorted_labels, palette=color_list)
plt.title("Bảng Phân Chia Số Lượng Mức Độ Béo Phì\n Năm 2023")
plt.xlabel("Các Mức Độ Béo Phì")
plt.ylabel("Số Lượng")

ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
plt.tight_layout()
plt.show()


