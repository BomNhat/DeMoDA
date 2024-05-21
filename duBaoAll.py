import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Định nghĩa hàm thêm đặc trưng và tính toán Physical_Activity_Level
def calculate_physical_activity_level(data):
    if 'FAF' in data.columns and 'TUE' in data.columns:
        data['Physical_Activity_Level'] = data['FAF'] - data['TUE']
        return data
    else:
        print("Columns 'FAF' and/or 'TUE' not found in the dataset.")
        return None

# Định nghĩa hàm trực quan hóa phân phối của đặc trưng mới
def plot_distribution_pairs(train, test, feature, palette):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=train, x=feature, color=palette[0], label='Train', kde=True)
    sns.histplot(data=test, x=feature, color=palette[1], label='Test', kde=True)
    plt.title(f'Distribution of {feature} in Train and Test Data', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig("Physical_Activity_Level.png")
    plt.legend()
    plt.show()

# Đọc dữ liệu
train = pd.read_csv("Dataset.csv")
test = pd.read_csv("Dataset.csv")

# Thêm đặc trưng và tính toán Physical_Activity_Level
train = calculate_physical_activity_level(train)
test = calculate_physical_activity_level(test)

# Trực quan hóa phân phối của đặc trưng mới
plot_distribution_pairs(train, test, feature="Physical_Activity_Level", palette=["blue", "orange"])
