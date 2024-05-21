import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")
# data = pd.read_csv("bmi_entry.csv")

# Vệ sinh dữ liệus
data = data.dropna()

# Phân tích dữ liệu
data["BMI"] = data["Weight"] / (data["Height"] ** 2)


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(data, test_size=1/3, random_state=42)

# Xây dựng mô hình Linear Regression
model = LinearRegression()
model.fit(train_data[["Age", "Height", "Weight"]], train_data["BMI"])

# Dự đoán trên tập kiểm tra
predictions = model.predict(test_data[["Age", "Height", "Weight"]])

# Đánh giá mô hình
mse = mean_squared_error(test_data["BMI"], predictions)
print(f"Mean Squared Error: {mse}")

# Trang trí biểu đồ
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 6))

# Phân bố BMI thực tế và dự đoán
plt.subplot(1, 2, 1)
sns.kdeplot(test_data["BMI"], label="Actual BMI", color="blue", fill=True)
sns.kdeplot(predictions, label="Predicted BMI", color="orange", fill=True)
plt.xlabel("BMI", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Distribution of Actual and Predicted BMI", fontsize=16)
plt.legend()

# Mối tương quan giữa BMI và tuổi
plt.subplot(1, 2, 2)
sns.scatterplot(x=test_data["Age"], y=test_data["BMI"], label="Actual BMI", color="blue", s=80, alpha=0.7)
sns.scatterplot(x=test_data["Age"], y=predictions, label="Predicted BMI", color="orange", s=80, alpha=0.7)
plt.xlabel("Age", fontsize=14)
plt.ylabel("BMI", fontsize=14)
plt.title("Correlation between Age and BMI", fontsize=16)
plt.legend()

plt.suptitle("Linear Regression Algorithm", fontsize=18)

# Lưu kết quả phân tích
plt.savefig("bmi_analysis_with_model_xinsosodep.png")

plt.show()