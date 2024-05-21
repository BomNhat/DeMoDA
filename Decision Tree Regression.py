import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")
data = pd.read_csv("bmi_entry.csv")

# Vệ sinh dữ liệu
data = data.dropna()

# Phân tích dữ liệu
data["BMI"] = data["Weight"] / (data["Height"] ** 2)


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(data, test_size=2/3, random_state=42)


# Tạo một đối tượng LabelEncoder
label_encoder = LabelEncoder()

# Chuyển đổi thuộc tính "Gender" từ chuỗi thành số
train_data["Gender"] = label_encoder.fit_transform(train_data["Gender"])
test_data["Gender"] = label_encoder.transform(test_data["Gender"])
train_data["MTRANS"] = label_encoder.fit_transform(train_data["MTRANS"])
test_data["MTRANS"] = label_encoder.transform(test_data["MTRANS"])

# Xây dựng mô hình Decision Tree Regression
model = DecisionTreeRegressor(random_state=42)
model.fit(train_data[["Age","Gender", "Height", "Weight","MTRANS"]], train_data["BMI"])

# Dự đoán trên tập kiểm tra
predictions = model.predict(test_data[["Age","Gender", "Height", "Weight","MTRANS"]])

# Đánh giá mô hình
mse = mean_squared_error(test_data["BMI"], predictions)
print(f"Mean Squared Error: {mse}")

# Trang trí biểu đồ
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(16, 8))

# Biểu đồ cột cho BMI thực tế và dự đoán
plt.subplot(1, 3, 1)
sns.barplot(x=["thực tế BMI", "dự đoán BMI"], y=[test_data["BMI"].mean(), predictions.mean()], palette="pastel")
plt.xlabel("BMI Type", fontsize=14)
plt.ylabel("Average BMI", fontsize=14)
plt.title("Biểu đồ cột BMI thực tế và dự đoán", fontsize=16)

# Phân bố BMI thực tế và dự đoán
plt.subplot(1, 3, 2)
sns.kdeplot(test_data["BMI"], label="thực tế BMI", color="blue", fill=True)
sns.kdeplot(predictions, label="dự đoán BMI", color="orange", fill=True)
plt.xlabel("BMI", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Phân bố BMI thực tế và dự đoán", fontsize=16)
plt.legend()

# Mối tương quan giữa BMI và tuổi
plt.subplot(1, 3, 3)
sns.scatterplot(x=test_data["Age"], y=test_data["BMI"], label="thực tế BMI", color="blue", s=80, alpha=0.7)
sns.scatterplot(x=test_data["Age"], y=predictions, label="dự đoán BMI", color="orange", s=80, alpha=0.7)
plt.xlabel("Age", fontsize=14)
plt.ylabel("BMI", fontsize=14)
plt.title("Mối tương quan giữa BMI và tuổi", fontsize=16)
plt.legend()

plt.suptitle("DecisionTree Regression Algorithm", fontsize=18)

# Lưu kết quả phân tích
plt.savefig("bmi_analysis_with_decision_tree.png")

plt.show()
