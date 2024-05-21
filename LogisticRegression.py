import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")

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

# Xây dựng mô hình Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(train_data[["Age", "Gender", "Height", "Weight", "MTRANS"]], train_data["BMI"] > 25)  # Phân loại dựa trên giới hạn BMI 25

# Dự đoán trên tập kiểm tra
predictions = model.predict(test_data[["Age", "Gender", "Height", "Weight", "MTRANS"]])

# Đánh giá mô hình
accuracy = accuracy_score(test_data["BMI"] > 25, predictions)
conf_matrix = confusion_matrix(test_data["BMI"] > 25, predictions)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Trang trí biểu đồ
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(16, 8))

# Biểu đồ cột cho BMI thực tế và dự đoán
plt.subplot(1, 3, 1)
sns.barplot(x=["Thực tế Béo phì", "Dự đoán Béo phì"], y=[(test_data["BMI"] > 25).mean(), predictions.mean()], palette="pastel")
plt.xlabel("Phân loại BMI", fontsize=14)
plt.ylabel("Tỉ lệ", fontsize=14)
plt.title("Biểu đồ cột phân loại\nBéo phì thực tế và dự đoán", fontsize=16)

# Phân bố BMI thực tế và dự đoán
plt.subplot(1, 3, 2)
sns.kdeplot(test_data["BMI"], label="Thực tế BMI", color="blue", fill=True)
sns.kdeplot(predictions, label="Dự đoán BMI", color="orange", fill=True)
plt.xlabel("BMI", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Phân bố BMI thực tế và dự đoán", fontsize=16)
plt.legend()

# Mối tương quan giữa BMI và tuổi
plt.subplot(1, 3, 3)
sns.scatterplot(x=test_data["Age"], y=test_data["BMI"], hue=(test_data["BMI"] > 25), palette={True: "red", False: "blue"}, s=80, alpha=0.7)
plt.xlabel("Age", fontsize=14)
plt.ylabel("BMI", fontsize=14)
plt.title(" Sự tương quan giữa BMI và\ntuổi với phân loại Béo phì", fontsize=16)
plt.suptitle("Logistic Regression Algorithm", fontsize=18)

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

# Biểu đồ boxplot cho BMI theo giới tính và phương tiện di chuyển
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x="Gender", y="BMI", data=train_data)
plt.title("Biểu đồ boxplot cho BMI theo giới tính")
plt.subplot(1, 2, 2)
sns.boxplot(x="MTRANS", y="BMI", data=train_data)
plt.title("Biểu đồ boxplot cho BMI theo phương tiện di chuyển")

# Biểu đồ phân tán 3D cho tuổi, chiều cao, cân nặng và BMI
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_data["Age"], train_data["Height"], train_data["Weight"], c=train_data["BMI"], cmap="coolwarm", s=50)
ax.set_xlabel("Age")
ax.set_ylabel("Height")
ax.set_zlabel("Weight")
ax.set_title("Biểu đồ phân tán 3D cho tuổi, chiều cao, cân nặng và BMI")

# Biểu đồ heatmap cho ma trận tương quan
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns
train_data_numerical = train_data[numerical_columns]
correlation_matrix = train_data_numerical.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Biểu đồ heatmap cho ma trận tương quan")

# Lưu kết quả phân tích
plt.savefig("bmi_analysis_with_logistic_regression.png")

plt.show()
