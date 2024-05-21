import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
data = pd.read_csv("Dataset2.csv")

# Vệ sinh dữ liệu
data = data.dropna()

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(data, test_size=0.1, random_state=2)

# Tạo một đối tượng LabelEncoder
label_encoder = LabelEncoder()

# Chuyển đổi thuộc tính "Gender" từ chuỗi thành số
train_data["Gender"] = label_encoder.fit_transform(train_data["Gender"])
test_data["Gender"] = label_encoder.transform(test_data["Gender"])
train_data["MTRANS"] = label_encoder.fit_transform(train_data["MTRANS"])
test_data["MTRANS"] = label_encoder.transform(test_data["MTRANS"])

# Xây dựng mô hình Logistic Regression
model = LogisticRegression()
model.fit(train_data[["Gender", "Height", "Weight", "MTRANS", "FAF"]], train_data["NObeyesdad"])  # Phân loại dựa trên cột "NObeyesdad" - nguyên nhân gây ra béo phì

# Dự đoán trên tập kiểm tra
predictions = model.predict(test_data[["Gender", "Height", "Weight", "MTRANS", "FAF"]])

# Đánh giá mô hình
accuracy = accuracy_score(test_data["NObeyesdad"], predictions)
conf_matrix = confusion_matrix(test_data["NObeyesdad"], predictions)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
