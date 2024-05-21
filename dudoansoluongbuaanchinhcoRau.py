import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")

# Loại bỏ các dòng có giá trị thiếu
data = data.dropna()

# Phân chia dữ liệu thành các tập huấn luyện và tập kiểm tra
X = data[['FCVC', 'NCP']]  # Đặc trưng: FCVC và NCP
y = data['Meal_habits']  # Nhãn: Meal_habits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Dự đoán Meal_habits trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Trực quan hóa
plt.figure(figsize=(10, 6))
sns.histplot(data['FCVC'], bins=20, kde=False, color='skyblue')
plt.xlabel('FCVC (Frequency of consumption of vegetables)')
plt.ylabel('Số lượng ID')
plt.title('Phân phối của FCVC')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['FCVC'], y=X_test['NCP'], hue=y_pred, palette='Set2')
plt.xlabel('FCVC (Frequency of consumption of vegetables)')
plt.ylabel('NCP (Number of main meals)')
plt.title('Meal Habits Prediction')
plt.legend(title='Meal_habits')
plt.savefig("DuDoanThoiQuenAnUong.png")
plt.show()
