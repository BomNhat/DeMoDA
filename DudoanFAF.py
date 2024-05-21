import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ tệp csv
data = pd.read_csv("Dataset.csv")

# Chia nhóm tuổi thành các nhóm
def age_group(age):
    if age < 18:
        return 'Under 18'
    elif 18 <= age < 30:
        return '18-29'
    elif 30 <= age < 40:
        return '30-39'
    elif 40 <= age < 50:
        return '40-49'
    elif 50 <= age < 60:
        return '50-59'
    else:
        return '60+'

data['Age_Group'] = data['Age'].apply(age_group)

# Chọn các đặc trưng và nhãn
X = data[[ 'MTRANS', 'FAF', 'NCP', 'CH2O']]
y = data['Age_Group']  # Chúng ta sẽ dự đoán nhóm tuổi

# Chuyển đổi biến phân loại thành biến giả (dummy variables)
X = pd.get_dummies(X)

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy logistic
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Trực quan hóa kết quả dự đoán và thực tế
plt.figure(figsize=(10, 6))
sns.countplot(x=y_test, color='blue', alpha=0.7, label='Thực tế')
sns.countplot(x=y_pred, color='orange', alpha=0.7, label='Dự đoán')
plt.xlabel('Nhóm Tuổi')
plt.ylabel('Số lượng')
plt.title('Dự đoán các nhóm tuổi dễ mắc bệnh béo phì\nDựa trên tiêu chí MTRANS, FAF, NCP, CH2O')

plt.show()
