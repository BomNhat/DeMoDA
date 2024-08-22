import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression


# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")

# Loại bỏ các dòng có giá trị thiếu
data = data.dropna()

# Phân chia dữ liệu thành các tập huấn luyện và tập kiểm tra
X = data[['FCVC', 'NCP']]  # Đặc trưng: FCVC và NCP
y = data['Meal_habits']  # Nhãn: Meal_habits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Dự đoán Meal_habits trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# Trực quan hóa
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test['FCVC'], y=X_test['NCP'], hue=y_pred, palette='Set2')
plt.xlabel('FCVC (Frequency of consumption of vegetables)')
plt.ylabel('NCP (Number of main meals)')
plt.title('Meal Habits Prediction')
plt.legend(title='Meal_habits')
plt.savefig("DuDoanThoiQuenAnUong.png")
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")

# Loại bỏ các dòng có giá trị thiếu
data = data.dropna()

# Trực quan hóa số lượng ID có FCVC
plt.figure(figsize=(10, 6))
sns.histplot(data['FCVC'], bins=20, kde=False, color='skyblue')
plt.xlabel('FCVC (Frequency of consumption of vegetables)')
plt.ylabel('Số lượng ID')
plt.title('Số lượng người Tiêu thụ Rau ở các bữa ăn chính. ')
plt.show()


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



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset.csv")

data = data.dropna()


data["BMI"] = data["Weight"] / (data["Height"] ** 2)


train_data, test_data = train_test_split(data, test_size=2/3, random_state=42)


label_encoder = LabelEncoder()
train_data["Gender"] = label_encoder.fit_transform(train_data["Gender"])
test_data["Gender"] = label_encoder.transform(test_data["Gender"])


model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(train_data[["Age", "Height", "Weight", "Gender"]], train_data["BMI"] > 25)  


predictions_dt = model_dt.predict(test_data[["Age", "Height", "Weight", "Gender"]])


results_dt = pd.DataFrame({'Predictions': predictions_dt, 'Gender': test_data['Gender']})

obesity_count_male_dt = results_dt[(results_dt['Predictions'] == True) & (results_dt['Gender'] == 1)].shape[0]
obesity_count_female_dt = results_dt[(results_dt['Predictions'] == True) & (results_dt['Gender'] == 0)].shape[0]

print(f"Số lượng trường hợp béo phì dự đoán cho nam (Decision Tree): {obesity_count_male_dt}")
print(f"Số lượng trường hợp béo phì dự đoán cho nữ (Decision Tree): {obesity_count_female_dt}")

labels_dt = ['Nam', 'Nữ']
values_dt = [obesity_count_male_dt, obesity_count_female_dt]

plt.pie(values_dt, labels=labels_dt, autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
plt.title('Dự đoán bệnh Béo Phì xảy ra ở giới tính (Decision Tree)')
plt.savefig("Dự đoán bệnh Béo Phì xảy ra ở  giới tính.png")
plt.show()
