import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv("Dataset.csv")

# Vệ sinh dữ liệu
data = data.dropna()

# Phân tích dữ liệu
data["BMI"] = data["Weight"] / (data["Height"] ** 2)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_data, test_data = train_test_split(data, test_size=2/3, random_state=42)

# Sử dụng LabelEncoder cho Gender
label_encoder = LabelEncoder()
train_data["Gender"] = label_encoder.fit_transform(train_data["Gender"])
test_data["Gender"] = label_encoder.transform(test_data["Gender"])

# Xây dựng mô hình Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(train_data[["Age", "Height", "Weight", "Gender"]], train_data["BMI"] > 25)  

# Dự đoán trên tập kiểm tra
predictions = model.predict(test_data[["Age", "Height", "Weight", "Gender"]])

# Tạo DataFrame mới từ dự đoán và giới tính thực tế
results = pd.DataFrame({'Predictions': predictions, 'Gender': test_data['Gender']})

# Đếm số lượng trường hợp béo phì dự đoán cho nam và nữ
obesity_count_male = results[(results['Predictions'] == True) & (results['Gender'] == 1)].shape[0]
obesity_count_female = results[(results['Predictions'] == True) & (results['Gender'] == 0)].shape[0]

print(f"Số lượng trường hợp béo phì dự đoán cho nam: {obesity_count_male}")
print(f"Số lượng trường hợp béo phì dự đoán cho nữ: {obesity_count_female}")

# So sánh số lượng trường hợp béo phì dự đoán cho nam và nữ
if obesity_count_male > obesity_count_female:
    print("Số lượng trường hợp béo phì dự đoán nhiều hơn ở nam.")
elif obesity_count_male < obesity_count_female:
    print("Số lượng trường hợp béo phì dự đoán nhiều hơn ở nữ.")
else:
    print("Số lượng trường hợp béo phì dự đoán bằng nhau giữa nam và nữ.")

# Vẽ biểu đồ tròn
labels = ['Nam', 'Nữ']
values = [obesity_count_male, obesity_count_female]

plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
plt.title('Dự đoán bệnh Béo Phì xảy ra ở  giới tính',fontsize=10)
plt.suptitle("Logistic Regression Algorithm", fontsize=18)
plt.savefig("DựđoánbệnhBéoPhìxảyraởgiớitính.png")
plt.show()
