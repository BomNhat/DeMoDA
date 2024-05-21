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
