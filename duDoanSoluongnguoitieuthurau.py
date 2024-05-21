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
