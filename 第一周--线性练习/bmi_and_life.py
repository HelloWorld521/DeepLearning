import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# 读取数据
bmi_life_data = pd.read_csv('bmi_and_life.csv')
x_values = bmi_life_data[['BMI']]
y_values = bmi_life_data[['Life expectancy']]

# 训练模型
bmi_life_model = linear_model.LinearRegression()   #创建回归模型
bmi_life_model.fit(x_values, y_values)

# 作图
plt.scatter(x_values, y_values)  #绘制散点图
plt.plot(x_values, bmi_life_model.predict(x_values)) #绘制回归线
plt.show()

print(bmi_life_model.predict(21))


