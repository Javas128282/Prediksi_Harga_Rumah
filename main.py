import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file = "perumahan.csv"
df = pd.read_csv(file, usecols=["price","bedrooms", "bathrooms", "sqft_living","grade", "yr_built"])
#plt.plot(df["bedrooms"], df["bathrooms"])
#plt.show()
#print(df.head())#head dari datasets
#print(df.shape)#info baris dan kolom
#print(df.info())
#print(df.describe())#statistik 
#print(df.isnull().sum())#cek missing value
x = df.drop(columns=["price"])#input/fitur
y = df["price"]#output/target yaitu menebak prices
'''singkatnya , x adalah data yang diketahui dan y adalah data yang ingin kita 
prediksi'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
line_reg = LinearRegression()
line_reg.fit(x_train, y_train) #melatih model supaya belajar hubungan antara x_train (fitur) dan y_train (target).
y_pred = line_reg.predict(x_test)# melatih model supaya belajar hubungan antara x_train (fitur) dan y_train (target).
score = line_reg.score(x_test, y_test)#.score untuk mengukur akurasi model dengan metode R²
print("Akurasi model: ", score)

#Memprediksi Kasus:
kasus1 = line_reg.predict([[3, 2, 2000, 7, 2015]])#3 bedrooms, 2 bathrooms, 2000 sqft_living, grade 7, built in 2015
print("Harga Rumah : ", kasus1)
'''contoh output:
Harga Rumah [297789.71962617] >> artinya 297.789-Ribu Rupiah
'''
