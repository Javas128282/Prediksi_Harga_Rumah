import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Baca data
file = "perumahan.csv"
df = pd.read_csv(file, usecols=["price","bedrooms", "bathrooms", "sqft_living","grade", "yr_built"])

# Pisah fitur dan target
x = df.drop(columns=["price"])
y = df["price"]

# Standarisasi
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=4)

# Gunakan Lasso dengan regularisasi ringan
lasso = Lasso(alpha=0.01)  # lebih kecil dari sebelumnya (lebih lunak)
lasso.fit(x_train, y_train)#dilatih menggunakan lasso regression

# Evaluasi
score = lasso.score(x_test, y_test)
print("Akurasi model Lasso: ", score)

# Koefisien tiap fitur
fitur = x.columns
koefisien = lasso.coef_
for f, k in zip(fitur, koefisien):
    print(f"{f}: {k}")

# Prediksi
kasus1 = scaler.transform([[3, 2, 2000, 7, 2015]])
hasil = lasso.predict(kasus1)
print("Harga Rumah (Lasso):", hasil)
