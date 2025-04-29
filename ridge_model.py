import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Baca data
file = "perumahan.csv"
df = pd.read_csv(file, usecols=["price", "bedrooms", "bathrooms", "sqft_living", "grade", "yr_built"])

# Pisah fitur dan target
x = df.drop(columns=["price"])
y = df["price"]

# Standarisasi (penting untuk Ridge)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Bagi data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=4)

# Gunakan Ridge Regression (L2)
ridge = Ridge(alpha=1.0)  # ubah alpha sesuai kebutuhan (coba 0.01, 1, 10)
ridge.fit(x_train, y_train)

# Evaluasi model
score = ridge.score(x_test, y_test)
print("Akurasi model Ridge: ", score)

# Tampilkan koefisien tiap fitur
fitur = x.columns
koefisien = ridge.coef_
for f, k in zip(fitur, koefisien):
    print(f"{f}: {k}")

# Prediksi 1 kasus
kasus1 = scaler.transform([[3, 2, 2000, 7, 2015]])
hasil = ridge.predict(kasus1)
print("Harga Rumah (Ridge):", hasil)
