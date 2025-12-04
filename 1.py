import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()


csv_path = './data/student-scores.csv'

df = pd.read_csv(csv_path)

print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(df.head())

# Пусть целевой признак - результаты по математике
target = df["math_score"]

print("\nНекоторые значения целевого признака:")
print(target.head())