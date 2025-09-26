import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

dataset = 'whenamancodes/students-performance-in-exams'
api.dataset_download_files(dataset, path='./data', unzip=True)

csv_files = [f for f in os.listdir('./data') if f.endswith('.csv')]
csv_path = os.path.join('./data', csv_files[0])

df = pd.read_csv(csv_path)

print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(df.head())

# Пусть целевой признак - результаты по математике
target = df["math score"]

print("\nНекоторые значения целевого признака:")
print(target.head())