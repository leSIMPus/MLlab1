import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Загрузка датасета ---
csv_path = './data/student-scores.csv'
df = pd.read_csv(csv_path)

print("Первые строки таблицы:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())

# --- 2. Визуализация распределения целевого признака ---
plt.figure(figsize=(8, 4))
sns.histplot(df['math_score'], bins=15, kde=True)
plt.title("Распределение результатов по математике")
plt.xlabel("Баллы")
plt.ylabel("Количество учеников")
plt.show()

# --- 3. Проверка и обработка пропусков ---
print("\nПроверка на пропущенные значения:")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("\n⚠️ Ошибка: в данных нет пропусков! Создаём искусственные пропуски для демонстрации...")
    # Сгенерируем пропуски случайно в 1% данных
    df.loc[df.sample(frac=0.01).index, 'part_time_job'] = np.nan

# Повторная проверка
print("\nПосле возможной генерации пропусков:")
print(df.isnull().sum())

# Заполним пропуски средним значением по столбцу
df['part_time_job'].fillna(df['part_time_job'].mean(), inplace=True)

# --- 4. Преобразование категориальных данных ---
cat_cols = df.select_dtypes(include=['object']).columns
print(f"\nКатегориальные признаки: {cat_cols.tolist()}")

if len(cat_cols) == 0:
    print("⚠️ Ошибка: нет категориальных признаков для кодирования! Добавим искусственный пример.")
    df['gender'] = np.random.choice(['male', 'female'], size=len(df))
    cat_cols = ['gender']

# Преобразуем категориальные признаки в числовые
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# --- 5. Визуализация корреляций ---
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляция признаков")
plt.show()

# --- 6. Нормализация (стандартизация) числовых признаков ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nПосле стандартизации:")
print(df.head())

# --- 7. Проверим итоговую готовность ---
print("\n✅ Все данные числовые и без пропусков:")
print(df.info())

print(df.columns.tolist())
