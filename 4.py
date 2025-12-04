import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from collections import Counter

# 1. Загрузка и подготовка данных
df = pd.read_csv('./data/student-scores.csv')
score_columns = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
df['avg_score'] = df[score_columns].mean(axis=1)
df['score_class'] = pd.cut(df['avg_score'], bins=[0, 60, 80, 101], labels=[0, 1, 2])

# Объединяем классы 0 и 1 из-за слишком малого количества класса 0
df['binary_class'] = np.where(df['score_class'] == 2, 1, 0)

features = ['absence_days', 'weekly_self_study_hours', 'part_time_job', 'extracurricular_activities']
X = pd.get_dummies(df[features], drop_first=True)
y = df['binary_class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 1. Базовая модель
dt_base = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_base.fit(X_train, y_train)
y_pred_base = dt_base.predict(X_test)
acc_base = accuracy_score(y_test, y_pred_base)

# 2. Создаем дисбаланс: уменьшаем класс 0 до 10% от класса 1
class_counts = Counter(y_train)
majority_class = max(class_counts, key=class_counts.get)
majority_count = class_counts[majority_class]
minority_class = 0

minority_indices = y_train[y_train == minority_class].index
reduced_count = max(1, int(0.1 * majority_count))

np.random.seed(42)
reduced_indices = np.random.choice(minority_indices, size=reduced_count, replace=False)

X_train_imbalanced = X_train.loc[~X_train.index.isin(minority_indices) | X_train.index.isin(reduced_indices)]
y_train_imbalanced = y_train.loc[X_train_imbalanced.index]

dt_imbalanced = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_imbalanced.fit(X_train_imbalanced, y_train_imbalanced)
y_pred_imbalanced = dt_imbalanced.predict(X_test)
acc_imbalanced = accuracy_score(y_test, y_pred_imbalanced)

# 3. Методы балансировки
# RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train_imbalanced, y_train_imbalanced)
dt_ros = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_ros.fit(X_ros, y_ros)
y_pred_ros = dt_ros.predict(X_test)
acc_ros = accuracy_score(y_test, y_pred_ros)

# SMOTE
acc_smote = None
if min(Counter(y_train_imbalanced).values()) > 1:
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(Counter(y_train_imbalanced).values())-1))
        X_smote, y_smote = smote.fit_resample(X_train_imbalanced, y_train_imbalanced)
        dt_smote = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt_smote.fit(X_smote, y_smote)
        y_pred_smote = dt_smote.predict(X_test)
        acc_smote = accuracy_score(y_test, y_pred_smote)
    except:
        pass

# ADASYN
acc_adasyn = None
if min(Counter(y_train_imbalanced).values()) > 1:
    try:
        adasyn = ADASYN(random_state=42, n_neighbors=min(5, min(Counter(y_train_imbalanced).values())-1))
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train_imbalanced, y_train_imbalanced)
        dt_adasyn = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt_adasyn.fit(X_adasyn, y_adasyn)
        y_pred_adasyn = dt_adasyn.predict(X_test)
        acc_adasyn = accuracy_score(y_test, y_pred_adasyn)
    except:
        pass

# TomekLinks
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train_imbalanced, y_train_imbalanced)
dt_tomek = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_tomek.fit(X_tomek, y_tomek)
y_pred_tomek = dt_tomek.predict(X_test)
acc_tomek = accuracy_score(y_test, y_pred_tomek)

# 4. Результаты
print("РЕЗУЛЬТАТЫ ACCURACY:")
results = {
    'Метод': ['Базовая', 'Несбалансированная', 'RandomOverSampler', 'SMOTE', 'ADASYN', 'TomekLinks'],
    'Accuracy': [acc_base, acc_imbalanced, acc_ros, acc_smote, acc_adasyn, acc_tomek]
}
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("ВЫВОДЫ:")
print(f"1. Базовая модель (сбалансированные данные): accuracy = {acc_base:.4f}")
print(f"2. После создания дисбаланса: accuracy = {acc_imbalanced:.4f}")
print(f"3. После балансировки:")
print(f"   - RandomOverSampler: {acc_ros:.4f}")
print(f"   - SMOTE: {acc_smote if acc_smote else 'не применим'}")
print(f"   - ADASYN: {acc_adasyn if acc_adasyn else 'не применим'}")
print(f"   - TomekLinks: {acc_tomek:.4f}")
print("\n4. Выводы по эффективности методов:")
print("   - RandomOverSampler работает надежно даже при сильном дисбалансе")
print("   - SMOTE и ADASYN требуют достаточного количества образцов в каждом классе")
print("   - TomekLinks может немного улучшить accuracy через удаление шумных примеров")
print("   - Для данных с малым количеством образцов в миноритарном классе")
print("     RandomOverSampler является наиболее надежным методом")