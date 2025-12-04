import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# задание 1
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

# задание 2
target = "status"
cols_all = df.columns.tolist()
feature_cols = [c for c in cols_all if c not in ("name", target)]

non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print("Обнаружены неправильно классифицированные числовые признаки:", non_numeric_cols)
    df[non_numeric_cols] = df[non_numeric_cols].apply(pd.to_numeric, errors="coerce")
else:
    print("Все числовые признаки классифицированы корректно.")

df.replace('?', np.nan, inplace=True)
missing_counts = df[feature_cols].isnull().sum().sum()
if missing_counts == 0:
    print("Пропущенных значений в числовых признаках нет.")
else:
    print(f"Обнаружено {missing_counts} пропущенных значений, заполняем медианой.")
    num_imputer = SimpleImputer(strategy="median")
    df[feature_cols] = num_imputer.fit_transform(df[feature_cols])

# задание 3
if df[target].nunique() > 10:
    df[target + "_disc"] = pd.cut(df[target], bins=5, labels=False)
    target_col = target + "_disc"
    print(f"Метка класса '{target}' дискретизирована в 5 диапазонов.")
else:
    target_col = target
    print(f"Метка класса '{target}' принимает {df[target].nunique()} различных значений. Дискретизация не требуется.")

# задание 4
X = df[feature_cols].copy()
y = df[target_col].copy()

use_chi2 = (X.min().min() >= 0)

if use_chi2:
    X_sel = MinMaxScaler().fit_transform(X)
    selector = SelectKBest(score_func=chi2, k=2)
else:
    X_sel = X.values
    selector = SelectKBest(score_func=f_classif, k=2)

selector.fit(X_sel, y)
mask = selector.get_support()
selected_features = [f for f, m in zip(feature_cols, mask) if m]

print("Найденные два признака:", selected_features)

# задание 5
corr_cols = selected_features + [target_col]
corr_df = df[corr_cols].copy()
corr_df[target_col] = pd.to_numeric(corr_df[target_col], errors='coerce')
corr_matrix = corr_df.corr(method='pearson')

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", square=True, cmap="vlag", cbar_kws={"shrink": .8})
plt.title("Матрица корреляций:")
plt.tight_layout()
plt.show()

# задание 6
plt.figure(figsize=(8,6))
classes = np.sort(df[target].unique())
palette = sns.color_palette(n_colors=len(classes))
for cls, col in zip(classes, palette):
    mask_cls = df[target] == cls
    plt.scatter(df.loc[mask_cls, selected_features[0]],
                df.loc[mask_cls, selected_features[1]],
                label=f"status={int(cls)}",
                alpha=0.8, edgecolors='w', s=60)

plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.title("Диаграмма рассеяния:")
plt.legend(title="Класс")
plt.grid(True)
plt.tight_layout()
plt.show()

# задание 7

X_all = df[feature_cols].values
X_all_scaled = StandardScaler().fit_transform(X_all)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_all_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df[target] = df[target].values

plt.figure(figsize=(8,6))
for cls, col in zip(classes, palette):
    mask_cls = pca_df[target] == cls
    plt.scatter(pca_df.loc[mask_cls, "PC1"],
                pca_df.loc[mask_cls, "PC2"],
                label=f"status={int(cls)}",
                alpha=0.8, edgecolors='w', s=60)

explained = pca.explained_variance_ratio_
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
plt.title("Диаграмма рассеяния с PCA:")
plt.legend(title="Класс")
plt.grid(True)
plt.tight_layout()
plt.show()
