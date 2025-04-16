# %% [markdown]
# # Klasifikasi Dataset Wine dengan Decision Tree
# Latihan menggunakan algoritma Decision Tree pada dataset wine dari sklearn

# %% [code]
# Import library yang dibutuhkan
from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% [code]
# Fungsi bantu untuk mengubah dataset sklearn menjadi DataFrame
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = sklearn_dataset.target
    return df

# Load dataset wine
wine = sklearn_to_df(datasets.load_wine())
wine.rename(columns={'target': 'class'}, inplace=True)

# Tampilkan ringkasan statistik
wine.describe().T

# %% [code]
# Tampilkan 10 data pertama
print(wine.head(10))

# %% [code]
# Visualisasi data dengan pairplot
sns.pairplot(wine, hue='class', palette='Set2')
plt.show()

# %% [code]
# Split data menjadi training dan testing
from sklearn.model_selection import train_test_split

x = wine.drop('class', axis=1)
y = wine['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("Jumlah data training:", len(x_train))
print("Jumlah data testing:", len(x_test))

# %% [code]
# Membuat model Decision Tree
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(x_train, y_train)

# Prediksi data testing
y_pred = model.predict(x_test)

# %% [code]
# Evaluasi model
from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))

# %% [code]
# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %% [code]
# Visualisasi pohon keputusan
from sklearn import tree

fig, ax = plt.subplots(figsize=(25, 15))
tree.plot_tree(model, feature_names=wine.columns[:-1], class_names=[str(i) for i in np.unique(y)], filled=True)
plt.title("Decision Tree - Wine Dataset")
plt.show()

# %% [code]
# Uji coba prediksi dengan data baru
wine_test_data = {
    'alcohol': 13.2,
    'malic_acid': 1.78,
    'ash': 2.14,
    'alcalinity_of_ash': 11.2,
    'magnesium': 100,
    'total_phenols': 2.6,
    'flavanoids': 2.5,
    'nonflavanoid_phenols': 0.3,
    'proanthocyanins': 1.5,
    'color_intensity': 5.0,
    'hue': 1.0,
    'od280/od315_of_diluted_wines': 3.0,
    'proline': 1000
}

wine_df = pd.DataFrame([wine_test_data])
prediction = model.predict(wine_df)
print("Prediksi kelas:", prediction)
