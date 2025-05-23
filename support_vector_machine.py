# M谩quinas de Vectores de Soporte (SVM)

#Explicaci贸n:
#Entrenamiento y prueba: Este script comienza cargando los datos y dividi茅ndolos en conjuntos de entrenamiento y prueba. Los estudiantes pueden ver c贸mo se separan los datos para evaluar el modelo en datos no vistos.
#Escalado de caracter铆sticas: Se aplica el StandardScaler para normalizar las caracter铆sticas, lo que es esencial para el modelo SVM, ya que es sensible a la escala de los datos.
#Entrenamiento del modelo: Se utiliza el SVC con un kernel lineal para entrenar el modelo y luego se realiza una predicci贸n sobre un nuevo ejemplo (edad 30, salario 87000).
#Evaluaci贸n: Se genera la matriz de confusi贸n y se calcula la precisi贸n del modelo, lo que ayuda a evaluar su rendimiento.
#Visualizaci贸n: Se visualizan los resultados en el conjunto de entrenamiento y prueba, mostrando las 谩reas de decisi贸n del modelo.

# SVM - Heart Disease Dataset (usando la columna 'condition' como variable objetivo)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Cargar el archivo real
dataset = pd.read_csv('heart.csv')
dataset.columns = dataset.columns.str.strip()  # Elimina espacios en los nombres

# Mostrar columnas para verificaci髇
print("Columnas del dataset:")
print(dataset.columns)

# Usar la columna 'condition' como la variable objetivo
y = dataset['condition'].values

# Usar dos caracter韘ticas: 'age' y 'chol' para visualizaci髇
X = dataset[['age', 'chol']].values

# Divisi髇 del conjunto
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("Conjunto de entrenamiento (X_train):", X_train)
print("Etiquetas de entrenamiento (y_train):", y_train)

# Escalado de caracter韘ticas
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento del modelo
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicci髇
y_pred = classifier.predict(X_test)
print("Predicciones (y_pred vs y_test):")
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

# Evaluaci髇
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusi髇:")
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("Precisi髇 del modelo:", accuracy)

# Visualizaci髇 - Entrenamiento
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=1))
plt.contourf(X1, X2, classifier.predict(
    sc.transform(np.array([X1.ravel(), X2.ravel()]).T)
).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.legend()
plt.show()

# Visualizaci髇 - Prueba
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 10, stop=X_set[:, 1].max() + 10, step=1))
plt.contourf(X1, X2, classifier.predict(
    sc.transform(np.array([X1.ravel(), X2.ravel()]).T)
).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM (Prueba)')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.legend()
plt.show()

















































































