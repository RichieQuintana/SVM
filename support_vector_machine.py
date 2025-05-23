# Máquinas de Vectores de Soporte (SVM)

#Explicación:
#Entrenamiento y prueba: Este script comienza cargando los datos y dividiéndolos en conjuntos de entrenamiento y prueba. Los estudiantes pueden ver cómo se separan los datos para evaluar el modelo en datos no vistos.
#Escalado de características: Se aplica el StandardScaler para normalizar las características, lo que es esencial para el modelo SVM, ya que es sensible a la escala de los datos.
#Entrenamiento del modelo: Se utiliza el SVC con un kernel lineal para entrenar el modelo y luego se realiza una predicción sobre un nuevo ejemplo (edad 30, salario 87000).
#Evaluación: Se genera la matriz de confusión y se calcula la precisión del modelo, lo que ayuda a evaluar su rendimiento.
#Visualización: Se visualizan los resultados en el conjunto de entrenamiento y prueba, mostrando las áreas de decisión del modelo.

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

# Mostrar columnas para verificaci�n
print("Columnas del dataset:")
print(dataset.columns)

# Usar la columna 'condition' como la variable objetivo
y = dataset['condition'].values

# Usar dos caracter�sticas: 'age' y 'chol' para visualizaci�n
X = dataset[['age', 'chol']].values

# Divisi�n del conjunto
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("Conjunto de entrenamiento (X_train):", X_train)
print("Etiquetas de entrenamiento (y_train):", y_train)

# Escalado de caracter�sticas
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento del modelo
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predicci�n
y_pred = classifier.predict(X_test)
print("Predicciones (y_pred vs y_test):")
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

# Evaluaci�n
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusi�n:")
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("Precisi�n del modelo:", accuracy)

# Visualizaci�n - Entrenamiento
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

# Visualizaci�n - Prueba
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

















































































