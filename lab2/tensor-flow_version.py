import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Чтение данных
df = pd.read_csv('lab2/dataset/heart_disease_data.csv')

# Подготовка данных
X = df.drop(['target'], axis=1)
y = df['target']

# Преобразование категориальных признаков в one-hot кодирование
X = pd.get_dummies(X, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Масштабирование числовых признаков
scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.float32), epochs=50, batch_size=32, verbose=1)

# Оценка модели на тестовых данных
y_pred = model.predict(X_test.values.astype(np.float32))
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy on test data: {accuracy}")

# Прогнозирование на новых данных
test_data = pd.DataFrame({
    'age': [58.0, 54.0, 56.0, 58.0, 51.0],
    'trestbps': [145, 130, 120, 130, 145],
    'chol': [1, 0, 1, 0, 1],
    'thalach': [150, 187, 172, 178, 165],
    'oldpeak': [2.3, 3.6, 3.5, 0.8, 1.3],
    'sex_0': [0, 0, 0, 1, 0],
    'sex_1': [1, 1, 1, 0, 1],
    'cp_0': [0, 0, 0, 0, 0],
    'cp_1': [1, 0, 0, 0, 0],
    'cp_2': [0, 1, 1, 1, 1],
    'cp_3': [0, 0, 0, 0, 0],
    'fbs_0': [0, 1, 0, 1, 1],
    'fbs_1': [1, 0, 1, 0, 0],
    'restecg_0': [1, 0, 1, 0, 1],
    'restecg_1': [0, 1, 0, 1, 0],
    'restecg_2': [0, 0, 0, 0, 0],
    'exang_0': [0, 1, 1, 0, 1],
    'exang_1': [1, 0, 0, 1, 0],
    'slope_0': [0, 0, 0, 0, 0],
    'slope_1': [1, 0, 1, 0, 1],
    'slope_2': [0, 1, 0, 1, 0],
    'ca_0': [0, 0, 0, 0, 0],
    'ca_1': [1, 1, 1, 0, 0],
    'ca_2': [0, 0, 0, 0, 0],
    'ca_3': [0, 0, 0, 1, 0],
    'ca_4': [0, 0, 0, 0, 1],
    'thal_0': [0, 0, 0, 0, 0],
    'thal_1': [1, 0, 0, 0, 0],
    'thal_2': [0, 1, 0, 1, 0],
    'thal_3': [0, 0, 1, 0, 1],
})

test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
test_predictions = model.predict(test_data.values.astype(np.float32))

print(test_predictions)
