import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib

# Karakter görüntülerinin bulunduğu dizin
data_dir = 'karakter-veriseti-artirilm'  # Artırılmış görüntülerin dizini
labels = []
images = []

# Veri setini yükleme
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (20, 20))  # Görüntü boyutunu ayarlama
        images.append(image.flatten())  # Görüntüyü düzleştir
        labels.append(label)  # Etiket ekleme

# Verileri numpy dizisine dönüştürme
X = np.array(images)
y = np.array(labels)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluşturma ve eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test seti ile tahmin yapma
y_pred = model.predict(X_test)

# Doğruluğu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f'Model doğruluğu: {accuracy * 100:.2f}%')

# Modeli .pkl dosyası olarak kaydetme
joblib.dump(model, '3_random_forest_model.pkl') 