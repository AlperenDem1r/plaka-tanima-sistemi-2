import cv2  # OpenCV kütüphanesini görüntü işleme için kullanır
import numpy as np  # Numpy kütüphanesini matematiksel işlemler için kullanır
from sklearn.ensemble import RandomForestClassifier  # Random Forest sınıflandırıcı modelini kullanmak için gerekli kütüphane
from sklearn.model_selection import train_test_split  # Eğitim ve test setlerine ayırmak için kullanılır
from sklearn.metrics import accuracy_score  # Modelin doğruluğunu hesaplamak için kullanılır
import os  # Dosya ve dizin işlemleri için kullanılır
import joblib  # Modeli kaydetmek ve yüklemek için kullanılır

# Bu dosya, plaka tespit modelinin eğitimini gerçekleştirir. Eğitim verilerini kullanarak modelin öğrenmesini sağlar.

# Karakter görüntülerinin bulunduğu dizin
# Artırılmış görüntülerin dizini

data_dir = 'karakter-veriseti-artirilm'  # Artırılmış görüntülerin dizini
labels = []  # Etiketleri saklamak için liste
images = []  # Görüntüleri saklamak için liste

# Veri setini yükleme
for label in os.listdir(data_dir):  # Her etiket dizinini döngü ile gezer
    label_dir = os.path.join(data_dir, label)  # Etiket dizinini oluşturur
    for image_file in os.listdir(label_dir):  # Her etiket altındaki görüntü dosyalarını döngü ile gezer
        image_path = os.path.join(label_dir, image_file)  # Görüntü dosyasının tam yolunu oluşturur
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Görüntüyü gri tonlamada okur
        image = cv2.resize(image, (20, 20))  # Görüntü boyutunu ayarlama
        images.append(image.flatten())  # Görüntüyü düzleştirip listeye ekler
        labels.append(label)  # Etiket ekleme

# Verileri numpy dizisine dönüştürme
X = np.array(images)  # Görüntüleri numpy dizisine çevirir
y = np.array(labels)  # Etiketleri numpy dizisine çevirir

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Veriyi eğitim ve test setlerine ayırır

# Random Forest modelini oluşturma ve eğitme
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Random Forest modelini oluşturur
model.fit(X_train, y_train)  # Modeli eğitim verileri ile eğitir

# Test seti ile tahmin yapma
y_pred = model.predict(X_test)  # Test verileri ile tahmin yapar

# Doğruluğu hesaplama
accuracy = accuracy_score(y_test, y_pred)  # Modelin doğruluğunu hesaplar
print(f'Model doğruluğu: {accuracy * 100:.2f}%')  # Doğruluğu ekrana yazdırır

# Modeli .pkl dosyası olarak kaydetme
joblib.dump(model, '3_random_forest_model.pkl')  # Eğitilmiş modeli dosyaya kaydeder 