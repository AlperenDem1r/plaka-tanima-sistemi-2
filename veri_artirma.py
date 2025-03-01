# Bu dosya, veri artırma tekniklerini uygular. Eğitim verilerini zenginleştirerek modelin daha iyi öğrenmesini sağlar.

import cv2  # OpenCV kütüphanesini görüntü işleme için kullanır
import numpy as np  # Numpy kütüphanesini matematiksel işlemler için kullanır
from pathlib import Path  # Dosya yollarını yönetmek için kullanılır
from tqdm import tqdm  # İlerleme çubuğu için kullanılır
import os  # Dosya ve dizin işlemleri için kullanılır

def rotate_image(image, angle):
    """Görüntüyü belirtilen açı kadar döndürür"""
    height, width = image.shape[:2]  # Görüntünün boyutlarını alır
    center = (width // 2, height // 2)  # Görüntünün merkezini hesaplar
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Dönüşüm matrisini oluşturur
    return cv2.warpAffine(image, rotation_matrix, (width, height))  # Görüntüyü döndürür

def adjust_brightness(image, factor):
    """Görüntünün parlaklığını ayarlar"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Görüntüyü HSV renk uzayına çevirir
    hsv = hsv.astype(np.float32)  # HSV değerlerini float32 tipine çevirir
    hsv[:,:,2] = hsv[:,:,2] * factor  # Parlaklık kanalını ayarlar
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)  # Parlaklık değerlerini 0-255 aralığına sınırlar
    hsv = hsv.astype(np.uint8)  # HSV değerlerini uint8 tipine çevirir
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Görüntüyü tekrar BGR formatına çevirir

def add_noise(image, noise_factor=0.1):
    """Görüntüye gürültü ekler"""
    noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.uint8)  # Gürültü oluşturur
    noisy_image = cv2.add(image, noise)  # Gürültüyü görüntüye ekler
    return noisy_image  # Gürültülü görüntüyü döner

def resize_image(image, scale):
    """Görüntü boyutunu artırır"""
    width = int(image.shape[1] * scale)  # Yeni genişliği hesaplar
    height = int(image.shape[0] * scale)  # Yeni yüksekliği hesaplar
    return cv2.resize(image, (width, height))  # Görüntüyü yeniden boyutlandırır

def create_augmented_image(image, index):
    """Görüntünün artırılmış versiyonunu oluşturur"""
    if index == 0:  # Boyut arttırma
        return resize_image(image, 1.5)  # Görüntüyü %50 büyütür
    elif index == 1:  # Gri tonlama
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tonlamaya çevirir
    elif index == 2:  # Binarizasyon
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tonlamaya çevirir
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # Eşikleme ile binarize eder
        return binary  # Binarize edilmiş görüntüyü döner
    elif index == 3:  # Açı değiştirme +15
        return rotate_image(image, 15)  # Görüntüyü 15 derece döndürür
    elif index == 4:  # Açı değiştirme -15
        return rotate_image(image, -15)  # Görüntüyü -15 derece döndürür
    else:
        return image  # Diğer durumlar için orijinal görüntüyü döndür

def augment_dataset(input_dir, output_dir, num_augmentations_per_image=6):
    """Veri setindeki her görüntü için veri artırma işlemi yapar"""
    try:
        # Çıktı dizinini oluştur
        output_path = Path(output_dir)  # Çıktı dizinini Path nesnesi olarak oluşturur
        output_path.mkdir(parents=True, exist_ok=True)  # Dizin yoksa oluşturur
        
        # Giriş dizinindeki tüm görüntüleri işle
        input_path = Path(input_dir)  # Giriş dizinini Path nesnesi olarak oluşturur
        input_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))  # Tüm JPG ve PNG dosyalarını alır
        
        if not input_files:
            print(f"HATA: {input_dir} klasöründe hiç görüntü bulunamadı!")  # Eğer görüntü yoksa hata mesajı gösterir
            return
        
        print(f"Toplam {len(input_files)} görüntü bulundu.")  # Toplam görüntü sayısını yazdırır
        print(f"Her görüntü için {num_augmentations_per_image} artırılmış versiyon oluşturulacak.")  # Her görüntü için kaç artırma yapılacağını belirtir
        print(f"Toplam {len(input_files) * num_augmentations_per_image} yeni görüntü oluşturulacak.")  # Toplam yeni görüntü sayısını hesaplar
        
        for img_path in tqdm(input_files, desc="Görüntüler işleniyor"):  # Her görüntü için döngü
            try:
                # Görüntüyü oku
                image = cv2.imread(str(img_path))  # Görüntüyü okur
                if image is None:
                    print(f"Hata: {img_path} okunamadı.")  # Eğer görüntü okunamazsa hata mesajı gösterir
                    continue
                
                # Her görüntü için belirtilen sayıda artırılmış versiyon oluştur
                for i in range(num_augmentations_per_image):  # Her görüntü için artırma döngüsü
                    try:
                        # Dönüşümü uygula
                        augmented = create_augmented_image(image, i)  # Görüntüyü artırır
                        
                        # Yeni dosya adı oluştur
                        new_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"  # Yeni dosya adını oluşturur
                        output_file = output_path / new_filename  # Çıktı dosya yolunu oluşturur
                        
                        # Görüntüyü kaydet
                        cv2.imwrite(str(output_file), augmented)  # Artırılmış görüntüyü kaydeder
                    except Exception as e:
                        print(f"Görüntü artırma hatası ({img_path.name}, {i}): {str(e)}")  # Hata mesajı gösterir
            except Exception as e:
                print(f"Görüntü işleme hatası ({img_path}): {str(e)}")  # Hata mesajı gösterir
    except Exception as e:
        print(f"Genel hata: {str(e)}")  # Genel hata mesajı gösterir

if __name__ == "__main__":
    # Giriş ve çıkış dizinlerini belirle
    current_dir = Path.cwd()  # Mevcut dizini alır
    input_directory = current_dir / "karakter-veriseti" / "arkaplan"  # Giriş dizinini oluşturur
    output_directory = current_dir / "karakter-veriseti-artirilm" / "arkaplan"  # Çıkış dizinini oluşturur
    
    print(f"Giriş dizini: {input_directory}")  # Giriş dizinini yazdırır
    print(f"Çıkış dizini: {output_directory}")  # Çıkış dizinini yazdırır
    
    # Veri artırma işlemini başlat
    augment_dataset(input_directory, output_directory)  # Veri artırma fonksiyonunu çağırır
    print("Veri artırma işlemi tamamlandı!")  # İşlem tamamlandığında mesaj gösterir 