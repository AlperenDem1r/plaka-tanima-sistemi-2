import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

def rotate_image(image, angle):
    """Görüntüyü belirtilen açı kadar döndürür"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def adjust_brightness(image, factor):
    """Görüntünün parlaklığını ayarlar"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:,:,2] = hsv[:,:,2] * factor
    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_noise(image, noise_factor=0.1):
    """Görüntüye gürültü ekler"""
    noise = np.random.normal(0, noise_factor * 255, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def resize_image(image, scale):
    """Görüntü boyutunu artırır"""
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height))

def create_augmented_image(image, index):
    """Görüntünün artırılmış versiyonunu oluşturur"""
    if index == 0:  # Boyut arttırma
        return resize_image(image, 1.5)
    elif index == 1:  # Gri tonlama
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif index == 2:  # Binarizasyon
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return binary
    elif index == 3:  # Açı değiştirme +15
        return rotate_image(image, 15)
    elif index == 4:  # Açı değiştirme -15
        return rotate_image(image, -15)
    else:
        return image  # Diğer durumlar için orijinal görüntüyü döndür

def augment_dataset(input_dir, output_dir, num_augmentations_per_image=6):
    """Veri setindeki her görüntü için veri artırma işlemi yapar"""
    try:
        # Çıktı dizinini oluştur
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Giriş dizinindeki tüm görüntüleri işle
        input_path = Path(input_dir)
        input_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        
        if not input_files:
            print(f"HATA: {input_dir} klasöründe hiç görüntü bulunamadı!")
            return
        
        print(f"Toplam {len(input_files)} görüntü bulundu.")
        print(f"Her görüntü için {num_augmentations_per_image} artırılmış versiyon oluşturulacak.")
        print(f"Toplam {len(input_files) * num_augmentations_per_image} yeni görüntü oluşturulacak.")
        
        for img_path in tqdm(input_files, desc="Görüntüler işleniyor"):
            try:
                # Görüntüyü oku
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Hata: {img_path} okunamadı.")
                    continue
                
                # Her görüntü için belirtilen sayıda artırılmış versiyon oluştur
                for i in range(num_augmentations_per_image):
                    try:
                        # Dönüşümü uygula
                        augmented = create_augmented_image(image, i)
                        
                        # Yeni dosya adı oluştur
                        new_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"
                        output_file = output_path / new_filename
                        
                        # Görüntüyü kaydet
                        cv2.imwrite(str(output_file), augmented)
                    except Exception as e:
                        print(f"Görüntü artırma hatası ({img_path.name}, {i}): {str(e)}")
            except Exception as e:
                print(f"Görüntü işleme hatası ({img_path}): {str(e)}")
    except Exception as e:
        print(f"Genel hata: {str(e)}")

if __name__ == "__main__":
    # Giriş ve çıkış dizinlerini belirle
    current_dir = Path.cwd()
    input_directory = current_dir / "karakter-veriseti" / "arkaplan"
    output_directory = current_dir / "karakter-veriseti-artirilm" / "arkaplan"
    
    print(f"Giriş dizini: {input_directory}")
    print(f"Çıkış dizini: {output_directory}")
    
    # Veri artırma işlemini başlat
    augment_dataset(input_directory, output_directory)
    print("Veri artırma işlemi tamamlandı!") 