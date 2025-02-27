import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """Veri setini eğitim, doğrulama ve test olarak böler"""
    # Görüntü dosyalarını listele
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
    
    # Dosya sayısını kontrol et
    if not image_files:
        print("Hata: Görüntü dosyası bulunamadı!")
        return
    
    print(f"Toplam {len(image_files)} görüntü bulundu.")
    
    # Dosyaları karıştır
    random.shuffle(image_files)
    
    # Bölme noktalarını hesapla
    train_end = int(len(image_files) * train_ratio)
    valid_end = int(len(image_files) * (train_ratio + valid_ratio))
    
    # Dosyaları böl
    train_files = image_files[:train_end]
    valid_files = image_files[train_end:valid_end]
    test_files = image_files[valid_end:]
    
    print(f"Eğitim seti: {len(train_files)} görüntü")
    print(f"Doğrulama seti: {len(valid_files)} görüntü")
    print(f"Test seti: {len(test_files)} görüntü")
    
    # Her set için dosyaları kopyala
    sets = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }
    
    for set_name, files in sets.items():
        # Hedef klasörler
        set_images_dir = os.path.join(output_dir, set_name, 'images')
        set_labels_dir = os.path.join(output_dir, set_name, 'labels')
        
        # Klasörleri oluştur
        os.makedirs(set_images_dir, exist_ok=True)
        os.makedirs(set_labels_dir, exist_ok=True)
        
        print(f"\n{set_name} seti için dosyalar kopyalanıyor...")
        for img_file in tqdm(files, desc=f"{set_name} seti"):
            # Görüntü dosyasını kopyala
            shutil.copy2(img_file, set_images_dir)
            
            # Etiket dosyasını kopyala
            label_file = Path(labels_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, set_labels_dir)
            else:
                print(f"Uyarı: {label_file} bulunamadı!")

if __name__ == "__main__":
    # Klasör yolları
    images_dir = "dataset/images"
    labels_dir = "dataset/labels_yolo"
    output_dir = "yolov8_dataset"
    
    # Veri setini böl
    split_dataset(images_dir, labels_dir, output_dir)
    
    # classes.txt dosyasını kopyala
    shutil.copy2(os.path.join(labels_dir, "classes.txt"), output_dir)
    
    print("\nVeri seti bölme işlemi tamamlandı!") 