# Veri setini eğitim, doğrulama ve test setlerine ayırmak için kullanılır.

import os  # Dosya ve dizin işlemleri için kullanılır
import shutil  # Dosya kopyalama işlemleri için kullanılır
import random  # Rastgele sayı üretimi için kullanılır
from pathlib import Path  # Dosya yollarını yönetmek için kullanılır
from tqdm import tqdm  # İlerleme çubuğu için kullanılır

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    """Veri setini eğitim, doğrulama ve test olarak böler"""
    # Görüntü dosyalarını listele
    image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))  # JPG ve PNG dosyalarını alır
    
    # Dosya sayısını kontrol et
    if not image_files:
        print("Hata: Görüntü dosyası bulunamadı!")  # Eğer görüntü yoksa hata mesajı gösterir
        return
    
    print(f"Toplam {len(image_files)} görüntü bulundu.")  # Toplam görüntü sayısını yazdırır
    
    # Dosyaları karıştır
    random.shuffle(image_files)  # Görüntü dosyalarını karıştırır
    
    # Bölme noktalarını hesapla
    train_end = int(len(image_files) * train_ratio)  # Eğitim setinin sonunu hesaplar
    valid_end = int(len(image_files) * (train_ratio + valid_ratio))  # Doğrulama setinin sonunu hesaplar
    
    # Dosyaları böl
    train_files = image_files[:train_end]  # Eğitim seti dosyaları
    valid_files = image_files[train_end:valid_end]  # Doğrulama seti dosyaları
    test_files = image_files[valid_end:]  # Test seti dosyaları
    
    print(f"Eğitim seti: {len(train_files)} görüntü")  # Eğitim seti sayısını yazdırır
    print(f"Doğrulama seti: {len(valid_files)} görüntü")  # Doğrulama seti sayısını yazdırır
    print(f"Test seti: {len(test_files)} görüntü")  # Test seti sayısını yazdırır
    
    # Her set için dosyaları kopyala
    sets = {
        'train': train_files,
        'valid': valid_files,
        'test': test_files
    }  # Setleri bir sözlükte saklar
    
    for set_name, files in sets.items():
        # Hedef klasörler
        set_images_dir = os.path.join(output_dir, set_name, 'images')  # Görüntü dosyaları için hedef dizin
        set_labels_dir = os.path.join(output_dir, set_name, 'labels')  # Etiket dosyaları için hedef dizin
        # Klasörleri oluştur
        os.makedirs(set_images_dir, exist_ok=True)  # Görüntü dizinini oluşturur
        os.makedirs(set_labels_dir, exist_ok=True)  # Etiket dizinini oluşturur
        
        print(f"\n{set_name} seti için dosyalar kopyalanıyor...")  # Hedef set için bilgi verir
        for img_file in tqdm(files, desc=f"{set_name} seti"):  # Her set için döngü
            # Görüntü dosyasını kopyala
            shutil.copy2(img_file, set_images_dir)  # Görüntü dosyasını kopyalar
            
            # Etiket dosyasını kopyala
            label_file = Path(labels_dir) / f"{img_file.stem}.txt"  # Etiket dosyasının yolunu oluşturur
            if label_file.exists():
                shutil.copy2(label_file, set_labels_dir)  # Etiket dosyasını kopyalar
            else:
                print(f"Uyarı: {label_file} bulunamadı!")  # Eğer etiket dosyası yoksa uyarı verir

if __name__ == "__main__":
    # Klasör yolları
    images_dir = "dataset/images"  # Görüntü dosyalarının bulunduğu dizin
    labels_dir = "dataset/labels_yolo"  # Etiket dosyalarının bulunduğu dizin
    output_dir = "yolov8_dataset"  # Çıktı dizininin adı
    
    # Veri setini böl
    split_dataset(images_dir, labels_dir, output_dir)  # Veri setini böler
    
    # classes.txt dosyasını kopyala
    shutil.copy2(os.path.join(labels_dir, "classes.txt"), output_dir)  # Sınıf dosyasını kopyalar
    
    print("\nVeri seti bölme işlemi tamamlandı!")  # İşlem tamamlandığında mesaj gösterir 