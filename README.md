# Plaka Tanıma Sistemi

Bu proje, fotoğraflardaki araba plakalarını tespit edip, metin olarak okumak amacıyla geliştirilmiştir. Proje, görüntü işleme, nesne tespiti ve OCR (Optik Karakter Tanıma) teknolojilerini kullanmaktadır.

## Gereksinimler
- Python 3.x
- OpenCV
- Numpy
- Scikit-learn
- PaddleOCR veya TesseractOCR
- Ultralytics YOLO
- PostgreSQL

## Proje Yapısı
- **`plaka_tespit_test.py`**: Plaka tespit modelinin test edilmesi için kullanılan dosya. Kamera, video ve fotoğraf üzerinde plaka tespiti yapabilir.
- **`db_operations.py`**: Veritabanı işlemleri için kullanılan dosya. İzinli plaka kontrolü ve plaka kayıt işlemlerini yapar.
- **`train.py`**: Plaka tespit modelinin eğitimini gerçekleştiren dosya.
- **`veri_artirma.py`**: Eğitim verilerini artırmak için kullanılan dosya.
- **`split_dataset.py`**: Veri setini eğitim, doğrulama ve test setlerine ayıran dosya.
- **`requirements.txt`**: Projede kullanılan Python kütüphanelerinin listesi.

## Kullanım
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. PostgreSQL veritabanını kurun ve aşağıdaki tabloları oluşturun:
   - `izinli_plakalar`: İzinli araçların plakalarını ve izin tarihlerini içeren tablo
   - `plakalar`: Tespit edilen plakaların kaydedildiği tablo

3. Eğitim verilerinizi hazırlayın ve `karakter-veriseti-artirilm` dizinine yerleştirin.

4. Modeli eğitmek için `train.py` dosyasını çalıştırın:
   ```bash
   python train.py
   ```

5. Eğitilen modeli test etmek için `plaka_tespit_test.py` dosyasını çalıştırın:
   ```bash
   python plaka_tespit_test.py
   ```

## Plaka Tespit ve OCR İşlemi

Proje iki ana adımdan oluşmaktadır:

1. **Plaka Tespiti**: YOLO (You Only Look Once) derin öğrenme modeli kullanılarak görüntüdeki plaka konumları tespit edilir.
2. **Plaka Metni Okuma**: Tespit edilen plaka bölgesinden PaddleOCR kullanılarak metin çıkarılır.
3. **Plaka Doğrulama**: Okunan plaka metni düzenlenir ve veritabanında izin kontrolü yapılır.

## Veri Seti
Proje, farklı açılardan ve mekanlardan çekilmiş araba fotoğraflarını içeren bir veri seti kullanmaktadır. Veri seti, etiketleme araçları kullanılarak etiketlenmiştir. Plaka tespiti için kullanılan etiketleme formatı, YOLO modelinin gereksinimlerine uygun olarak aşağıdaki gibi olmalıdır:

### YOLO Etiketleme Formatı
- Her görüntü için bir etiket dosyası oluşturulmalıdır. Etiket dosyası, görüntü dosyası ile aynı isimde olmalı ve `.txt` uzantısına sahip olmalıdır.
- Etiket dosyası, her bir nesne için bir satır içermelidir. Her satır şu formatta olmalıdır:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
  - `<class_id>`: Plakanın sınıf kimliği (0, 1, 2, ... gibi).
  - `<x_center>` ve `<y_center>`: Plakanın görüntüdeki merkezinin normalleştirilmiş koordinatları (0 ile 1 arasında).
  - `<width>` ve `<height>`: Plakanın normalleştirilmiş genişliği ve yüksekliği (0 ile 1 arasında).

Örneğin, bir plaka görüntüsü için etiket dosyası şu şekilde olabilir:
```
0 0.5 0.5 0.2 0.1
```

## Veritabanı Yapısı

Proje iki ana veritabanı tablosu kullanmaktadır:

1. **izinli_plakalar**: İzinli araçların bilgilerini içerir
   - `plaka`: Plakanın kendisi (Primary Key)
   - `aktif`: İznin aktif olup olmadığı (Boolean)
   - `baslangic_tarih`: İzin başlangıç tarihi
   - `bitis_tarih`: İzin bitiş tarihi

2. **plakalar**: Tespit edilen tüm plakaların kaydını tutar
   - `id`: Otomatik artan benzersiz kimlik
   - `plaka`: Tespit edilen plaka metni
   - `durum`: İzinli olup olmadığı (Boolean)
   - `tespit_zamani`: Plakanın ne zaman tespit edildiği

## Katkıda Bulunma
Herhangi bir katkıda bulunmak isterseniz, lütfen bir pull request oluşturun.
