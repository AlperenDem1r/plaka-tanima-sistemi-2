import cv2  # OpenCV kütüphanesini görüntü işleme için kullanır
from ultralytics import YOLO  # YOLO modelini kullanmak için gerekli kütüphane
import time  # Zaman işlemleri için kullanılır
import tkinter as tk  # Tkinter kütüphanesini GUI oluşturmak için kullanır
from tkinter import filedialog, messagebox  # Dosya diyalogları ve mesaj kutuları için kullanılır
import os  # Dosya ve dizin işlemleri için kullanılır
import joblib  # Modeli kaydetmek ve yüklemek için kullanılır
from paddleocr import PaddleOCR  # OCR işlemleri için PaddleOCR kullanılıyor
import re  # Regüler ifadeler için kullanılır
from db_operations import PlakaTespitDB  # Yeni sınıfı içe aktar

# Bu dosya, plaka tespit modelinin test edilmesi için kullanılır. Test verileri ile modelin doğruluğunu kontrol eder.

class PlakaTespitTest:
    """
    Bu sınıf, plaka tespit sisteminin ana işlevselliğini sağlar.
    Kamera, video ve fotoğraf üzerinde plaka tespiti yapabilir.
    """
    
    def __init__(self):
        """
        Sınıfın başlangıç ayarlarını yapar.
        - Veritabanı bağlantısını kurar
        - YOLO modelini yükler
        - PaddleOCR modelini yükler
        - Arayüzü oluşturur
        """
        # Veritabanı işlemleri için sınıfı başlat
        self.db = PlakaTespitDB()
        
        # Model yükleme
        try:
            self.model = YOLO('plaka_tespit/plaka_model/weights/best.pt')  # YOLO modelini yükler
            print("Model başarıyla yüklendi.")  # Modelin başarıyla yüklendiğini belirtir
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")  # Hata durumunda mesaj gösterir
            return
            
        # PaddleOCR modelini yükleme
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # PaddleOCR modelini yükler
            print("PaddleOCR modeli başarıyla yüklendi.")  # Modelin başarıyla yüklendiğini belirtir
        except Exception as e:
            print(f"PaddleOCR modeli yüklenirken hata oluştu: {e}")  # Hata durumunda mesaj gösterir
            return

        # Son izinli plaka tespitinin zamanını tut
        self.son_izinli_tespit_zamani = 0
        
        # Tkinter penceresi oluştur
        self.root = tk.Tk()  # Tkinter penceresini başlatır
        self.root.title("Plaka Tespit ve Kontrol Sistemi")  # Pencere başlığını ayarlar
        self.root.geometry("500x400")  # Pencere boyutunu ayarlar
        
        # Butonları oluştur
        self.create_widgets()  # Arayüz elemanlarını oluşturur
        
        # Pencereyi göster
        self.root.mainloop()  # Tkinter döngüsünü başlatır
    
    def create_widgets(self):
        """
        Grafiksel arayüzdeki butonları ve diğer öğeleri oluşturur.
        """
        # Başlık
        title = tk.Label(self.root, text="Plaka Tespit ve Kontrol Sistemi", font=("Arial", 16))  # Başlık etiketi oluşturur
        title.pack(pady=20)  # Başlık etiketini yerleştirir
        
        # Kamera butonu
        camera_btn = tk.Button(self.root, text="Kamera ile Test Et", 
                             command=self.kamera_test,
                             width=20, height=2)  # Kamera ile test butonu oluşturur
        camera_btn.pack(pady=10)  # Butonu yerleştirir
        
        # Video butonu
        video_btn = tk.Button(self.root, text="Video ile Test Et",
                            command=self.video_test,
                            width=20, height=2)  # Video ile test butonu oluşturur
        video_btn.pack(pady=10)  # Butonu yerleştirir
        
        # Fotoğraf butonu
        photo_btn = tk.Button(self.root, text="Fotoğraf ile Test Et",
                            command=self.foto_test,
                            width=20, height=2)  # Fotoğraf ile test butonu oluşturur
        photo_btn.pack(pady=10)  # Butonu yerleştirir
        
        # Çıkış butonu
        exit_btn = tk.Button(self.root, text="Çıkış",
                           command=self.root.quit,
                           width=20, height=2)  # Çıkış butonu oluşturur
        exit_btn.pack(pady=10)  # Butonu yerleştirir
    
    def plaka_metni_duzenle(self, metin):
        """
        OCR ile okunan plaka metnini düzenler.
        
        Parametreler:
            metin (str): Düzenlenecek plaka metni
            
        Dönüş:
            str: Düzenlenmiş plaka metni
        """
        # Boşlukları temizle
        metin = metin.strip()
        
        # Başında 3 sayı varsa, 2. ve 3. sayıları al
        if re.match(r'^(\d{3})', metin):
            metin = metin[1:3] + metin[3:]
        
        # Özel karakterleri temizle
        metin = re.sub(r'[^A-Za-z0-9 ]+', '', metin)
        
        # 8 ile başlıyorsa 3 ile değiştir
        # if metin.startswith('8'):
        #     metin = '3' + metin[1:]
        
        # Plaka formatını düzelt
        if len(metin.split()) == 3:
            parcalar = metin.split()
            # Sayıları ve harfleri düzelt
            parcalar[0] = parcalar[0].replace('B', '8').replace('I', '1').replace('O', '0').replace('S', '5').replace('h', '4')
            parcalar[1] = parcalar[1].replace('8', 'B').replace('1', 'I').replace('0', 'O').replace('5', 'S').replace('4', 'H')
            parcalar[2] = parcalar[2].replace('B', '8').replace('I', '1').replace('O', '0').replace('S', '5').replace('h', '4')
            metin = ' '.join(parcalar)
        
        # Uzunluğu kontrol et
        if len(metin.replace(' ', '')) > 8:
            metin = metin[:8]
        else:
            metin = metin[:10]
        
        return metin
    
    def tespit_et(self, frame):
        """
        Görüntü üzerinde plaka tespiti yapar.
        
        Parametreler:
            frame: İşlenecek görüntü
            
        Dönüş:
            tuple: (İşlenmiş görüntü, Tespit edilen plaka sayısı)
        """
        # Tahmin yap
        results = self.model.predict(frame, conf=0.25)  # Görüntüde plaka tespiti yapar
        
        # Sonuçları görüntüleme
        for result in results:
            boxes = result.boxes  # Tespit edilen kutuları alır
            for box in boxes:
                # Koordinatları alma
                x1, y1, x2, y2 = box.xyxy[0]  # Kutunun koordinatlarını alır
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Koordinatları tam sayıya çevirir
                
                # Plaka bölgesini kesme
                plaka_region = frame[y1:y2, x1:x2]  # Plaka bölgesini keser

                # PaddleOCR ile metni okuma
                try:
                    ocr_result = self.ocr.ocr(plaka_region, cls=True)
                    
                    # OCR sonuçlarını kontrol et
                    if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                        plaka_text = ocr_result[0][0][1][0]  # Tespit edilen metni al
                    else:
                        plaka_text = ""
                except Exception as e:
                    print(f"OCR işlemi sırasında hata: {e}")
                    plaka_text = ""

                # Plaka metnini düzenleme
                plaka_text = self.plaka_metni_duzenle(plaka_text)
                plaka_text = plaka_text.replace(' ', '')  # Plaka metnindeki boşlukları kaldır      
                
                # Plaka kontrolü ve kayıt
                if plaka_text and len(plaka_text) >= 4:  # En az 4 karakter varsa işlem yap
                    # İzin kontrolü
                    izin_durumu = self.db.plaka_izin_kontrol(plaka_text)
                    # Veritabanına kaydet
                    plaka_id, self.son_izinli_tespit_zamani = self.db.plaka_kaydet(
                        plaka_text, 
                        izin_durumu, 
                        self.son_izinli_tespit_zamani
                    )
                    
                    # Görüntüye plaka ve durum bilgisini ekle
                    durum_renk = (0, 255, 0) if izin_durumu else (0, 0, 255)  # Yeşil: İzinli, Kırmızı: İzinsiz
                    durum_text = "IZINLI ve GIREBILIR" if izin_durumu else "IZINSIZ ve GIREMEZ"
                    
                    # Kutu çizme
                    cv2.rectangle(frame, (x1, y1), (x2, y2), durum_renk, 2)
                    
                    # Plaka ve durum bilgisini gösterme
                    text_position = (x1-5, y1-10)  # Plaka metninin konumunu ayarlar
                    if text_position[1] < 0:
                        text_position = (x1+15, y1 + 20)  # Eğer metin üstte kalıyorsa, aşağıya kaydır
                    
                    cv2.putText(frame, f'{plaka_text} - {durum_text}', text_position,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, durum_renk, 2)

        # Görüntüyü yeniden boyutlandırma
        frame = cv2.resize(frame, (600, 600))  # Görüntüyü yeniden boyutlandırır
        
        return frame, len(boxes)  # İşlenmiş görüntüyü ve tespit edilen plaka sayısını döner
    
    def kamera_test(self):
        """Kamera ile test"""
        cap = cv2.VideoCapture(0)  # Varsayılan kamerayı açar
        if not cap.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı!")  # Kamera açılamazsa hata mesajı gösterir
            return

        prev_time = time.time()  # Önceki zaman
        while True:
            ret, frame = cap.read()  # Kameradan görüntü alır
            if not ret:
                break

            # FPS hesaplama
            current_time = time.time()  # Şu anki zamanı alır
            fps = 1 / (current_time - prev_time)  # FPS hesaplar
            prev_time = current_time  # Önceki zamanı günceller
            
            # Tespit
            frame, plaka_sayisi = self.tespit_et(frame)  # Görüntüde plaka tespiti yapar
            
            # FPS gösterme
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # FPS değerini görüntüye yazar

            # Görüntüyü göster
            cv2.imshow('Kamera Testi', frame)  # İşlenmiş görüntüyü gösterir

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # 'q' tuşuna basılırsa döngüden çık

        cap.release()  # Kamerayı kapatır
        cv2.destroyAllWindows()  # Tüm pencereleri kapatır
    
    def video_test(self):
        """Video dosyası ile test"""
        # Video dosyası seç
        video_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if not video_path:
            return
            
        cap = cv2.VideoCapture(video_path)  # Video dosyasını açar
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video dosyası açılamadı!")  # Video açılamazsa hata mesajı gösterir
            return

        prev_time = time.time()  # Önceki zaman
        while True:
            ret, frame = cap.read()  # Videodan görüntü alır
            if not ret:
                break

            # FPS hesaplama
            current_time = time.time()  # Şu anki zamanı alır
            fps = 1 / (current_time - prev_time)  # FPS hesaplar
            prev_time = current_time  # Önceki zamanı günceller
            
            # Tespit
            frame, plaka_sayisi = self.tespit_et(frame)  # Görüntüde plaka tespiti yapar
            
            # FPS gösterme
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # FPS değerini görüntüye yazar

            # Görüntüyü göster
            cv2.imshow('Video Testi', frame)  # İşlenmiş görüntüyü gösterir

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # 'q' tuşuna basılırsa döngüden çık

        cap.release()  # Kamerayı kapatır
        cv2.destroyAllWindows()  # Tüm pencereleri kapatır
    
    def foto_test(self):
        """Fotoğraf ile test"""
        # Fotoğraf dosyası seç
        image_path = filedialog.askopenfilename(
            title="Fotoğraf Seç",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not image_path:
            return
            
        # Görüntüyü oku
        image = cv2.imread(image_path)  # Fotoğrafı okur
        if image is None:
            messagebox.showerror("Hata", "Fotoğraf dosyası açılamadı!")  # Fotoğraf açılamazsa hata mesajı gösterir
            return
        
        # Tespit
        image, plaka_sayisi = self.tespit_et(image)  # Görüntüde plaka tespiti yapar
        
        # Sonucu göster
        cv2.imshow('Fotoğraf Testi', image)  # İşlenmiş görüntüyü gösterir
        
        # Sonucu kaydet
        output_dir = "test_sonuclari"  # Sonuçların kaydedileceği dizin
        os.makedirs(output_dir, exist_ok=True)  # Dizin yoksa oluşturur
        output_path = os.path.join(output_dir, f"sonuc_{os.path.basename(image_path)}")  # Çıktı dosya yolunu oluşturur
        cv2.imwrite(output_path, image)  # İşlenmiş görüntüyü kaydeder
        
        cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekler
        cv2.destroyAllWindows()  # Tüm pencereleri kapatır

    def __del__(self):
        """Sınıf yok edildiğinde veritabanı bağlantısını kapat"""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            print("Veritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    app = PlakaTespitTest()  # Uygulamayı başlat 