import cv2  # OpenCV kütüphanesini görüntü işleme için kullanır
from ultralytics import YOLO  # YOLO modelini kullanmak için gerekli kütüphane
import numpy as np  # Numpy kütüphanesini matematiksel işlemler için kullanır
import time  # Zaman işlemleri için kullanılır
import tkinter as tk  # Tkinter kütüphanesini GUI oluşturmak için kullanır
from tkinter import filedialog, messagebox  # Dosya diyalogları ve mesaj kutuları için kullanılır
import os  # Dosya ve dizin işlemleri için kullanılır
import joblib  # Modeli kaydetmek ve yüklemek için kullanılır
import pytesseract  # OCR işlemleri için kullanılır
import re  # Regüler ifadeler için kullanılır

# Bu dosya, plaka tespit modelinin test edilmesi için kullanılır. Test verileri ile modelin doğruluğunu kontrol eder.

class PlakaTespitTest:
    def __init__(self):
        # Model yükleme
        try:
            self.model = YOLO('plaka_tespit/plaka_model/weights/best.pt')  # YOLO modelini yükler
            print("Model başarıyla yüklendi.")  # Modelin başarıyla yüklendiğini belirtir
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")  # Hata durumunda mesaj gösterir
            return

        # Random Forest modelini yükleme
        self.model_rf = joblib.load('random_forest_model.pkl')  # Eğitilmiş Random Forest modelini yükler

        # Tkinter penceresi oluştur
        self.root = tk.Tk()  # Tkinter penceresini başlatır
        self.root.title("Plaka Tespit Test")  # Pencere başlığını ayarlar
        self.root.geometry("400x300")  # Pencere boyutunu ayarlar
        
        # Butonları oluştur
        self.create_widgets()  # Arayüz elemanlarını oluşturur
        
        # Pencereyi göster
        self.root.mainloop()  # Tkinter döngüsünü başlatır
    
    def create_widgets(self):
        """Arayüz elemanlarını oluştur"""
        # Başlık
        title = tk.Label(self.root, text="Plaka Tespit Sistemi", font=("Arial", 16))  # Başlık etiketi oluşturur
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
    
    def tespit_et(self, frame):
        """Görüntü üzerinde plaka tespiti yapar"""
        # Tahmin yap
        results = self.model.predict(frame, conf=0.25)  # Görüntüde plaka tespiti yapar
        
        # Sonuçları görüntüleme
        for result in results:
            boxes = result.boxes  # Tespit edilen kutuları alır
            for box in boxes:
                # Koordinatları alma
                x1, y1, x2, y2 = box.xyxy[0]  # Kutunun koordinatlarını alır
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Koordinatları tam sayıya çevirir
                conf = float(box.conf[0])  # Güven skorunu alır
                
                # Plaka bölgesini kesme
                plaka_region = frame[y1:y2, x1:x2]  # Plaka bölgesini keser

                # OCR ile metni okuma
                plaka_text = pytesseract.image_to_string(plaka_region, config='--psm 8')  # Plaka metnini okur

                # Plaka metnini düzenleme
                plaka_text = plaka_text.strip()  # Plaka metnindeki boşlukları temizler

                # Başında 3 sayı varsa, 2. ve 3. sayıları al
                if re.match(r'^(\d{3})', plaka_text):
                    plaka_text = plaka_text[1:3] + plaka_text[3:]  # 2. ve 3. sayıları al

                # Özel karakterleri temizleme
                plaka_text = re.sub(r'[^A-Za-z0-9 ]+', '', plaka_text)

                # Plaka metnini kontrol etme
                if plaka_text.startswith('8'):
                    plaka_text = '3' + plaka_text[1:]  # Plaka metnini düzeltir

                # Plaka metnini gösterme
                text_position = (x1-5, y1-10)  # Plaka metninin konumunu ayarlar
                if text_position[1] < 0:
                    text_position = (x1+15, y1 + 20)  # Eğer metin üstte kalıyorsa, aşağıya kaydır
                # Plaka metnini kontrol etme ve düzeltme
                if len(plaka_text.split()) == 3:  # Eğer plaka metni 3 parçadan oluşuyorsa
                    parts = plaka_text.split()  # Plaka metnini parçalara ayırır
                    # 1. bölümdeki sayıları kontrol et
                    parts[0] = parts[0].replace('B', '8').replace('I', '1').replace('O', '0').replace('S', '5').replace('h', '4')  # Karakter düzeltmeleri yapar
                    # 2. bölümdeki harfleri kontrol et
                    parts[1] = parts[1].replace('8', 'B').replace('1', 'I').replace('0', 'O').replace('5', 'S').replace('4', 'H')  # Karakter düzeltmeleri yapar
                    # 3. bölümdeki sayıları kontrol et
                    parts[2] = parts[2].replace('B', '8').replace('I', '1').replace('O', '0').replace('S', '5').replace('h', '4')  # Karakter düzeltmeleri yapar
                    # Düzgün plaka metnini birleştir
                    plaka_text = ' '.join(parts)  # Plaka metnini birleştirir

                # Plaka metnini boşluk karakteri hariç 8 karakterden azsa, 10 karakter ile sınırlama
                if len(plaka_text.replace(' ', '')) > 8:
                    plaka_text = plaka_text[:8]  # Plaka metnini 8 karakterle sınırlar
                else:
                    plaka_text = plaka_text[:10]  # Plaka metnini 10 karakterle sınırlar

                # Kutu çizme
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Tespit edilen plaka bölgesinin etrafına kutu çizer
                # Güven skorunu gösterme
                cv2.putText(frame, f'{plaka_text}', text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)  # Plaka metnini görüntüye yazar
                # Plaka metnindeki boşlukları kaldır
                plaka_text = plaka_text.replace(' ', '')  # Plaka metnindeki boşlukları kaldır
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
        
        # messagebox.showinfo("Bilgi", 
        #                   f"Tespit edilen plaka sayısı: {plaka_sayisi}\n"
        #                   f"Sonuç kaydedildi: {output_path}")
        
        cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekler
        cv2.destroyAllWindows()  # Tüm pencereleri kapatır

if __name__ == "__main__":
    app = PlakaTespitTest()  # Uygulamayı başlat 