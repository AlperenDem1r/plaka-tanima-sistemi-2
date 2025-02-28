import cv2
from ultralytics import YOLO
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import joblib
import pytesseract
import re

class PlakaTespitTest:
    def __init__(self):
        # Model yükleme
        try:
            self.model = YOLO('plaka_tespit/plaka_model/weights/best.pt')
            print("Model başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}")
            return

        # Random Forest modelini yükleme
        self.model_rf = joblib.load('random_forest_model.pkl')

        # Tkinter penceresi oluştur
        self.root = tk.Tk()
        self.root.title("Plaka Tespit Test")
        self.root.geometry("400x300")
        
        # Butonları oluştur
        self.create_widgets()
        
        # Pencereyi göster
        self.root.mainloop()
    
    def create_widgets(self):
        """Arayüz elemanlarını oluştur"""
        # Başlık
        title = tk.Label(self.root, text="Plaka Tespit Sistemi", font=("Arial", 16))
        title.pack(pady=20)
        
        # Kamera butonu
        camera_btn = tk.Button(self.root, text="Kamera ile Test Et", 
                             command=self.kamera_test,
                             width=20, height=2)
        camera_btn.pack(pady=10)
        
        # Video butonu
        video_btn = tk.Button(self.root, text="Video ile Test Et",
                            command=self.video_test,
                            width=20, height=2)
        video_btn.pack(pady=10)
        
        # Fotoğraf butonu
        photo_btn = tk.Button(self.root, text="Fotoğraf ile Test Et",
                            command=self.foto_test,
                            width=20, height=2)
        photo_btn.pack(pady=10)
        
        # Çıkış butonu
        exit_btn = tk.Button(self.root, text="Çıkış",
                           command=self.root.quit,
                           width=20, height=2)
        exit_btn.pack(pady=10)
    
    def tespit_et(self, frame):
        """Görüntü üzerinde plaka tespiti yapar"""
        # Tahmin yap
        results = self.model.predict(frame, conf=0.25)
        
        # Sonuçları görüntüleme
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Koordinatları alma
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                
                # Plaka bölgesini kesme
                plaka_region = frame[y1:y2, x1:x2]

                # OCR ile metni okuma
                plaka_text = pytesseract.image_to_string(plaka_region, config='--psm 8')

                # Plaka metnini düzenleme
                plaka_text = plaka_text.strip()

                # Başında 3 sayı varsa, 2. ve 3. sayıları al
                if re.match(r'^(\d{3})', plaka_text):
                    plaka_text = plaka_text[1:3] + plaka_text[3:]  # 2. ve 3. sayıları al

                # Özel karakterleri temizleme
                plaka_text = re.sub(r'[^A-Za-z0-9 ]+', '', plaka_text)

                # Plaka metnini kontrol etme
                if plaka_text.startswith('8'):
                    plaka_text = '3' + plaka_text[1:]

                # Plaka metnini gösterme
                text_position = (x1-5, y1-10)
                if text_position[1] < 0:
                    text_position = (x1+15, y1 + 20)  # Eğer metin üstte kalıyorsa, aşağıya kaydır
                # Plaka metnini kontrol etme ve düzeltme
                if len(plaka_text.split()) == 3:  # Eğer plaka metni 3 parçadan oluşuyorsa
                    parts = plaka_text.split()
                    # 1. bölümdeki sayıları kontrol et
                    parts[0] = parts[0].replace('B', '8').replace('I', '1').replace('O', '0').replace('S', '5').replace('h', '4')
                    # 2. bölümdeki harfleri kontrol et
                    parts[1] = parts[1].replace('8', 'B').replace('1', 'I').replace('0', 'O').replace('5', 'S').replace('4', 'H')
                    # 3. bölümdeki sayıları kontrol et
                    parts[2] = parts[2].replace('B', '8').replace('I', '1').replace('O', '0').replace('S', '5').replace('h', '4')
                    # Düzgün plaka metnini birleştir
                    plaka_text = ' '.join(parts)

                # Plaka metnini boşluk karakteri hariç 8 karakterden azsa, 10 karakter ile sınırlama
                if len(plaka_text.replace(' ', '')) > 8:
                    plaka_text = plaka_text[:8]
                else:
                    plaka_text = plaka_text[:10]


                # Kutu çizme
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # Güven skorunu gösterme
                cv2.putText(frame, f'{plaka_text}', text_position,
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)
                # Plaka metnindeki boşlukları kaldır
                plaka_text = plaka_text.replace(' ', '')
        # Görüntüyü yeniden boyutlandırma
        frame = cv2.resize(frame, (600, 600))
        
        return frame, len(boxes)
    
    def kamera_test(self):
        """Kamera ile test"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı!")
            return

        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS hesaplama
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Tespit
            frame, plaka_sayisi = self.tespit_et(frame)
            
            # FPS gösterme
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü göster
            cv2.imshow('Kamera Testi', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def video_test(self):
        """Video dosyası ile test"""
        # Video dosyası seç
        video_path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if not video_path:
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Hata", "Video dosyası açılamadı!")
            return

        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS hesaplama
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Tespit
            frame, plaka_sayisi = self.tespit_et(frame)
            
            # FPS gösterme
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Görüntüyü göster
            cv2.imshow('Video Testi', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
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
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Hata", "Fotoğraf dosyası açılamadı!")
            return
        
        # Tespit
        image, plaka_sayisi = self.tespit_et(image)
        
        # Sonucu göster
        cv2.imshow('Fotoğraf Testi', image)
        
        # Sonucu kaydet
        output_dir = "test_sonuclari"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sonuc_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, image)
        
        # messagebox.showinfo("Bilgi", 
        #                   f"Tespit edilen plaka sayısı: {plaka_sayisi}\n"
        #                   f"Sonuç kaydedildi: {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PlakaTespitTest() 