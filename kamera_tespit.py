import cv2
from ultralytics import YOLO
import numpy as np
import time

def kamera_tespit():
    # Model yükleme
    try:
        model = YOLO('plaka_tespit/plaka_model/weights/best.pt')
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return

    # Kamera açma
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı!")
            break

        # FPS hesaplama
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Tahmin yapma
        results = model.predict(frame, conf=0.25)
        
        # Sonuçları görüntüleme
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Koordinatları alma
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                
                # Kutu çizme
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Güven skorunu gösterme
                cv2.putText(frame, f'Plaka: {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # FPS gösterme
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Görüntüyü gösterme
        cv2.imshow('Plaka Tespiti', frame)

        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları temizle
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    kamera_tespit() 