from ultralytics import YOLO
import cv2
import numpy as np
import os

def test_model():
    """Eğitilmiş modeli test görüntüleri üzerinde test eder"""
    # Modeli yükle
    model_path = "plaka_tespit/plaka_model/weights/best.pt"
    model = YOLO(model_path)
    
    # Test görüntülerinin bulunduğu klasör
    test_dir = "yolov8_dataset/test/images"
    
    # Test görüntülerini listele
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    
    print(f"Toplam {len(test_images)} test görüntüsü bulundu.")
    
    # Her görüntü için tahmin yap
    for img_name in test_images:
        # Görüntüyü oku
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Hata: {img_path} okunamadı!")
            continue
        
        # Tahmin yap
        results = model.predict(image, conf=0.25)  # confidence threshold = 0.25
        
        # Sonuçları görüntü üzerine çiz
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                conf = float(box.conf[0])
                
                # Dikdörtgen ve güven skorunu çiz
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Plaka: {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Sonucu göster
        print(f"\nTest ediliyor: {img_name}")
        print(f"Tespit edilen plaka sayısı: {len(results[0].boxes)}")
        
        # Görüntüyü kaydet
        output_dir = "test_sonuclari"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sonuc_{img_name}")
        cv2.imwrite(output_path, image)
        
        # Görüntüyü göster
        cv2.imshow("Test Sonucu", image)
        key = cv2.waitKey(0)
        
        # 'q' tuşuna basılırsa çık
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("\nTest tamamlandı! Sonuçlar 'test_sonuclari' klasörüne kaydedildi.")

if __name__ == "__main__":
    test_model() 