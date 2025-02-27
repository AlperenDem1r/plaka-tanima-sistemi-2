from ultralytics import YOLO
import os

# Model indirme yolu
os.environ['YOLO_CONFIG_DIR'] = os.path.join(os.getcwd(), 'yolo_config')

# Model
model = YOLO('yolov8n.pt')  # yolov8n.pt küçük model, daha hızlı eğitim

# Eğitim
results = model.train(
    data=os.path.join(os.getcwd(), 'plaka.yaml'),  # yaml dosyasının tam yolu
    epochs=50,            # epoch sayısı
    imgsz=640,            # görüntü boyutu
    batch=16,             # batch size
    patience=50,          # early stopping
    device='cpu',         # eğitim cihazı (GPU varsa 'cuda' yapın)
    project='plaka_tespit',  # proje adı
    name='plaka_model',   # model adı
    exist_ok=True,        # var olan klasörün üzerine yaz
    cache=False           # önbelleği devre dışı bırak
) 