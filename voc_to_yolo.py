import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

def convert_voc_to_yolo(xml_file, image_width, image_height):
    """VOC formatındaki XML'i YOLO formatına dönüştürür"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        # Sınıf adı - bizim durumumuzda sadece 'plaka' var (0 index'li)
        class_id = 0
        
        # Bounding box koordinatları
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # YOLO formatına dönüştürme
        # YOLO formatı: <class_id> <x_center> <y_center> <width> <height>
        # Tüm değerler normalize edilmiş (0-1 arası)
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        # 0-1 arasına sınırlama
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        width = min(max(width, 0), 1)
        height = min(max(height, 0), 1)
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines

def process_dataset(voc_dir, output_dir):
    """Tüm veri setini işler"""
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # classes.txt dosyasını oluştur
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('plaka\n')
    
    # XML dosyalarını listele
    xml_files = list(Path(voc_dir).glob('*.xml'))
    print(f"Toplam {len(xml_files)} XML dosyası bulundu.")
    
    for xml_file in tqdm(xml_files, desc="XML dosyaları dönüştürülüyor"):
        try:
            # XML'i parse et
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Görüntü boyutlarını al
            size = root.find('size')
            image_width = int(size.find('width').text)
            image_height = int(size.find('height').text)
            
            # YOLO formatına dönüştür
            yolo_lines = convert_voc_to_yolo(xml_file, image_width, image_height)
            
            # YOLO dosyasını kaydet
            output_file = os.path.join(output_dir, xml_file.stem + '.txt')
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
                
        except Exception as e:
            print(f"Hata: {xml_file.name} dosyası işlenirken hata oluştu: {str(e)}")

if __name__ == "__main__":
    # Klasör yolları
    voc_dir = "dataset/labels"  # XML dosyalarının bulunduğu klasör
    output_dir = "dataset/labels_yolo"  # YOLO formatında etiketlerin kaydedileceği klasör
    
    # Dönüşümü başlat
    process_dataset(voc_dir, output_dir)
    print("Dönüşüm tamamlandı!") 