import psycopg2
import time

class PlakaTespitDB:
    """
    Bu sınıf, plaka tespiti ile ilgili tüm veritabanı işlemlerini yönetir.
    Plaka kaydetme ve izin kontrolü gibi işlemleri gerçekleştirir.
    """
    
    def __init__(self):
        """
        Veritabanı bağlantısını başlatır.
        Bağlantı bilgileri burada ayarlanır.
        """
        try:
            # Veritabanına bağlanma
            self.conn = psycopg2.connect(
                dbname="plaka_tanima_db",  # Veritabanı adı
                user="postgres",           # Kullanıcı adı
                password="4613",           # Şifre
                host="localhost",          # Sunucu adresi
                port="5432"               # Port numarası
            )
            # Veritabanı üzerinde işlem yapmak için cursor oluşturma
            self.cursor = self.conn.cursor()
            print("Veritabanı bağlantısı başarılı.")
        except Exception as e:
            print(f"Veritabanı bağlantı hatası: {e}")
            self.conn = None
            self.cursor = None

    def plaka_izin_kontrol(self, plaka):
        """
        Verilen plakanın izinli olup olmadığını kontrol eder.
        
        Parametreler:
            plaka (str): Kontrol edilecek plaka numarası
            
        Dönüş:
            bool: Plaka izinli ise True, değilse False
        """
        try:
            # Eğer veritabanı bağlantısı yoksa False dön
            if not self.cursor:
                return False
            
            # İzinli plakaları sorgula
            sorgu = """
                SELECT * FROM izinli_plakalar 
                WHERE plaka = %s 
                AND aktif = TRUE 
                AND CURRENT_DATE BETWEEN baslangic_tarih AND bitis_tarih
            """
            # Sorguyu çalıştır
            self.cursor.execute(sorgu, (plaka,))
            sonuc = self.cursor.fetchone()
            
            # Sonuç varsa True, yoksa False dön
            return bool(sonuc)
            
        except Exception as e:
            print(f"Plaka kontrol hatası: {e}")
            return False

    def plaka_kaydet(self, plaka, durum, son_izinli_tespit_zamani):
        """
        Tespit edilen plakayı veritabanına kaydeder.
        
        Parametreler:
            plaka (str): Kaydedilecek plaka numarası
            durum (bool): Plakanın izinli olup olmadığı
            son_izinli_tespit_zamani (float): Son izinli plakanın tespit edildiği zaman
            
        Dönüş:
            tuple: (plaka_id, yeni_son_tespit_zamani) veya (None, son_izinli_tespit_zamani)
        """
        try:
            # Veritabanı bağlantısı yoksa işlem yapma
            if not self.cursor:
                return None, son_izinli_tespit_zamani
            
            # Şu anki zamanı al
            simdiki_zaman = time.time()
            
            # Son izinli tespitten bu yana 15 saniye geçmediyse kayıt yapma
            if simdiki_zaman - son_izinli_tespit_zamani < 15:
                print("Son izinli tespitten 15 saniye geçmedi, kayıt yapılmıyor.")
                return None, son_izinli_tespit_zamani
            
            # Plakayı veritabanına kaydet
            sorgu = "INSERT INTO plakalar (plaka, durum) VALUES (%s, %s) RETURNING id;"
            self.cursor.execute(sorgu, (plaka, durum))
            plaka_id = self.cursor.fetchone()[0]
            self.conn.commit()
            
            # Eğer izinli plaka ise son tespit zamanını güncelle
            yeni_son_tespit_zamani = simdiki_zaman if durum else son_izinli_tespit_zamani
            
            if durum:
                print(f"İzinli plaka tespit edildi. Sonraki 15 saniye kayıt yapılmayacak.")
            
            print(f"Plaka başarıyla kaydedildi. ID: {plaka_id}")
            return plaka_id, yeni_son_tespit_zamani
            
        except Exception as e:
            # Hata durumunda işlemi geri al
            if self.conn:
                self.conn.rollback()
            print(f"Plaka kaydedilirken hata oluştu: {e}")
            return None, son_izinli_tespit_zamani

    def __del__(self):
        """
        Sınıf silindiğinde veritabanı bağlantısını düzgün bir şekilde kapatır.
        """
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            print("Veritabanı bağlantısı kapatıldı.") 