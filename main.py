import argparse
import sys
import time
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO ile Gerçek Zamanlı Webcam Nesne Tespiti"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Kullanılacak YOLO model ağırlığı (Örn: yolo11n.pt, yolov8n.pt veya özel .pt dosyası)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam cihaz indeksi (Varsayılan: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Güven eşiği (Confidence threshold, Varsayılan: 0.45)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Kamera genişlik çözünürlüğü (Varsayılan: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Kamera yükseklik çözünürlüğü (Varsayılan: 720)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("🚀 YOLO Gerçek Zamanlı Webcam Nesne Tespiti Başlatılıyor")
    print(f"📦 Model: {args.model}")
    print(f"📷 Kamera İndeksi: {args.camera}")
    print(f"🎯 Güven Eşiği: {args.conf}")
    print("=" * 60)

    # 1. YOLO Modelini Yükle
    try:
        print(f"[*] Model yükleniyor ({args.model})...")
        model = YOLO(args.model)
        print("[+] Model başarıyla yüklendi!")
    except Exception as e:
        print(f"[!] Model yüklenirken hata oluştu: {e}")
        print("[*] 'yolov8n.pt' modeli ile deneniyor...")
        try:
            model = YOLO("yolov8n.pt")
            print("[+] yolov8n.pt başarıyla yüklendi!")
        except Exception as e2:
            print(f"[!] Yedek model de yüklenemedi: {e2}")
            sys.exit(1)

    # 2. Kamerayı Başlat
    print(f"[*] Kamera ({args.camera}) açılıyor...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[!] HATA: Kamera {args.camera} açılamadı! Lütfen kamera izinlerini ve bağlantısını kontrol edin.")
        sys.exit(1)

    print("[+] Kamera açıldı! Çıkmak için pencere açıkken 'q' tuşuna basın.")
    print("-" * 60)

    prev_time = time.time()
    last_log_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Kameradan görüntü alınamadı, çıkılıyor...")
                break

            # FPS Hesaplama
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # 3. Model Tahmini (Inference)
            results = model.predict(source=frame, conf=args.conf, verbose=False)
            result = results[0]

            # 4. Tespit Edilen Nesneleri Çıkar ve Terminale Yazdır
            detected_items = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf_score = float(box.conf[0].item())
                    cls_name = model.names.get(cls_id, f"Class {cls_id}")
                    detected_items.append((cls_name, conf_score))

            # Terminal çıktısı (her 0.5 saniyede bir veya tespit varsa güncelle)
            if curr_time - last_log_time >= 0.5:
                if detected_items:
                    # Nesneleri özetle: "person (2), cell phone (1)"
                    summary = {}
                    for name, _ in detected_items:
                        summary[name] = summary.get(name, 0) + 1
                    summary_str = ", ".join([f"{k} x{v}" for k, v in summary.items()])
                    print(f"[{time.strftime('%H:%M:%S')}] 🔍 Tespit Edilenler ({len(detected_items)}): {summary_str} | FPS: {fps:.1f}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] ⚪ Nesne tespit edilmedi | FPS: {fps:.1f}")
                last_log_time = curr_time

            # 5. Görselleştirme (Annotated Frame)
            annotated_frame = result.plot()

            # Bilgi Paneli (Overlay)
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 80), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | Obje Sayisi: {len(detected_items)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                "Cikis icin 'q' tusuna basin",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Pencereyi göster
            cv2.imshow("YOLO Nesne Tespiti - Webcam", annotated_frame)

            # Tuş kontrolü ('q' veya ESC ile çıkış)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[*] Kullanıcı çıkış yaptı.")
                break

    except KeyboardInterrupt:
        print("\n[*] Program kullanıcı tarafından durduruldu (Ctrl+C).")
    finally:
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Kamera ve pencereler kapatıldı. Görüşmek üzere!")


if __name__ == "__main__":
    main()
