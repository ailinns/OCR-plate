"""
License Plate Detection Pipeline
OpenCV → AI Detection → ขึงภาพ (Perspective Warp) → EasyOCR → Output

Requirements:
    pip install opencv-python easyocr numpy imutils
"""

import cv2
import numpy as np
import easyocr
import imutils
import time
import sys
from pathlib import Path


# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "vehicle_detector.pt"   # ← เปลี่ยนจาก best.pt
MODEL_TYPE = "yolov8"                # ← ถูกอยู่แล้ว ไม่ต้องแก้
CAMERA_INDEX   = 0          # 0 = default webcam
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
MIN_PLATE_AREA = 1500       # px² — กรองพื้นที่เล็กเกินไปออก
ASPECT_RANGE   = (1.5, 6.0) # aspect ratio ที่ยอมรับสำหรับป้ายทะเบียน
SAVE_OUTPUT    = True       # บันทึกภาพ output
OCR_LANG       = ["th", "en"]


# ─── Stage 1: Detect plate region (contour-based) ────────────────────────────
def detect_plate_contour(frame: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """
    Returns:
        plate_region  – cropped grayscale plate (or None)
        bbox          – (x, y, w, h) in original frame (or None)
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged   = cv2.Canny(blurred, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for c in contours:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area        = w * h
            aspect      = w / float(h)

            if area < MIN_PLATE_AREA:
                continue
            if not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
                continue

            plate_gray = gray[y:y+h, x:x+w]
            return plate_gray, (x, y, w, h)

    return None, None


# ─── Stage 2: Perspective warp (ขึงภาพ) ──────────────────────────────────────
def warp_plate(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crops the plate from bbox and applies perspective transform
    เพื่อให้ได้ภาพตรง (frontal view)
    """
    x, y, w, h = bbox
    roi         = frame[y:y+h, x:x+w]
    gray_roi    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh   = cv2.threshold(gray_roi, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scale up เพื่อให้ EasyOCR อ่านได้ดีขึ้น
    scale      = max(1.0, 300.0 / w)
    new_w      = int(w * scale)
    new_h      = int(h * scale)
    warped     = cv2.resize(thresh, (new_w, new_h),
                            interpolation=cv2.INTER_CUBIC)

    # เพิ่ม padding รอบป้าย
    pad        = 8
    warped     = cv2.copyMakeBorder(warped, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=255)
    return warped


# ─── Stage 3: EasyOCR ─────────────────────────────────────────────────────────
def read_plate(reader: easyocr.Reader, plate_img: np.ndarray) -> list[dict]:
    """
    Returns list of { text, confidence } sorted by confidence desc
    """
    results = reader.readtext(plate_img, detail=1,
                               paragraph=False,
                               allowlist=None)
    parsed  = []
    for (_, text, conf) in results:
        text_clean = text.strip()
        if text_clean:
            parsed.append({"text": text_clean, "confidence": round(conf * 100, 1)})

    return sorted(parsed, key=lambda r: r["confidence"], reverse=True)


# ─── Stage 4: Draw overlay ───────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray,
                 bbox:  tuple,
                 ocr_results: list[dict]) -> np.ndarray:
    out   = frame.copy()
    x, y, w, h = bbox

    # Bounding box สีเขียว
    cv2.rectangle(out, (x, y), (x+w, y+h), (29, 200, 120), 2)

    # Corner markers
    cs = min(20, w // 4, h // 2)
    corners = [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]
    dirs    = [(1,1), (-1,1), (1,-1), (-1,-1)]
    for (cx, cy), (dx, dy) in zip(corners, dirs):
        cv2.line(out, (cx, cy), (cx + dx*cs, cy), (93, 202, 163), 3)
        cv2.line(out, (cx, cy), (cx, cy + dy*cs), (93, 202, 163), 3)

    # Text สถานะ
    plate_text = "  ".join(r["text"] for r in ocr_results[:3]) or "—"
    conf_val   = ocr_results[0]["confidence"] if ocr_results else 0

    bg_y1 = max(0, y - 36)
    cv2.rectangle(out, (x, bg_y1), (x + w, y), (20, 20, 20), -1)

    cv2.putText(out, plate_text, (x + 5, y - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"conf {conf_val:.0f}%", (x + 5, y - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (93, 202, 163), 1)

    # Pipeline label (มุมบนซ้าย)
    labels = ["OpenCV", "Detection", "Warp", "EasyOCR", "Done"]
    for i, label in enumerate(labels):
        color = (255, 255, 255) if i < 4 else (80, 220, 140)
        cv2.putText(out, f"[{i+1}] {label}",
                    (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    return out


# ─── Main pipeline ────────────────────────────────────────────────────────────
def process_image(image_path: str, reader: easyocr.Reader) -> None:
    """ใช้กับ image file"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    frame = imutils.resize(frame, width=FRAME_WIDTH)
    run_frame(frame, reader, save_prefix=Path(image_path).stem)


def run_frame(frame: np.ndarray,
              reader: easyocr.Reader,
              save_prefix: str = "plate") -> dict | None:
    """
    รัน full pipeline บน 1 frame
    คืน dict ผลลัพธ์ หรือ None ถ้าไม่เจอป้าย
    """
    t0 = time.perf_counter()
    print("\n" + "─"*50)
    print("► Stage 1 │ OpenCV — frame analysis")

    # ── Detection ───────────────────────────────
    print("► Stage 2 │ AI Detection — finding plate contour")
    plate_gray, bbox = detect_plate_contour(frame)

    if plate_gray is None:
        print("  [!] No plate found in frame")
        return None

    x, y, w, h = bbox
    print(f"  Plate found: x={x} y={y} w={w} h={h}  area={w*h}px²")

    # ── Warp ────────────────────────────────────
    print("► Stage 3 │ ขึงภาพ — perspective warp & crop")
    warped = warp_plate(frame, bbox)
    print(f"  Warped size: {warped.shape[1]}×{warped.shape[0]}px")

    # ── EasyOCR ─────────────────────────────────
    print("► Stage 4 │ EasyOCR — reading characters")
    ocr_results = read_plate(reader, warped)

    if not ocr_results:
        print("  [!] OCR: no text detected")
    else:
        for r in ocr_results:
            print(f"  Text: {r['text']!r:20s}  conf: {r['confidence']}%")

    # ── Output ──────────────────────────────────
    print("► Stage 5 │ Output")
    output_frame = draw_overlay(frame, bbox, ocr_results)

    elapsed = time.perf_counter() - t0
    print(f"  Pipeline time: {elapsed*1000:.0f} ms")
    print("─"*50)

    if SAVE_OUTPUT:
        out_path      = f"output_{save_prefix}.jpg"
        warped_path   = f"output_{save_prefix}_plate.jpg"
        cv2.imwrite(out_path, output_frame)
        cv2.imwrite(warped_path, warped)
        print(f"  Saved: {out_path}")
        print(f"  Saved: {warped_path}")

    # แสดงหน้าต่าง
    cv2.imshow("License Plate Detector — full frame", output_frame)
    cv2.imshow("Cropped Plate (EasyOCR input)", warped)

    result = {
        "bbox":        bbox,
        "ocr":         ocr_results,
        "elapsed_ms":  round(elapsed * 1000),
        "best_text":   ocr_results[0]["text"] if ocr_results else None,
        "confidence":  ocr_results[0]["confidence"] if ocr_results else 0,
    }
    return result


def run_camera(reader: easyocr.Reader) -> None:
    """Real-time camera mode"""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    print(f"Camera opened ({FRAME_WIDTH}×{FRAME_HEIGHT})")
    print("Keys: [SPACE] capture & run pipeline   [Q] quit")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed")
            break

        # แสดง live feed พร้อม instruction
        display = frame.copy()
        cv2.putText(display, "SPACE: detect   Q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow("Camera — License Plate Detector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            frame_count += 1
            run_frame(frame, reader, save_prefix=f"cam_{frame_count}")

    cap.release()
    cv2.destroyAllWindows()


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  License Plate Detector")
    print("  OpenCV → Detection → Warp → EasyOCR")
    print("=" * 50)

    print("\nLoading EasyOCR reader (th + en)...")
    reader = easyocr.Reader(OCR_LANG, gpu=False)
    print("EasyOCR ready.\n")

    if len(sys.argv) > 1:
        # โหมดรูปภาพ: python plate_detector.py image.jpg
        image_path = sys.argv[1]
        print(f"Image mode: {image_path}")
        process_image(image_path, reader)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # โหมดกล้อง
        print("Camera mode (no image path given)")
        run_camera(reader)