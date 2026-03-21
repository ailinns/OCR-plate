"""
ตรวจสอบ best.pt — บอก version และ class names
รัน: python check_model.py best.pt
"""
import sys, pathlib

path = sys.argv[1] if len(sys.argv) > 1 else "vehicle_detector.pt"

if not pathlib.Path(path).exists():
    print(f"[ERROR] ไม่เจอไฟล์: {path}")
    sys.exit(1)

print(f"Checking: {path}")
print("─" * 40)

# ลอง YOLOv8 ก่อน
try:
    from ultralytics import YOLO
    m = YOLO(path)
    names = m.names  # dict {0: 'plate', ...}
    print(f"Framework : YOLOv8 (ultralytics)")
    print(f"Classes   : {names}")
    print(f"Task      : {m.task}")
    print()
    print("► ใช้ใน plate_detector.py:")
    print(f'  MODEL_TYPE = "yolov8"')
    first_class = list(names.values())[0] if names else "plate"
    print(f'  # class name = "{first_class}"')
    sys.exit(0)
except Exception as e1:
    print(f"[YOLOv8] ไม่ได้ → {e1}")

# ลอง YOLOv5
try:
    import torch
    m = torch.hub.load("ultralytics/yolov5", "custom",
                       path=path, force_reload=False, verbose=False)
    names = m.names
    print(f"Framework : YOLOv5")
    print(f"Classes   : {names}")
    print()
    print("► ใช้ใน plate_detector.py:")
    print(f'  MODEL_TYPE = "yolov5"')
    sys.exit(0)
except Exception as e2:
    print(f"[YOLOv5] ไม่ได้ → {e2}")

print("\n[!] ไม่สามารถโหลดได้ทั้ง v8 และ v5")
print("    ลองติดตั้ง: pip install ultralytics")