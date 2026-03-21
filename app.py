"""
License Plate Detection — Streamlit App
OpenCV → YOLO (vehicle_detector.pt) → ขึงภาพ → EasyOCR → Output
"""

import streamlit as st
import cv2
import numpy as np
import easyocr
import time
import csv
import os
import ssl
import urllib.request
from ultralytics import YOLO
from difflib import get_close_matches

# ─── SSL Fix ─────────────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context
urllib.request.install_opener(
    urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl._create_unverified_context())
    )
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sarabun:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
.main, [data-testid="stAppViewContainer"] { background: #0a0c10; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2433; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e2e8f0 !important; }
.plate-card {
    background: #0f1117; border: 1px solid #1e2433;
    border-radius: 12px; padding: 20px 24px; margin: 12px 0;
}
.plate-text {
    font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem;
    font-weight: 600; letter-spacing: 6px; color: #34d399;
    text-align: center; padding: 14px;
    background: #052e16; border: 1px solid #065f46;
    border-radius: 8px; margin: 8px 0;
}
.province-badge { font-size: 1rem; color: #6ee7b7; text-align: center; margin-top: 6px; }
.tag { display:inline-block; font-family:'IBM Plex Mono',monospace;
       font-size:0.7rem; padding:3px 10px; border-radius:20px; margin:2px; }
.tag-car  { background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb; }
.tag-moto { background:#3b1f00; color:#fbbf24; border:1px solid #d97706; }
.tag-conf { background:#1a1a2e; color:#a78bfa; border:1px solid #7c3aed; }
</style>
""", unsafe_allow_html=True)


# ─── ครบ 77 จังหวัด ──────────────────────────────────────────────────────────
ALL_PROVINCES = [
    "กรุงเทพมหานคร","กระบี่","กาญจนบุรี","กาฬสินธุ์","กำแพงเพชร",
    "ขอนแก่น","จันทบุรี","ฉะเชิงเทรา","ชลบุรี","ชัยนาท","ชัยภูมิ",
    "ชุมพร","เชียงราย","เชียงใหม่","ตรัง","ตราด","ตาก","นครนายก",
    "นครปฐม","นครพนม","นครราชสีมา","นครศรีธรรมราช","นครสวรรค์",
    "นนทบุรี","นราธิวาส","น่าน","บึงกาฬ","บุรีรัมย์","ปทุมธานี",
    "ประจวบคีรีขันธ์","ปราจีนบุรี","ปัตตานี","พระนครศรีอยุธยา",
    "พะเยา","พังงา","พัทลุง","พิจิตร","พิษณุโลก","เพชรบุรี","เพชรบูรณ์",
    "แพร่","ภูเก็ต","มหาสารคาม","มุกดาหาร","แม่ฮ่องสอน","ยโสธร",
    "ยะลา","ร้อยเอ็ด","ระนอง","ระยอง","ราชบุรี","ลพบุรี","ลำปาง",
    "ลำพูน","เลย","ศรีสะเกษ","สกลนคร","สงขลา","สตูล","สมุทรปราการ",
    "สมุทรสงคราม","สมุทรสาคร","สระแก้ว","สระบุรี","สิงห์บุรี","สุโขทัย",
    "สุพรรณบุรี","สุราษฎร์ธานี","สุรินทร์","หนองคาย","หนองบัวลำภู",
    "อ่างทอง","อำนาจเจริญ","อุดรธานี","อุตรดิตถ์","อุทัยธานี","อุบลราชธานี",
]

def match_province(text: str) -> str:
    if not text.strip():
        return "—"
    m = get_close_matches(text.strip(), ALL_PROVINCES, n=1, cutoff=0.5)
    return m[0] if m else text.strip()


# ─── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(path: str):
    reader    = easyocr.Reader(["th", "en"], gpu=False)
    reader_en = easyocr.Reader(["en"],        gpu=False)  # อ่านตัวเลขแม่นกว่า
    model     = YOLO(path)
    return reader, reader_en, model


# ============================================================
# ขึงภาพ (Rectify) — ตรงตาม vehicle_detector.py
# YOLO bbox → pad_box → crop → GrabCut → Quad → Perspective Warp → Deskew
# ============================================================

OUT_W         = 400
PAD_RATIO     = 0.30   # ขยาย bbox ก่อน crop เผื่อ GrabCut เห็นขอบ
GC_MARGIN     = 0.05
GC_ITERS      = 3
EXPAND        = 1.04
MAX_DESKEW_DEG = 20.0

ASPECT_BY_NAME = {
    "car-license-plate":              2.8,
    "motorcycle-license-plate":       1.0,
    "license plate car":              2.8,
    "license plate motorcycle":       1.0,
    "license plate car - motorcycle": 2.8,
}

def _norm_name(s: str) -> str:
    return (s or "").strip().lower()

def _pad_box(x1, y1, x2, y2, W, H, pad=PAD_RATIO):
    bw, bh = x2-x1, y2-y1
    px, py = int(bw*pad), int(bh*pad)
    return (max(0,x1-px), max(0,y1-py), min(W-1,x2+px), min(H-1,y2+py))

def order_points(pts: np.ndarray) -> np.ndarray:
    """เรียง 4 มุม: tl, tr, br, bl"""
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect

def expand_quad(quad: np.ndarray, factor: float, W: int, H: int) -> np.ndarray:
    c = quad.mean(axis=0, keepdims=True)
    q = (quad - c) * factor + c
    q[:,0] = np.clip(q[:,0], 0, W-1)
    q[:,1] = np.clip(q[:,1], 0, H-1)
    return q

def warp_fixed_aspect(img: np.ndarray, quad4: np.ndarray,
                      out_w: int, aspect: float) -> np.ndarray:
    """Perspective transform ให้ได้ขนาด out_w × (out_w/aspect)"""
    quad  = order_points(quad4.astype("float32"))
    out_h = max(60, int(out_w / aspect))
    dst   = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype="float32")
    M     = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))

def rotate_keep_size(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """หมุนแก้เอียง — คงขนาดเดิม ไม่ crop ไม่ zoom"""
    if abs(angle_deg) < 0.15:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def estimate_skew_angle_deg(img_bgr: np.ndarray) -> float:
    """หาองศาเอียงจากเส้นแนวนอน (Hough Lines) — คืนค่ามัธยฐาน"""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=int(0.35*img_bgr.shape[1]),
                            maxLineGap=25)
    if lines is None:
        return 0.0
    angs = []
    for x1,y1,x2,y2 in lines.reshape(-1,4):
        dx, dy = x2-x1, y2-y1
        if abs(dx) < 2: continue
        ang = np.degrees(np.arctan2(dy, dx))
        if -25 <= ang <= 25:
            angs.append(ang)
    return float(np.median(angs)) if angs else 0.0

def grabcut_foreground_mask(crop_bgr: np.ndarray,
                            margin: float = GC_MARGIN,
                            iters: int = GC_ITERS) -> np.ndarray:
    """GrabCut หา foreground (ป้ายทะเบียน) ออกจาก background"""
    H, W  = crop_bgr.shape[:2]
    mask  = np.zeros((H,W), np.uint8)
    mx, my = int(W*margin), int(H*margin)
    rect  = (mx, my, max(1,W-2*mx), max(1,H-2*my))
    bgd   = np.zeros((1,65), np.float64)
    fgd   = np.zeros((1,65), np.float64)
    cv2.grabCut(crop_bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    fg    = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype("uint8")
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    fg    = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg    = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
    return fg

def best_quad_from_mask(crop_bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray | None:
    """หา 4 มุมป้ายจาก foreground mask — ใช้ contour approx หรือ minAreaRect"""
    H, W  = crop_bgr.shape[:2]
    masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=fg_mask)
    gray  = cv2.GaussianBlur(cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY), (5,5), 0)
    edges = cv2.Canny(gray, 50, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(11,7)),
                             iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    best_q, best_a = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.12 * H * W: continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and area > best_a:
            best_a = area
            best_q = approx.reshape(4,2).astype(np.float32)
    if best_q is not None:
        return best_q
    # fallback: minAreaRect ของ contour ใหญ่สุด
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return cv2.boxPoints(rect).astype(np.float32)

def _quad_from_edges_direct(crop_bgr: np.ndarray) -> np.ndarray | None:
    """
    หา quad โดยตรงจาก edge detection (ไม่ต้องผ่าน GrabCut)
    ใช้เป็น fallback เมื่อ GrabCut ไม่สำเร็จ
    """
    H, W = crop_bgr.shape[:2]
    gray  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(11,7)),
                             iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.08 * H * W:
            continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2).astype(np.float32)
    # minAreaRect fallback
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    return cv2.boxPoints(rect).astype(np.float32)


def _simple_resize_warp(crop_bgr: np.ndarray, out_w: int, aspect: float) -> np.ndarray:
    """Fallback สุดท้าย: resize ตาม aspect ratio เฉยๆ"""
    out_h = max(60, int(out_w / aspect))
    return cv2.resize(crop_bgr, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def _pre_deskew(img: np.ndarray, max_deg: float = 20.0) -> np.ndarray:
    """
    หาองศาเอียงหลักก่อน warp — ใช้ minAreaRect บน edge เพื่อหาองศาที่ใหญ่กว่า Hough
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    c     = max(cnts, key=cv2.contourArea)
    rect  = cv2.minAreaRect(c)
    ang   = rect[2]
    # minAreaRect คืน -90..0 → แปลงให้เป็น rotation ที่ถูก
    if ang < -45:
        ang = 90 + ang
    ang = float(np.clip(ang, -max_deg, max_deg))
    return rotate_keep_size(img, ang)


def rectify_plate_2d(crop_bgr: np.ndarray, out_w: int, aspect: float):
    """
    ขึงภาพ 3 ระดับ:
      Pre-step: deskew ก่อน (minAreaRect)
      Level 1: GrabCut → quad → perspective warp  (ดีสุด)
      Level 2: Edge detection → quad → perspective warp  (fallback)
      Level 3: Simple resize
    คืน (vis, warped)
    """
    # Pre-deskew — แก้การหมุนใหญ่ก่อน
    crop_bgr = _pre_deskew(crop_bgr, max_deg=20.0)
    H, W = crop_bgr.shape[:2]
    quad = None
    method = "resize"

    # Level 1: GrabCut
    try:
        fg   = grabcut_foreground_mask(crop_bgr, GC_MARGIN, GC_ITERS)
        quad = best_quad_from_mask(crop_bgr, fg)
        if quad is not None:
            method = "grabcut"
    except Exception:
        quad = None

    # Level 2: Edge detection (ถ้า GrabCut ไม่ได้ quad)
    if quad is None:
        try:
            quad = _quad_from_edges_direct(crop_bgr)
            if quad is not None:
                method = "edge"
        except Exception:
            quad = None

    # วาด vis
    vis = crop_bgr.copy()
    if quad is not None:
        q_exp = expand_quad(quad, EXPAND, W, H)
        cv2.polylines(vis, [q_exp.astype(np.int32)], True, (0,255,0), 2)
        cv2.putText(vis, method, (5,20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1)
        warped = warp_fixed_aspect(crop_bgr, q_exp, out_w=out_w, aspect=aspect)
    else:
        # Level 3: simple resize
        cv2.putText(vis, "resize", (5,20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,200,255), 1)
        warped = _simple_resize_warp(crop_bgr, out_w, aspect)

    # validate warped — ถ้าเล็กหรือ blank ให้ fallback simple resize
    if warped is None or warped.size == 0:
        warped = _simple_resize_warp(crop_bgr, out_w, aspect)
    else:
        wH, wW = warped.shape[:2]
        # ถ้า warped เล็กกว่า 20% ของ out_w หรือ std ต่ำมาก (blank) → fallback
        if wW < out_w * 0.2 or np.std(warped) < 5:
            warped = _simple_resize_warp(crop_bgr, out_w, aspect)
            method = "resize-fallback"

    # Deskew ±6°
    try:
        ang    = float(np.clip(estimate_skew_angle_deg(warped), -MAX_DESKEW_DEG, MAX_DESKEW_DEG))
        warped = rotate_keep_size(warped, -ang)
    except Exception:
        pass

    return vis, warped


def _best_threshold(gray: np.ndarray) -> np.ndarray:
    """ลอง 3 วิธี — คืนตัวที่ black pixel ratio ใกล้ 30% มากสุด"""
    _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t2    = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    _, t3 = cv2.threshold(clahe.apply(gray), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    def score(t): return -abs(np.sum(t==0)/t.size - 0.30)
    return max([t1,t2,t3], key=score)


def preprocess_plate(img: np.ndarray, aspect: float = 2.8):
    """
    ขึงภาพ + threshold — ตรงตาม vehicle_detector.py
    คืน (vis_rgb, warped_rgb, th):
      vis_rgb    = crop + quad (debug view สีเขียว)
      warped_rgb = ภาพขึงตรง RGB (แสดงใน Streamlit)
      th         = grayscale threshold (ส่ง EasyOCR)
    """
    vis_bgr, warped = rectify_plate_2d(img, out_w=OUT_W, aspect=aspect)

    # scale up ถ้า warped เล็กเกิน
    wH, wW = warped.shape[:2]
    if wW < 400:
        scale  = 400 / max(wW, 1)
        warped = cv2.resize(warped, (int(wW*scale), int(wH*scale)),
                            interpolation=cv2.INTER_CUBIC)

    vis_rgb    = cv2.cvtColor(vis_bgr,  cv2.COLOR_BGR2RGB)
    warped_rgb = cv2.cvtColor(warped,   cv2.COLOR_BGR2RGB)
    gray       = cv2.cvtColor(warped,   cv2.COLOR_BGR2GRAY)
    gray       = cv2.bilateralFilter(gray, 5, 50, 50)
    # Sharpen ก่อน threshold — ทำให้ขอบตัวอักษรชัดขึ้น
    kernel     = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    gray       = cv2.filter2D(gray, -1, kernel)
    th         = _best_threshold(gray)
    th         = cv2.copyMakeBorder(th, 12, 12, 20, 20,
                                    cv2.BORDER_CONSTANT, value=255)
    return vis_rgb, warped_rgb, th


# ─── OCR + จัดเรียงตาม row ───────────────────────────────────────────────────
THAI_CONSONANTS = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
DIGITS          = "0123456789"
PLATE_ALLOWLIST = THAI_CONSONANTS + DIGITS
THAI_CHARS      = set(THAI_CONSONANTS)
D2C = {"5": "ร", "8": "ถ"}
C2D = {
    "ต": "6",   # ต ↔ 6
    "ร": "5",   # ร ↔ 5
    "ง": "9",   # ง ↔ 9
    "ว": "0",   # ว ↔ 0
    "ใ": "1",
    "ไ": "1",
    "ก": "6",
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
}
C2C = {
    "ธ": "ฐ",   # ธ ↔ ฐ
    "ฑ": "ฏ",   # ฑ ↔ ฏ
}
# Thai char ที่มักอ่านได้จาก digit (ใน digit zone ให้แปลงกลับ)
THAI_DIGIT_MAP = {
    "ต": "6",   # ต หน้าตาคล้าย 6 มาก
    "ร": "5",   # ร หน้าตาคล้าย 5
    "ง": "9",   # ง หน้าตาคล้าย 9
    "ว": "0",   # ว หน้าตาคล้าย 0
    "ใ": "1",
    "ไ": "1",
    "ก": "6",   # บางครั้ง ก อ่านเป็น 6
}

# ─── Context-aware OCR correction ────────────────────────────────────────────
# เมื่ออยู่ในส่วน "พยัญชนะ" → digit ที่หน้าตาคล้าย consonant ให้แปลงเป็น consonant
DIGIT_TO_CONSONANT = {
    "5": "ร",   # 5 หน้าตาคล้าย ร มาก
    "8": "ถ",   # 8 ↔ ถ
    "1": "า",   # ไม่ค่อยเกิด แต่ไว้ก่อน
    "0": "ว",
}
# เมื่ออยู่ในส่วน "ตัวเลข" → consonant ที่หน้าตาคล้าย digit ให้แปลงเป็น digit
CONSONANT_TO_DIGIT = {
    "ร": "5",
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "ส": "5",
}
# Thai consonant misreads (ไม่ขึ้นกับตำแหน่ง)
CONSONANT_FIXES = {
    "ธ": "ฐ",   # ธ ↔ ฐ
    "ฑ": "ฏ",   # ฑ ↔ ฏ
    "ก": "ถ",   # ก ↔ ถ (เฉพาะถ้า context บอกว่าน่าจะเป็น ถ)
}

THAI_CHARS = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

def _is_thai(ch: str) -> bool:
    return ch in THAI_CHARS

def _is_digit(ch: str) -> bool:
    return ch.isdigit()

def fix_ocr_text(text: str) -> str:
    """
    Context-aware correction สำหรับป้ายทะเบียนไทย

    Rules:
    1. digit ที่ตามด้วย Thai char ทันที → แปลงเป็น consonant (เช่น 5→ร, 8→ถ)
    2. consonant ที่อยู่นอก Thai zone → แปลงเป็น digit (เช่น ร→5)
    3. consonant misread → แปลงเป็น consonant ที่ถูก (เช่น ธ→ฐ)
    4. เพิ่ม space ระหว่าง consonant zone กับ digit zone
    """
    import re
    s = text.replace(" ", "")
    if not s:
        return text
    chars = list(s)
    n     = len(chars)

    # Pass 1: digit → consonant เฉพาะถ้า char ถัดไปเป็น Thai
    for i in range(n):
        if chars[i] in D2C and i < n - 1 and chars[i + 1] in THAI_CHARS:
            chars[i] = D2C[chars[i]]

    # Pass 2: consonant → correct consonant (C2C misreads)
    for i in range(n):
        chars[i] = C2C.get(chars[i], chars[i])

    # Pass 3: Thai char นอก Thai zone → digit
    thai_idx = [i for i, c in enumerate(chars) if c in THAI_CHARS]
    if thai_idx:
        t_start, t_end = thai_idx[0], thai_idx[-1]
        for i in range(n):
            if (i < t_start or i > t_end) and chars[i] in THAI_CHARS:
                chars[i] = C2D.get(chars[i], chars[i])

    fixed = "".join(chars)
    # เพิ่ม space ระหว่าง consonant กับ digit
    fixed = re.sub(r"([ก-๙])([0-9])", r"\1 \2", fixed)
    fixed = re.sub(r"([0-9])([ก-๙])", r"\1 \2", fixed)
    return fixed.strip()

def ocr_plate(reader, plate_proc: np.ndarray, v_type: str = "Car",
              reader_en=None) -> tuple:
    """
    คืน (plate_text, province, conf%, raw_rows)
    ใช้ reader_en อ่านตัวเลขแยกต่างหาก → แม่นกว่า Thai model อ่านตัวเลข
    """
    # Pass 1: Thai+EN allowlist พยัญชนะ+เลข
    raw_p = reader.readtext(plate_proc, detail=1, paragraph=False,
                            allowlist=PLATE_ALLOWLIST)
    # Pass 2: Thai ไม่มี allowlist (จับจังหวัด)
    raw_a = reader.readtext(plate_proc, detail=1, paragraph=False)
    # Pass 3: EN-only reader อ่านตัวเลข (แม่นกว่า Thai model มาก)
    raw_d = []
    if reader_en is not None:
        raw_d = reader_en.readtext(plate_proc, detail=1, paragraph=False,
                                   allowlist="0123456789")
    raw   = [r for r in raw_p + raw_a + raw_d if r[2] > 0.05 and r[1].strip()]

    seen, unique = set(), []
    for r in raw:
        key = (round(r[0][0][0]/10), round(r[0][0][1]/10))
        if key not in seen:
            seen.add(key); unique.append(r)
    raw = unique

    if not raw:
        return "—", "—", 0.0, []

    def cy(r): return (r[0][0][1] + r[0][2][1]) / 2
    def cx(r): return (r[0][0][0] + r[0][1][0]) / 2

    raw.sort(key=cy)
    row_thresh = plate_proc.shape[0] * 0.20
    rows, cur  = [], [raw[0]]
    for r in raw[1:]:
        if abs(cy(r) - cy(cur[-1])) < row_thresh:
            cur.append(r)
        else:
            rows.append(cur); cur = [r]
    rows.append(cur)
    for row in rows:
        row.sort(key=cx)

    row_texts = [" ".join(r[1].strip() for r in row) for row in rows]
    # แก้ตัวอักษรที่ OCR มักอ่านผิด
    row_texts = [fix_ocr_text(t) for t in row_texts]
    conf      = max(r[2] for r in raw)

    if v_type == "Motorcycle":
        if len(row_texts) >= 3:
            plate_str = f"{row_texts[0]} {row_texts[-1]}"
            province  = match_province(row_texts[1])
        elif len(row_texts) == 2:
            has_prov = bool(get_close_matches(row_texts[0], ALL_PROVINCES, n=1, cutoff=0.5))
            if has_prov:
                province  = match_province(row_texts[0])
                plate_str = row_texts[1]
            else:
                plate_str = row_texts[0]
                province  = match_province(row_texts[1])
        else:
            plate_str = row_texts[0]; province = "—"
    else:
        plate_str = row_texts[0]
        province  = match_province(row_texts[-1]) if len(row_texts) > 1 else "—"

    return plate_str, province, round(conf * 100, 1), row_texts


def get_vehicle_type(label: str, x1: int, y1: int, x2: int, y2: int) -> str:
    """YOLO class เป็นหลัก + aspect ratio เป็น fallback"""
    yolo_moto  = "Motorcycle" in label
    aspect     = (x2 - x1) / max(y2 - y1, 1)
    ratio_moto = aspect < 1.7
    if yolo_moto == ratio_moto:
        return "Motorcycle" if yolo_moto else "Car"
    return "Motorcycle" if ratio_moto else "Car"


# ─── Draw box overlay ─────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, plate_str, v_type, conf_yolo):
    color = (251, 191, 36) if v_type == "Motorcycle" else (52, 211, 153)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cs = min(16, (x2-x1)//4, (y2-y1)//4)
    for (bx, by), (dx, dy) in zip(
        [(x1,y1),(x2,y1),(x1,y2),(x2,y2)],
        [(1,1),(-1,1),(1,-1),(-1,-1)]
    ):
        cv2.line(frame,(bx,by),(bx+dx*cs,by),color,3)
        cv2.line(frame,(bx,by),(bx,by+dy*cs),color,3)

    label = f"{plate_str}  {conf_yolo:.0f}%"
    bg_y  = max(0, y1-34)
    cv2.rectangle(frame, (x1,bg_y), (x2,y1), (10,10,10), -1)
    cv2.putText(frame, label, (x1+5,y1-9), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, label, (x1+5,y1-9), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1)


# ─── Save log + capture ───────────────────────────────────────────────────────
def save_log(plate, province, v_type, frame):
    os.makedirs("captures", exist_ok=True)
    ts   = time.strftime("%Y-%m-%d %H:%M:%S")
    path = f"captures/{plate.replace(' ','_')}_{int(time.time())}.png"
    cv2.imwrite(path, frame)
    with open("logs.csv", "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([ts, plate, province, v_type, path])


# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [("run", False), ("last_plate", ""), ("last_time", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Config")
    model_path   = st.text_input("Model path", value="vehicle_detector.pt")
    conf_thresh  = st.slider("YOLO confidence", 0.1, 0.9, 0.5, 0.05)
    ocr_interval = st.slider("OCR interval (วิ)", 0.5, 5.0, 1.5, 0.5)
    st.markdown("---")
    st.markdown("### 📋 Pipeline")
    st.markdown("1. OpenCV — กล้อง\n2. YOLO — detect ป้าย\n3. ขึงภาพ — crop\n4. EasyOCR — อ่านข้อความ\n5. Output + log")
    st.markdown("---")
    if st.button("📂 ดู logs.csv"):
        if os.path.exists("logs.csv"):
            import pandas as pd
            df = pd.read_csv("logs.csv", header=None,
                             names=["time","plate","province","type","file"])
            st.dataframe(df.tail(20), use_container_width=True)
        else:
            st.info("ยังไม่มี log")


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 License Plate Detector")

if not os.path.exists(model_path):
    st.error(f"ไม่พบ {model_path} — วางไว้ใน folder เดียวกับ app.py")
    st.stop()

with st.spinner("Loading models..."):
    reader, reader_en, model = load_models(model_path)
st.success("Models ready ✓", icon="✅")

tab_cam, tab_img = st.tabs(["🎥 Webcam", "📷 Upload Image"])


# ── Webcam tab ────────────────────────────────────────────────────────────────
with tab_cam:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.run = True
    with c2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.run = False

    frame_box = st.empty()
    info_box  = st.empty()
    crop_box  = st.empty()

    if st.session_state.run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("❌ เปิดกล้องไม่ได้")
            st.session_state.run = False
        else:
            frame_count = 0
            YOLO_EVERY  = 5   # รัน YOLO ทุก N frame — ลด lag
            last_boxes  = []  # cache box จาก YOLO frame ก่อน

            while st.session_state.run:
                # flush buffer — อ่านทิ้งจนถึง frame ล่าสุด
                for _ in range(2):
                    cap.grab()
                ret, frame = cap.retrieve()
                if not ret:
                    break

                frame_count += 1
                display = frame.copy()

                # ── YOLO เฉพาะทุก N frame ──────────────────────────
                if frame_count % YOLO_EVERY == 0:
                    results    = model(frame, conf=conf_thresh, imgsz=224, verbose=False)
                    last_boxes = []
                    for result in results:
                        for box in result.boxes:
                            x1b, y1b, x2b, y2b = map(int, box.xyxy[0].cpu().numpy())
                            last_boxes.append({
                                "xyxy":  (x1b, y1b, x2b, y2b),
                                "conf":  float(box.conf[0].cpu().numpy()),
                                "label": result.names[int(box.cls[0])],
                            })

                # ── วาด box + OCR จาก cache ────────────────────────
                for det in last_boxes:
                    x1, y1, x2, y2 = det["xyxy"]
                    conf_yolo       = det["conf"]
                    label           = det["label"]
                    fH, fW          = frame.shape[:2]
                    x1, y1, x2, y2  = _pad_box(x1,y1,x2,y2, fW,fH, PAD_RATIO)

                    v_type  = get_vehicle_type(label, x1, y1, x2, y2)
                    v_th    = "🏍️ มอเตอร์ไซค์" if v_type == "Motorcycle" else "🚗 รถยนต์"
                    aspect  = ASPECT_BY_NAME.get(_norm_name(label), 1.0 if v_type=="Motorcycle" else 2.8)

                    # OCR ตาม interval — ไม่รันทุก frame
                    do_ocr = time.time() - st.session_state.last_time > ocr_interval
                    if do_ocr:
                        try:
                            plate_img              = frame[y1:y2, x1:x2]
                            vis_rgb, warped_rgb, plate_th = preprocess_plate(plate_img, aspect)
                            plate_str, province, conf_ocr, rows = ocr_plate(reader, plate_th, v_type, reader_en)

                            if plate_str and plate_str != st.session_state.last_plate:
                                st.session_state.last_plate = plate_str
                                save_log(plate_str, province, v_type, frame)

                            st.session_state.last_time = time.time()

                            tag = "tag-moto" if v_type == "Motorcycle" else "tag-car"
                            info_box.markdown(f"""
                            <div class="plate-card">
                                <div style="font-size:1.4rem;font-weight:600;margin-bottom:8px">{v_th}</div>
                                <span class="tag {tag}">{v_type}</span>
                                <span class="tag tag-conf">YOLO {conf_yolo*100:.0f}%</span>
                                <span class="tag tag-conf">OCR {conf_ocr}%</span>
                                <div class="plate-text">{plate_str}</div>
                                <div class="province-badge">📍 {province}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            crop_box.image(warped_rgb, caption="Plate crop (ขึงแล้ว)", width=300)

                        except Exception:
                            st.session_state.last_time = time.time()

                    draw_box(display, x1, y1, x2, y2,
                             st.session_state.last_plate or "...",
                             v_type, conf_yolo * 100)

                # ── แสดง frame — resize ลดก่อน render ─────────────
                small = cv2.resize(display, (640, 360), interpolation=cv2.INTER_LINEAR)
                frame_box.image(small, channels="BGR", use_container_width=True)

            cap.release()


# ── Upload Image tab ──────────────────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader("เลือกรูปภาพ", type=["jpg","jpeg","png","bmp"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # resize รูปใหญ่ให้เล็กลงก่อนทุกอย่าง
        iH, iW = frame.shape[:2]
        if iW > 960:
            scale  = 960 / iW
            frame  = cv2.resize(frame, (960, int(iH*scale)), interpolation=cv2.INTER_AREA)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**ภาพต้นฉบับ**")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                     use_container_width=False, width=420)

        display = frame.copy()
        results = model(frame, conf=conf_thresh, verbose=False)
        found   = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf_yolo = float(box.conf[0].cpu().numpy())
                iH, iW   = frame.shape[:2]
                x1, y1, x2, y2 = _pad_box(x1,y1,x2,y2, iW,iH, PAD_RATIO)
                label     = result.names[int(box.cls[0])]
                v_type    = get_vehicle_type(label, x1, y1, x2, y2)
                v_th      = "🏍️ มอเตอร์ไซค์" if v_type == "Motorcycle" else "🚗 รถยนต์"
                aspect    = ASPECT_BY_NAME.get(_norm_name(label), 1.0 if v_type=="Motorcycle" else 2.8)

                plate_img = frame[y1:y2, x1:x2]
                vis_rgb, warped_rgb, plate_th = preprocess_plate(plate_img, aspect)
                plate_str, province, conf_ocr, rows = ocr_plate(reader, plate_th, v_type, reader_en)

                draw_box(display, x1,y1,x2,y2, plate_str, v_type, conf_yolo*100)
                found.append({
                    "type": v_type, "type_th": v_th, "plate": plate_str,
                    "province": province,
                    "conf_yolo": round(conf_yolo*100,1),
                    "conf_ocr": conf_ocr,
                    "vis": vis_rgb,
                    "crop": warped_rgb,
                })

        with col_b:
            st.markdown("**ผลลัพธ์**")
            st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                     use_container_width=False, width=420)

        if found:
            st.markdown("---")
            st.markdown("### 🏷️ ป้ายทะเบียนที่พบ")
            cols = st.columns(min(len(found), 3))
            for i, r in enumerate(found):
                tag = "tag-moto" if r["type"] == "Motorcycle" else "tag-car"
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="plate-card">
                        <div style="font-size:1.3rem;font-weight:600;margin-bottom:8px">{r['type_th']}</div>
                        <span class="tag {tag}">{r['type']}</span>
                        <span class="tag tag-conf">YOLO {r['conf_yolo']}%</span>
                        <span class="tag tag-conf">OCR {r['conf_ocr']}%</span>
                        <div class="plate-text">{r['plate']}</div>
                        <div class="province-badge">📍 {r['province']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(r["vis"],  caption="Debug: quad", width=320)
                    st.image(r["crop"], caption="ขึงแล้ว (warped)", width=320)
        else:
            st.warning("ไม่พบป้ายทะเบียนในภาพนี้")
