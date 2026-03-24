"""
Flow การทำงาน
License Plate Detector — OpenCV → YOLO → Warp → EasyOCR
"""
import streamlit as st
import cv2, numpy as np, easyocr, time, csv, os, re, ssl
from ultralytics import YOLO
from difflib import get_close_matches

ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Sarabun',sans-serif}
.main,[data-testid="stAppViewContainer"]{background:#0a0c10}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0f1117;border-right:1px solid #1e2433}
.plate-card{background:#0f1117;border:1px solid #1e2433;border-radius:12px;padding:18px 22px;margin:10px 0}
.plate-text{font-family:'IBM Plex Mono',monospace;font-size:2rem;font-weight:600;letter-spacing:5px;
    color:#34d399;text-align:center;padding:12px;background:#052e16;border:1px solid #065f46;border-radius:8px}
.prov{font-size:1rem;color:#6ee7b7;text-align:center;margin-top:5px}
.tag{display:inline-block;font-size:0.7rem;padding:2px 9px;border-radius:20px;margin:2px;font-family:'IBM Plex Mono',monospace}
.tc{background:#1e3a5f;color:#60a5fa;border:1px solid #2563eb}
.tm{background:#3b1f00;color:#fbbf24;border:1px solid #d97706}
.tk{background:#1a1a2e;color:#a78bfa;border:1px solid #7c3aed}
</style>""", unsafe_allow_html=True)

#จังหวัดใช้สำหรับแก้ไข OCR ที่อ่านผิด
PROVINCES = ["กรุงเทพมหานคร","กระบี่","กาญจนบุรี","กาฬสินธุ์","กำแพงเพชร","ขอนแก่น","จันทบุรี","ฉะเชิงเทรา","ชลบุรี","ชัยนาท","ชัยภูมิ","ชุมพร","เชียงราย","เชียงใหม่","ตรัง","ตราด","ตาก","นครนายก","นครปฐม","นครพนม","นครราชสีมา","นครศรีธรรมราช","นครสวรรค์","นนทบุรี","นราธิวาส","น่าน","บึงกาฬ","บุรีรัมย์","ปทุมธานี","ประจวบคีรีขันธ์","ปราจีนบุรี","ปัตตานี","พระนครศรีอยุธยา","พะเยา","พังงา","พัทลุง","พิจิตร","พิษณุโลก","เพชรบุรี","เพชรบูรณ์","แพร่","ภูเก็ต","มหาสารคาม","มุกดาหาร","แม่ฮ่องสอน","ยโสธร","ยะลา","ร้อยเอ็ด","ระนอง","ระยอง","ราชบุรี","ลพบุรี","ลำปาง","ลำพูน","เลย","ศรีสะเกษ","สกลนคร","สงขลา","สตูล","สมุทรปราการ","สมุทรสงคราม","สมุทรสาคร","สระแก้ว","สระบุรี","สิงห์บุรี","สุโขทัย","สุพรรณบุรี","สุราษฎร์ธานี","สุรินทร์","หนองคาย","หนองบัวลำภู","อ่างทอง","อำนาจเจริญ","อุดรธานี","อุตรดิตถ์","อุทัยธานี","อุบลราชธานี"]
def province(text):
    m = get_close_matches(text.strip(), PROVINCES, n=1, cutoff=0.5)
    return m[0] if m else text.strip() or "—"

#โมเดล
@st.cache_resource
def load_models(path):
    return easyocr.Reader(["th","en"], gpu=True), YOLO(path)

#อ่านป้ายแม่นขึ้น
THAI   = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
ALLOW  = "".join(THAI)+"0123456789"
D2C    = {"5":"ร","8":"ถ"}
C2D    = {"ต":"6","ร":"5","ง":"9","ว":"0","ใ":"1","ไ":"1","ก":"6","O":"0","o":"0","I":"1","l":"1"}
C2C    = {"ธ":"ฐ","ฑ":"ฏ"}
ASPECT = {"car-license-plate":2.8,"motorcycle-license-plate":1.0,"license plate car":2.8,"license plate motorcycle":1.0}
CLAHE  = cv2.createCLAHE(3.0,(4,4))  # สร้างครั้งเดียว ไม่สร้างใหม่ทุก frame

def _order(pts):
    r=np.zeros((4,2),dtype="float32"); s=pts.sum(1); d=np.diff(pts,axis=1)
    r[0]=pts[np.argmin(s)]; r[2]=pts[np.argmax(s)]; r[1]=pts[np.argmin(d)]; r[3]=pts[np.argmax(d)]
    return r

def _rotate(img, deg):
    if abs(deg)<0.15: return img
    h,w=img.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),deg,1.0)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def _find_quad(crop):
    H,W=crop.shape[:2]
    try:
        mask=np.zeros((H,W),np.uint8); mx,my=int(W*.05),int(H*.05)
        bgd,fgd=np.zeros((1,65),np.float64),np.zeros((1,65),np.float64)
        cv2.grabCut(crop,mask,(mx,my,max(1,W-2*mx),max(1,H-2*my)),bgd,fgd,2,cv2.GC_INIT_WITH_RECT)
        fg=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype("uint8")
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        fg=cv2.morphologyEx(cv2.morphologyEx(fg,cv2.MORPH_CLOSE,k,iterations=2),cv2.MORPH_OPEN,k,iterations=1)
        masked=cv2.bitwise_and(crop,crop,mask=fg)
        edges=cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY),(5,5),0),50,160)
    except Exception:
        gray=cv2.bilateralFilter(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),9,75,75)
        edges=cv2.Canny(gray,30,120)
    edges=cv2.morphologyEx(cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1),
                           cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(11,7)),iterations=2)
    cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:10]
    for c in cnts:
        area=cv2.contourArea(c)
        if area<0.08*H*W: continue
        approx=cv2.approxPolyDP(c,0.02*cv2.arcLength(c,True),True)
        if len(approx)==4: return approx.reshape(4,2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(max(cnts,key=cv2.contourArea))).astype(np.float32)

def warp_plate(crop, aspect=2.8):
    """ขึงภาพ"""
    gray0=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    _,th0=cv2.threshold(gray0,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts0,_=cv2.findContours(th0,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts0:
        ang=cv2.minAreaRect(max(cnts0,key=cv2.contourArea))[2]
        if ang<-45: ang=90+ang
        crop=_rotate(crop,float(np.clip(ang,-20,20)))

    H,W=crop.shape[:2]; out_w=400; out_h=max(60,int(out_w/aspect))
    quad=_find_quad(crop)

    if quad is not None:
        q=_order(quad.astype("float32"))
        cx,cy=quad.mean(0)
        q=(q-[[cx,cy]])*1.04+[[cx,cy]]
        q[:,0]=np.clip(q[:,0],0,W-1); q[:,1]=np.clip(q[:,1],0,H-1)
        dst=np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]],dtype="float32")
        warped=cv2.warpPerspective(crop,cv2.getPerspectiveTransform(q,dst),(out_w,out_h))
        if warped.size==0 or warped.shape[1]<80 or np.std(warped)<5:
            warped=cv2.resize(crop,(out_w,out_h),interpolation=cv2.INTER_CUBIC)
    else:
        warped=cv2.resize(crop,(out_w,out_h),interpolation=cv2.INTER_CUBIC)

    g2=cv2.GaussianBlur(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY),(5,5),0)
    lines=cv2.HoughLinesP(cv2.Canny(g2,60,180),1,np.pi/180,80,
                          minLineLength=int(0.35*warped.shape[1]),maxLineGap=25)
    if lines is not None:
        angs=[np.degrees(np.arctan2(y2-y1,x2-x1)) for x1,y1,x2,y2 in lines.reshape(-1,4)
              if abs(x2-x1)>=2 and abs(np.degrees(np.arctan2(y2-y1,x2-x1)))<=25]
        if angs: warped=_rotate(warped,-float(np.clip(np.median(angs),-20,20)))

    # scale up + threshold
    wH,wW=warped.shape[:2]
    if wW<400: warped=cv2.resize(warped,(400,int(wH*400/max(wW,1))),interpolation=cv2.INTER_CUBIC)
    warped_rgb=cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)
    gray=cv2.bilateralFilter(cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY),5,50,50)
    gray=cv2.filter2D(gray,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32))
    _,t1=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    t2=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,10)
    clahe=CLAHE; _,t3=cv2.threshold(clahe.apply(gray),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    def sc(t): return -abs(np.sum(t==0)/t.size-0.30)
    th=cv2.copyMakeBorder(max([t1,t2,t3],key=sc),12,12,20,20,cv2.BORDER_CONSTANT,value=255)
    return warped_rgb, th

def fix(text):
    s=text.replace(" ",""); chars=list(s); n=len(chars)
    for i in range(n):                 
        if chars[i] in D2C and i<n-1 and chars[i+1] in THAI: chars[i]=D2C[chars[i]]
    for i in range(n): chars[i]=C2C.get(chars[i],chars[i])
    ti=[i for i,c in enumerate(chars) if c in THAI]
    if ti:
        for i in range(n):
            if (i<ti[0] or i>ti[-1]) and chars[i] in THAI: chars[i]=C2D.get(chars[i],chars[i])
    # ถ้าตัวอักษรไทยเกิน 2 ตัว → ตัวแรกสุดที่เป็นไทยให้แปลงเป็นตัวเลข
    ti2=[i for i,c in enumerate(chars) if c in THAI]
    if len(ti2)>2:
        chars[ti2[0]]=C2D.get(chars[ti2[0]],chars[ti2[0]])
    f="".join(chars)
    return re.sub(r"([0-9])([ก-๙])",r"\1 \2",re.sub(r"([ก-๙])([0-9])",r"\1 \2",f)).strip()

def read_plate(reader, th, v_type="Car"):
    r1=[r for r in reader.readtext(th,detail=1,paragraph=False,allowlist=ALLOW) if r[2]>0.05 and r[1].strip()]
    raw=r1 if r1 else [r for r in reader.readtext(th,detail=1,paragraph=False) if r[2]>0.05 and r[1].strip()]
    seen,uniq=set(),[]
    for r in raw:
        k=(round(r[0][0][0]/10),round(r[0][0][1]/10))
        if k not in seen: seen.add(k); uniq.append(r)
    if not uniq: return "—","—",0.0
    def cy(r): return (r[0][0][1]+r[0][2][1])/2
    def cx(r): return (r[0][0][0]+r[0][1][0])/2
    uniq.sort(key=cy); thr=th.shape[0]*0.2
    rows,cur=[],[uniq[0]]
    for r in uniq[1:]:
        if abs(cy(r)-cy(cur[-1]))<thr: cur.append(r)
        else: rows.append(cur); cur=[r]
    rows.append(cur)
    for row in rows: row.sort(key=cx)
    texts=[fix(" ".join(r[1] for r in row)) for row in rows]
    conf=max(r[2] for r in uniq)
    if v_type=="Motorcycle":
        if len(texts)>=3: return f"{texts[0]} {texts[-1]}",province(texts[1]),round(conf*100,1)
        if len(texts)==2:
            return (texts[1],province(texts[0]),round(conf*100,1)) if get_close_matches(texts[0],PROVINCES,n=1,cutoff=0.5) \
                   else (texts[0],province(texts[1]),round(conf*100,1))
        return texts[0],"—",round(conf*100,1)
    return texts[0], (province(texts[-1]) if len(texts)>1 else "—"), round(conf*100,1)

def vtype(label,x1,y1,x2,y2):
    moto="Motorcycle" in label; ratio=(x2-x1)/max(y2-y1,1)<1.7
    return ("Motorcycle" if moto else "Car") if moto==ratio else ("Motorcycle" if ratio else "Car")

def draw_box(frame,x1,y1,x2,y2,txt,vt,conf):
    col=(251,191,36) if vt=="Motorcycle" else (52,211,153)
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
    cs=min(16,(x2-x1)//4,(y2-y1)//4)
    for (bx,by),(dx,dy) in zip([(x1,y1),(x2,y1),(x1,y2),(x2,y2)],[(1,1),(-1,1),(1,-1),(-1,-1)]):
        cv2.line(frame,(bx,by),(bx+dx*cs,by),col,3); cv2.line(frame,(bx,by),(bx,by+dy*cs),col,3)
    bg=max(0,y1-34); cv2.rectangle(frame,(x1,bg),(x2,y1),(10,10,10),-1)
    lbl=f"{txt}  {conf:.0f}%"
    cv2.putText(frame,lbl,(x1+5,y1-9),cv2.FONT_HERSHEY_DUPLEX,.65,(255,255,255),2)
    cv2.putText(frame,lbl,(x1+5,y1-9),cv2.FONT_HERSHEY_DUPLEX,.65,col,1)

def pad_box(x1,y1,x2,y2,W,H,p=0.30):
    bw,bh=x2-x1,y2-y1; px,py=int(bw*p),int(bh*p)
    return max(0,x1-px),max(0,y1-py),min(W-1,x2+px),min(H-1,y2+py)

def card(r):
    tag="tm" if r["vt"]=="Motorcycle" else "tc"
    return f"""<div class="plate-card">
        <div style="font-size:1.3rem;font-weight:600;margin-bottom:8px">{r['vth']}</div>
        <span class="tag {tag}">{r['vt']}</span>
        <span class="tag tk">YOLO {r['cy']}%</span>
        <span class="tag tk">OCR {r['co']}%</span>
        <div class="plate-text">{r['plate']}</div>
        <div class="prov">📍 {r['prov']}</div></div>"""

def save_log(plate,prov,vt,frame):
    os.makedirs("captures",exist_ok=True)
    ts=time.strftime("%Y-%m-%d %H:%M:%S"); path=f"captures/{plate.replace(' ','_')}_{int(time.time())}.png"
    cv2.imwrite(path,frame)
    with open("logs.csv","a",newline="",encoding="utf-8") as f: csv.writer(f).writerow([ts,plate,prov,vt,path])

for k,v in [("run",False),("last",""),("lt",0)]:
    if k not in st.session_state: st.session_state[k]=v


with st.sidebar:
    st.markdown("ตั้งค่า")
    model_path  = st.text_input("Model path","vehicle_detector.pt")
    conf_thresh = st.slider("YOLO confidence",0.1,0.9,0.5,0.05)
    ocr_ivl     = st.slider("OCR interval (วิ)",0.5,5.0,2.0,0.5)
    if st.button("logs.csv"):
        if os.path.exists("logs.csv"):
            import pandas as pd
            st.dataframe(pd.read_csv("logs.csv",header=None,
                         names=["time","plate","province","type","file"]).tail(20))
        else: st.info("ยังไม่มี log")

st.markdown("#License Plate Detector")
if not os.path.exists(model_path): st.error(f"ไม่พบ {model_path}"); st.stop()

with st.spinner("กำลังโหลด..."):
    reader, model = load_models(model_path)
st.success("พร้อมใช้งาน")


tab_cam, tab_img = st.tabs(["Webcam","Upload"])
with tab_cam:
    c1,c2=st.columns(2)
    with c1:
        if st.button("เริ่มเปิดกล้อง",use_container_width=True): st.session_state.run=True
    with c2:
        if st.button("หยุดการทำงาน", use_container_width=True): st.session_state.run=False
    fb=st.empty(); ib=st.empty(); cb=st.empty()

    if st.session_state.run:
        cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        if not cap.isOpened(): st.error("ไม่สามารถเปิดกล้องได้"); st.session_state.run=False
        else:
            fn=0; cache=[]
            while st.session_state.run:
                for _ in range(2): cap.grab()
                ret,frame=cap.retrieve()
                if not ret: break
                fn+=1; disp=frame.copy()
                if fn%5==0:
                    res=model(frame,conf=conf_thresh,imgsz=224,verbose=False)
                    cache=[{"xyxy":tuple(map(int,b.xyxy[0].cpu().numpy())),
                            "conf":float(b.conf[0].cpu().numpy()),
                            "label":res[0].names[int(b.cls[0])]} for r in res for b in r.boxes]
                for d in cache:
                    x1,y1,x2,y2=d["xyxy"]; fH,fW=frame.shape[:2]
                    x1,y1,x2,y2=pad_box(x1,y1,x2,y2,fW,fH)
                    vt=vtype(d["label"],x1,y1,x2,y2)
                    asp=ASPECT.get(d["label"].strip().lower(),1.0 if vt=="Motorcycle" else 2.8)
                    if time.time()-st.session_state.lt>ocr_ivl:
                        try:
                            wr,th=warp_plate(frame[y1:y2,x1:x2],asp)
                            pl,pv,co=read_plate(reader,th,vt)
                            if pl and pl!=st.session_state.last: st.session_state.last=pl; save_log(pl,pv,vt,frame)
                            st.session_state.lt=time.time()
                            vth="มอเตอร์ไซค์" if vt=="Motorcycle" else "รถยนต์"
                            ib.markdown(card({"vt":vt,"vth":vth,"cy":d["conf"]*100,"co":co,"plate":pl,"prov":pv}),unsafe_allow_html=True)
                            cb.image(wr,caption="ขึงแล้ว",width=300)
                        except Exception: st.session_state.lt=time.time()
                    draw_box(disp,x1,y1,x2,y2,st.session_state.last or "...",vt,d["conf"]*100)
                fb.image(cv2.resize(disp,(640,360)),channels="BGR",use_container_width=True)
            cap.release()

with tab_img:
    uploaded=st.file_uploader("เลือกรูปภาพ",type=["jpg","jpeg","png","bmp"])
    if uploaded:
        frame=cv2.imdecode(np.frombuffer(uploaded.read(),np.uint8),cv2.IMREAD_COLOR)
        iH,iW=frame.shape[:2]
        if iW>960: frame=cv2.resize(frame,(960,int(iH*960/iW)),interpolation=cv2.INTER_AREA)
        ca,cb=st.columns(2)
        with ca: st.markdown("**ต้นฉบับ**"); st.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),width=420)
        disp=frame.copy(); res=model(frame,conf=conf_thresh,verbose=False); found=[]
        for r in res:
            for box in r.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy()); conf_y=float(box.conf[0].cpu().numpy())
                iH,iW=frame.shape[:2]; x1,y1,x2,y2=pad_box(x1,y1,x2,y2,iW,iH)
                label=r.names[int(box.cls[0])]; vt=vtype(label,x1,y1,x2,y2)
                asp=ASPECT.get(label.strip().lower(),1.0 if vt=="Motorcycle" else 2.8)
                wr,th=warp_plate(frame[y1:y2,x1:x2],asp); pl,pv,co=read_plate(reader,th,vt)
                draw_box(disp,x1,y1,x2,y2,pl,vt,conf_y*100)
                found.append({"vt":vt,"vth":"มอเตอร์ไซค์" if vt=="Motorcycle" else "รถยนต์",
                              "cy":round(conf_y*100,1),"co":co,"plate":pl,"prov":pv,"crop":wr})
        with cb: st.markdown("**ผลลัพธ์**"); st.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB),width=420)
        if found:
            st.markdown("---"); st.markdown("ป้ายทะเบียนที่พบ")
            cols=st.columns(min(len(found),3))
            for i,r in enumerate(found):
                with cols[i%3]:
                    st.markdown(card(r),unsafe_allow_html=True)
                    st.image(r["crop"],caption="ขึงแล้ว",width=320)
        else: st.warning("ไม่พบป้ายทะเบียน")