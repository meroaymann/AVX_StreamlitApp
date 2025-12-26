# -*- coding: utf-8 -*-
# ArabicX-Viz | Streamlit single-file app (organized, interactive UI)
# Run: streamlit run app.py
from __future__ import annotations

import io

import math
import os

import hashlib
import base64

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch

import cv2

from ultralytics import YOLO

from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

import arabic_reshaper
from bidi.algorithm import get_display

# Optional deps for LIME
try:
    from lime import lime_image  # type: ignore
except Exception:
    lime_image = None

try:
    from skimage.segmentation import slic  # type: ignore
except Exception:
    slic = None

# Optional Mask R-CNN (not required)
try:
    import torchvision  # type: ignore
except Exception:
    torchvision = None

    # Optional deps for SHAP (multimodal)
try:
    import shap  # type: ignore
except Exception:
    shap = None

# ============================================================
# 0) App config
# ============================================================
st.set_page_config(
    page_title="ArabicX-Viz | Arabic VQA + Visual Explainability",
    page_icon="🧠",
    layout="wide",
)

# ============================================================
# 1) UI THEME (CSS) + small UI helpers
# ============================================================
def _auto_crop_logo(pil_img: Image.Image, tol: int = 18) -> Image.Image:
    """
    Removes a square background baked into the logo image.
    - If image has transparency, crops by non-transparent pixels.
    - Otherwise estimates background color from corners and crops by color distance.
    """
    img = pil_img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]

    # If alpha exists and has transparent areas, crop by alpha
    if np.min(alpha) < 255:
        ys, xs = np.where(alpha > 0)
        if len(xs) and len(ys):
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            return img.crop((x1, y1, x2 + 1, y2 + 1))

    # No useful alpha, crop by corner background color
    corners = np.array(
        [
            arr[0, 0, :3],
            arr[0, -1, :3],
            arr[-1, 0, :3],
            arr[-1, -1, :3],
        ],
        dtype=np.float32,
    )
    bg = corners.mean(axis=0)

    rgb = arr[:, :, :3].astype(np.float32)
    dist = np.sqrt(np.sum((rgb - bg) ** 2, axis=2))
    mask = dist > float(tol)

    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return img.crop((x1, y1, x2 + 1, y2 + 1))

    return img


def _file_to_data_uri(path: str) -> Optional[str]:
    try:
        pil = Image.open(path)

        # IMPORTANT: crop the square background from the logo file itself
        pil = _auto_crop_logo(pil, tol=18).convert("RGBA")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")  # force PNG so transparency stays
        b = buf.getvalue()
        return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")
    except Exception:
        return None


def _find_logo_data_uri(size: str = "big") -> Optional[str]:
    path = (
        "src\images\ArabicXViz_logo_big.jpeg"
        if size == "big"
        else "src\images\ArabicXViz_logo_small.jpeg"
    )
    if not os.path.exists(path):
        return None
    return _file_to_data_uri(path)



def inject_pro_ui(logo_big: Optional[str], logo_small: Optional[str]) -> None:
    st.markdown(
        f"""
        <style>
        header, footer, #MainMenu {{ display: none; }}

        .stApp {{
          background: linear-gradient(135deg,#f7f8ff,#fff7f0);
        }}

        /* ===== LANDING ===== */
        .landing {{
          height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          animation: fadeIn 0.8s ease;
        }}

        .landing img {{
          width: 380px;
          cursor: pointer;
          animation: zoomIn 1.2s ease;
          transition: transform 0.4s ease;
        }}

        .landing img:hover {{
          transform: scale(1.05);
        }}

        .landing span {{
          margin-bottom: 18px;
          font-size: 14px;
          opacity: 0.7;
        }}

        /* ===== HEADER ===== */
        .appHeader {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 22px;
          background: white;
          box-shadow: 0 6px 18px rgba(0,0,0,0.06);
          animation: slideDown 0.6s ease;
        }}

        .brand {{
          display: flex;
          align-items: center;
          gap: 12px;
        }}

        .brand img {{
          width: 48px;
        }}

        .brand h3 {{
          margin: 0;
          font-weight: 900;
        }}

        .pageNav {{
          display: flex;
          gap: 12px;
        }}

        .navBtn {{
          padding: 8px 14px;
          border-radius: 20px;
          border: 1px solid #ddd;
          background: #fafafa;
          cursor: pointer;
          transition: all 0.25s;
        }}

        .navBtn:hover {{
          background: #4e63ff;
          color: white;
        }}

        /* ===== CARDS ===== */
        .bigCard {{
          border-radius: 22px;
          padding: 26px;
          background: white;
          box-shadow: 0 10px 28px rgba(0,0,0,0.08);
          animation: fadeUp 0.6s ease;
        }}

        @keyframes zoomIn {{ from {{ transform: scale(0.7); opacity:0 }} to {{ transform: scale(1); opacity:1 }} }}
        @keyframes fadeIn {{ from {{ opacity:0 }} to {{ opacity:1 }} }}
        @keyframes fadeUp {{ from {{ transform: translateY(20px); opacity:0 }} to {{ opacity:1 }} }}
        @keyframes slideDown {{ from {{ transform: translateY(-20px); opacity:0 }} to {{ opacity:1 }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def scroll_to_top() -> None:
    st.markdown(
        """
        <script>
        window.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        unsafe_allow_html=True,
    )

def app_logo_svg(size_px: int = 44) -> str:
    return f"""
    <svg width="{size_px}" height="{size_px}" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#4e63ff" stop-opacity="0.9"/>
          <stop offset="1" stop-color="#ff7a00" stop-opacity="0.85"/>
        </linearGradient>
      </defs>
      <rect x="4" y="4" width="56" height="56" rx="16" fill="url(#g1)" opacity="0.25"/>
      <rect x="10" y="10" width="44" height="44" rx="14" fill="url(#g1)" opacity="0.35"/>
      <path d="M21 42 L31 20 L41 42" fill="none" stroke="#111827" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M27 34 L35 34" fill="none" stroke="#111827" stroke-width="4" stroke-linecap="round"/>
      <circle cx="45" cy="42" r="5" fill="#111827" opacity="0.92"/>
    </svg>
    """

def page_landing():
    if "started" not in st.session_state:
        st.session_state.started = False

    if not st.session_state.started:
        st.markdown(
            f"""
            <div class="landing">
              <span>Click the logo to start</span>
              <img src="{st.session_state.logo_big}">
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start", key="start_app"):
            st.session_state.started = True
            st.session_state.page = "About"
        st.stop()

# ============================================================
# 2) Navigation
# ============================================================
UI_PAGES = ["About", "How to Test", "System"]

def get_page() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    return str(st.session_state.page)


def set_page(page: str):
    st.session_state.page = page
    scroll_to_top()


def nav_buttons_row() -> None:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="small")
    with c1:
        if st.button("🏠 Home", use_container_width=True):
            set_page("Home")
    with c2:
        if st.button("🧩 System", use_container_width=True):
            set_page("About the System")
    with c3:
        if st.button("🧪 How to Test", use_container_width=True):
            set_page("How to Test")
    with c4:
        if st.button("🚀 Demo", use_container_width=True):
            set_page("Try the System")


# ============================================================
# 3) Core helpers: images, normalization, overlays
# ============================================================
def _center_crop(pil_img: Image.Image, size: int) -> Image.Image:
    w, h = pil_img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return pil_img.crop((left, top, left + size, top + size))


def blip_model_seen_image(pil_img: Image.Image, target: int = 480) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = float(target) / float(min(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    img = _center_crop(img, target)
    return img


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def resize_mask_to_image(mask01: np.ndarray, W: int, H: int) -> np.ndarray:
    mask01 = mask01.astype(np.float32)
    resized = cv2.resize(mask01, (W, H), interpolation=cv2.INTER_NEAREST)
    return np.clip(resized, 0.0, 1.0)


def resize_heat_to_image(heat01: np.ndarray, W: int, H: int) -> np.ndarray:
    heat01 = heat01.astype(np.float32)
    resized = cv2.resize(heat01, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(resized, 0.0, 1.0)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(np.min(x))
    mx = float(np.max(x))
    if mx > 1e-8:
        x = x / mx
    return np.clip(x, 0.0, 1.0)


def postprocess_heatmap(
    heat01: np.ndarray,
    blur_sigma: float = 6.0,
    clip_percentile: float = 99.0,
    gamma: float = 0.7,
) -> np.ndarray:
    if heat01 is None or heat01.size == 0:
        return heat01
    h = heat01.astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        h = cv2.GaussianBlur(h, ksize=(0, 0), sigmaX=float(blur_sigma), sigmaY=float(blur_sigma))
    h = normalize01(h)
    if clip_percentile is not None and 50.0 <= float(clip_percentile) <= 100.0:
        p = np.percentile(h, float(clip_percentile))
        if p > 1e-8:
            h = np.clip(h / float(p), 0.0, 1.0)
    if gamma and gamma > 0:
        h = np.power(h, float(gamma))
    return normalize01(h)


def overlay_heatmap_jet(pil_img: Image.Image, heat01: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    if heat01.shape[:2] != (H, W):
        heat01 = resize_heat_to_image(heat01, W=W, H=H)
    h = (heat01 * 255).astype(np.uint8)
    colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - float(alpha)) * img + float(alpha) * colored
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def apply_mask_black(pil_img: Image.Image, mask01: np.ndarray) -> Image.Image:
    bgr = pil_to_bgr(pil_img).astype(np.float32)
    H, W = bgr.shape[:2]
    if mask01.shape[:2] != (H, W):
        mask01 = resize_mask_to_image(mask01, W=W, H=H)
    m = mask01[..., None].astype(np.float32)
    out = bgr * m
    out = np.clip(out, 0, 255).astype(np.uint8)
    return bgr_to_pil(out)


def remove_mask_inpaint(pil_img: Image.Image, mask01: np.ndarray) -> Image.Image:
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]
    if mask01.shape[:2] != (H, W):
        mask01 = resize_mask_to_image(mask01, W=W, H=H)
    m255 = (mask01 > 0.5).astype(np.uint8) * 255
    out = cv2.inpaint(bgr, m255, 3, cv2.INPAINT_TELEA)
    return bgr_to_pil(out)


def uploaded_image_key(uploaded_file) -> str:
    try:
        b = uploaded_file.getvalue()
        return hashlib.md5(b).hexdigest()
    except Exception:
        return "unknown"


# ============================================================
# 4) Arabic text helpers
# ============================================================
def rtl_ar_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    try:
        return get_display(arabic_reshaper.reshape(s))
    except Exception:
        return s


def build_expl_ar(question_ar: str, answer_ar: str, caption_ar: str, top_obj: Optional[Dict[str, Any]]) -> str:
    s = f"أجاب النموذج عن السؤال: «{question_ar}» بالإجابة: «{answer_ar}»."
    if top_obj is None:
        s += " لم يتم اكتشاف كائنات واضحة في الصورة لتفسير الإجابة على مستوى الكائن."
    else:
        s += (
            " اعتمد التفسير على تدخلات على مستوى الكائنات (إبقاء أو إزالة كائن) "
            f"وتم اعتبار الكائن الأكثر تأثيرًا هو: «{top_obj.get('name','')}» "
            f"(درجة التأثير ≈ {float(top_obj.get('score',0.0)):.3f})."
        )
    if caption_ar:
        s += f" وهذا يتماشى مع الوصف الآلي للصورة: «{caption_ar}»."
    return s


# ============================================================
# 5) Models (cached)
# ============================================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap_name = "nlpconnect/vit-gpt2-image-captioning"
    cap_model = VisionEncoderDecoderModel.from_pretrained(cap_name).to(device)
    cap_feat = ViTImageProcessor.from_pretrained(cap_name)
    cap_tok = AutoTokenizer.from_pretrained(cap_name)

    vqa_name = "Salesforce/blip-vqa-base"
    vqa_proc = BlipProcessor.from_pretrained(vqa_name)
    vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_name).to(device)

    mt_name = "facebook/m2m100_418M"
    mt_tok = M2M100Tokenizer.from_pretrained(mt_name)
    mt_model = M2M100ForConditionalGeneration.from_pretrained(mt_name).to(device)

    yolo = YOLO("yolov8n-seg.pt")

    return device, cap_model, cap_feat, cap_tok, vqa_proc, vqa_model, mt_model, mt_tok, yolo


device, cap_model, cap_feat, cap_tok, vqa_proc, vqa_model, mt_model, mt_tok, yolo = load_models()

# ============================================================
# 6) Inference helpers: translate, caption, answer+confidence
# ============================================================
def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    mt_tok.src_lang = src_lang
    enc = mt_tok(text, return_tensors="pt").to(device)
    out = mt_model.generate(
        **enc,
        forced_bos_token_id=mt_tok.get_lang_id(tgt_lang),
        max_length=128,
        num_beams=4,
    )
    return mt_tok.batch_decode(out, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def caption_en(pil_img: Image.Image) -> str:
    cap_model.eval()
    pixel = cap_feat(images=pil_img, return_tensors="pt").pixel_values.to(device)
    out_ids = cap_model.generate(
        pixel,
        max_length=20,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return cap_tok.decode(out_ids[0], skip_special_tokens=True).strip()


@torch.no_grad()
def blip_answer_and_conf(pil_img: Image.Image, question_en: str) -> Tuple[str, float]:
    question_en = (question_en or "").strip()
    if not question_en:
        return "no question", 0.0

    vqa_model.eval()
    inputs = vqa_proc(pil_img, question_en, return_tensors="pt").to(device)
    gen = vqa_model.generate(
        **inputs,
        max_new_tokens=16,
        num_beams=3,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    ans = vqa_proc.decode(gen.sequences[0], skip_special_tokens=True).strip()
    ans = ans.replace("\n", " ").strip()

    probs: List[float] = []
    for step_logits in gen.scores:
        step_prob = torch.softmax(step_logits[0], dim=-1).max().item()
        probs.append(float(step_prob))
    conf = float(np.mean(probs)) if probs else 0.0

    if not ans:
        return "no idea", 0.0
    return ans, conf


def _norm_ans(s: str) -> str:
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    return s


# ============================================================
# 6.5) Answer likelihood (for LIME)
# ============================================================
@torch.no_grad()
def blip_answer_logprob_proxy(pil_img: Image.Image, question_en: str, target_answer_en: str) -> float:
    """
    Proxy probability-like score for a specific answer string under BLIP-VQA.
    We compute teacher-forced loss for the target answer tokens and map to exp(-loss).
    Higher = model likes that answer more for this image/question.
    """
    q = (question_en or "").strip()
    a = (target_answer_en or "").strip()
    if not q or not a:
        return 0.0

    vqa_model.eval()

    base_inputs = vqa_proc(pil_img, q, return_tensors="pt").to(device)
    tok = vqa_proc.tokenizer
    ans_ids = tok(a, return_tensors="pt", add_special_tokens=True).input_ids.to(device)

    out = vqa_model(
        pixel_values=base_inputs["pixel_values"],
        input_ids=base_inputs.get("input_ids", None),
        attention_mask=base_inputs.get("attention_mask", None),
        labels=ans_ids,
        return_dict=True,
    )
    loss = out.loss
    if loss is None:
        return 0.0

    score = float(torch.exp(-loss).detach().cpu().item())
    if not np.isfinite(score):
        return 0.0
    return float(score)


# ============================================================
# 7) XAI: YOLO segments + interventions + answer-aware heatmap
# ============================================================
def yolo_segments(pil_img: Image.Image, conf_thres: float = 0.25) -> List[Dict[str, Any]]:
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]

    res = yolo.predict(bgr, conf=conf_thres, verbose=False)[0]

    objs: List[Dict[str, Any]] = []
    if res.masks is None or res.boxes is None:
        return objs

    names = res.names
    boxes = res.boxes
    masks = res.masks.data.detach().cpu().numpy()

    for i in range(masks.shape[0]):
        cls = int(boxes.cls[i].item())
        det_conf = float(boxes.conf[i].item())
        name = names.get(cls, str(cls))

        xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        mask01 = resize_mask_to_image(masks[i], W=W, H=H)
        objs.append(
            {
                "i": i + 1,
                "cls": cls,
                "name": name,
                "det_conf": det_conf,
                "mask01": mask01,
                "bbox": (x1, y1, x2, y2),
            }
        )

    return objs


def score_objects_by_intervention(
    pil_img: Image.Image,
    question_en: str,
    conf_thres: float = 0.25,
    topk: int = 10,
) -> Tuple[str, float, List[Dict[str, Any]], np.ndarray]:
    base_ans, base_conf = blip_answer_and_conf(pil_img, question_en)

    objs = yolo_segments(pil_img, conf_thres=conf_thres)
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]
    if not objs:
        return base_ans, float(base_conf), [], np.zeros((H, W), dtype=np.float32)

    heat = np.zeros((H, W), dtype=np.float32)
    scored: List[Dict[str, Any]] = []

    for o in objs:
        mask01 = o["mask01"]

        img_only = apply_mask_black(pil_img, mask01)
        _, only_conf = blip_answer_and_conf(img_only, question_en)

        img_removed = remove_mask_inpaint(pil_img, mask01)
        _, removed_conf = blip_answer_and_conf(img_removed, question_en)

        drop = float(max(base_conf - removed_conf, 0.0))
        gain = float(max(only_conf - base_conf, 0.0))
        score = float(drop + gain)

        entry = dict(o)
        entry.update(
            {
                "base_ans": base_ans,
                "base_conf": float(base_conf),
                "only_conf": float(only_conf),
                "removed_conf": float(removed_conf),
                "drop": drop,
                "gain": gain,
                "score": score,
                "img_only": img_only,
                "img_removed": img_removed,
            }
        )
        scored.append(entry)
        heat += mask01 * score

    heat01 = heat / (heat.max() + 1e-8) if heat.max() > 1e-8 else heat
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return base_ans, float(base_conf), scored[:topk], heat01


def answer_aware_heatmap_from_objects(objs_sorted: List[Dict[str, Any]], H: int, W: int) -> np.ndarray:
    heat = np.zeros((H, W), dtype=np.float32)
    for o in objs_sorted:
        mask01 = o["mask01"]
        if mask01.shape[:2] != (H, W):
            mask01 = resize_mask_to_image(mask01, W=W, H=H)
        heat += mask01 * float(o.get("score", 0.0))
    if heat.max() > 1e-8:
        heat = heat / (heat.max() + 1e-8)
    return heat


# ============================================================
# 7.5) Counterfactual explanations
# ============================================================
def _mask_area_fraction(mask01: np.ndarray) -> float:
    m = (mask01 > 0.5).astype(np.float32)
    denom = float(m.size) if m.size else 1.0
    return float(m.sum() / denom)


def apply_local_blur_mask(pil_img: Image.Image, mask01: np.ndarray, sigma: float = 12.0) -> Image.Image:
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]
    if mask01.shape[:2] != (H, W):
        mask01 = resize_mask_to_image(mask01, W=W, H=H)

    blurred = cv2.GaussianBlur(bgr, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    m = (mask01 > 0.5).astype(np.float32)[..., None]
    out = bgr.astype(np.float32) * (1.0 - m) + blurred.astype(np.float32) * m
    return bgr_to_pil(np.clip(out, 0, 255).astype(np.uint8))


def apply_color_shift_mask(
    pil_img: Image.Image,
    mask01: np.ndarray,
    hue_delta: int = 35,
    sat_scale: float = 1.15,
) -> Image.Image:
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]
    if mask01.shape[:2] != (H, W):
        mask01 = resize_mask_to_image(mask01, W=W, H=H)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = (mask01 > 0.5).astype(np.float32)

    hsv[..., 0] = (hsv[..., 0] + float(hue_delta) * m) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + (float(sat_scale) - 1.0) * m), 0.0, 255.0)

    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return bgr_to_pil(out)


def apply_grayscale_mask(pil_img: Image.Image, mask01: np.ndarray) -> Image.Image:
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]
    if mask01.shape[:2] != (H, W):
        mask01 = resize_mask_to_image(mask01, W=W, H=H)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    m = (mask01 > 0.5).astype(np.float32)[..., None]
    out = bgr.astype(np.float32) * (1.0 - m) + gray3.astype(np.float32) * m
    return bgr_to_pil(np.clip(out, 0, 255).astype(np.uint8))


def compute_counterfactuals(
    pil_img: Image.Image,
    question_en: str,
    objs_sorted: List[Dict[str, Any]],
    max_objects: int = 6,
    blur_sigmas: Optional[List[float]] = None,
    topk: int = 5,
) -> Dict[str, Any]:
    blur_sigmas = blur_sigmas or [10.0, 14.0, 18.0]

    base_ans, base_conf = blip_answer_and_conf(pil_img, question_en)
    base_key = _norm_ans(base_ans)

    if not objs_sorted:
        return {
            "base_ans": base_ans,
            "base_conf": float(base_conf),
            "base_key": base_key,
            "candidates": [],
            "note": "No detected objects. Counterfactuals need regions (YOLO masks) here.",
        }

    candidates: List[Dict[str, Any]] = []
    for o in objs_sorted[: max(1, int(max_objects))]:
        mask01 = o["mask01"]
        area = _mask_area_fraction(mask01)

        edits: List[Tuple[str, Image.Image, Dict[str, Any]]] = []
        edits.append(("remove_inpaint", remove_mask_inpaint(pil_img, mask01), {"method": "inpaint_remove"}))
        for s in blur_sigmas:
            edits.append(
                (
                    f"blur_sigma={s:.0f}",
                    apply_local_blur_mask(pil_img, mask01, sigma=float(s)),
                    {"method": "local_blur", "sigma": float(s)},
                )
            )
        edits.append(("color_shift", apply_color_shift_mask(pil_img, mask01, hue_delta=35, sat_scale=1.15), {"method": "color_shift", "hue_delta": 35, "sat_scale": 1.15}))
        edits.append(("grayscale_region", apply_grayscale_mask(pil_img, mask01), {"method": "grayscale_region"}))

        for edit_name, img_edit, meta in edits:
            ans2, conf2 = blip_answer_and_conf(img_edit, question_en)
            key2 = _norm_ans(ans2)

            if key2 == base_key:
                continue

            candidates.append(
                {
                    "object_name": o.get("name", "obj"),
                    "object_idx": int(o.get("i", 0)),
                    "det_conf": float(o.get("det_conf", 0.0)),
                    "area_frac": float(area),
                    "edit_name": edit_name,
                    "meta": meta,
                    "ans_edit": ans2,
                    "conf_edit": float(conf2),
                    "conf_drop": float(base_conf - conf2),
                    "image_edit": img_edit,
                    "bbox": o.get("bbox", None),
                }
            )

    candidates.sort(key=lambda d: (float(d.get("area_frac", 1.0)), abs(float(d.get("conf_drop", 0.0)))))
    candidates = candidates[: max(1, int(topk))]

    return {
        "base_ans": base_ans,
        "base_conf": float(base_conf),
        "base_key": base_key,
        "candidates": candidates,
        "note": "Counterfactual = controlled local edit on a detected region that flips the BLIP answer.",
    }


# ============================================================
# 8) XAI: Question-guided cross-attention heatmap (proxy)
# ============================================================
def _to_square_grid_from_src_len(src_len: int) -> Optional[int]:
    if src_len <= 1:
        return None
    g = int(round(math.sqrt(src_len - 1)))
    if 1 + g * g == src_len:
        return g
    return None


def _pick_attention_tensor(attn_stack) -> List[torch.Tensor]:
    if attn_stack is None:
        return []
    layers = list(attn_stack)
    per_layer: List[torch.Tensor] = []
    for a in layers:
        if a is None:
            continue
        if hasattr(a, "dim") and a.dim() == 4:
            a0 = a[0]
            if a0.dim() == 3:
                per_layer.append(a0)
            continue
        if hasattr(a, "dim") and a.dim() == 3:
            per_layer.append(a)
            continue
    return per_layer


@torch.no_grad()
def blip_question_guided_attention_heatmap(
    pil_img: Image.Image,
    question_en: str,
    layer_reduce: str = "mean",
    token_reduce: str = "mean",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    vqa_model.eval()
    inputs = vqa_proc(pil_img, question_en, return_tensors="pt").to(device)

    proxy_cross = None
    proxy_source = None
    proxy_note = None

    try:
        if hasattr(vqa_model, "vision_model") and hasattr(vqa_model, "text_encoder"):
            vision_out = vqa_model.vision_model(
                pixel_values=inputs["pixel_values"],
                return_dict=True,
                output_attentions=False,
            )
            vision_embeds = vision_out.last_hidden_state

            te = vqa_model.text_encoder(
                input_ids=inputs.get("input_ids", None),
                attention_mask=inputs.get("attention_mask", None),
                encoder_hidden_states=vision_embeds,
                encoder_attention_mask=torch.ones(
                    vision_embeds.shape[:2],
                    device=vision_embeds.device,
                    dtype=torch.long,
                ),
                return_dict=True,
                output_attentions=True,
            )

            if hasattr(te, "cross_attentions") and te.cross_attentions is not None:
                proxy_cross = te.cross_attentions
                proxy_source = "text_encoder.cross_attentions"
            else:
                proxy_note = "text_encoder did not expose cross_attentions"
        else:
            proxy_note = "model missing vision_model/text_encoder"
    except Exception as e:
        proxy_note = f"proxy_failed: {type(e).__name__}: {e}"

    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]

    if proxy_cross is None:
        return np.zeros((H, W), dtype=np.float32), {
            "mode": "no_cross_attn_found",
            "source": proxy_source,
            "note": proxy_note or "No proxy cross-attn available",
        }

    layers = _pick_attention_tensor(proxy_cross)
    if not layers:
        return np.zeros((H, W), dtype=np.float32), {
            "mode": "cross_attn_empty_after_parse",
            "source": proxy_source,
            "note": proxy_note,
        }

    if layer_reduce == "last":
        layers = layers[-1:]
        layer_used = "last"
    else:
        layer_used = "mean"

    attn = torch.stack(layers, dim=0).mean(dim=0)  # (heads, tgt, src)
    attn = attn.mean(dim=0)  # (tgt, src)

    tgt_len, src_len = attn.shape

    am = inputs.get("attention_mask", None)
    if am is not None:
        am_np = am[0].detach().float().cpu().numpy()
    else:
        am_np = np.ones((tgt_len,), dtype=np.float32)

    start = 1 if tgt_len > 1 else 0
    q_attn = attn[start:, :]
    q_attn_np = q_attn.detach().cpu().numpy().astype(np.float32)

    q_mask = am_np[start:] if am_np.ndim == 1 else np.ones((q_attn_np.shape[0],), dtype=np.float32)
    q_mask = np.clip(q_mask.astype(np.float32), 0.0, 1.0)

    q_len = q_attn_np.shape[0]
    if q_mask.shape[0] >= q_len:
        q_mask = q_mask[:q_len]
    else:
        q_mask = np.pad(q_mask, (0, q_len - q_mask.shape[0]), constant_values=1.0)

    q_attn_np = q_attn_np * q_mask[:, None]

    if token_reduce == "max":
        tok_attn = np.max(q_attn_np, axis=0)
    else:
        denom = float(q_mask.sum()) if q_mask.sum() > 1e-8 else float(q_len)
        tok_attn = np.sum(q_attn_np, axis=0) / max(1e-8, denom)

    tok_attn = tok_attn.astype(np.float32)

    g = _to_square_grid_from_src_len(int(src_len))
    if g is None:
        return np.zeros((H, W), dtype=np.float32), {
            "mode": "src_not_square",
            "source": proxy_source,
            "tgt_len": int(tgt_len),
            "src_len": int(src_len),
            "note": proxy_note,
        }

    grid = tok_attn[1:].reshape(g, g)  # drop CLS
    grid = normalize01(grid)

    heat01 = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
    heat01 = normalize01(heat01)

    return heat01, {
        "mode": "question_cross_attn_ok",
        "source": proxy_source,
        "layer_reduce": layer_used,
        "token_reduce": token_reduce,
        "tgt_len": int(tgt_len),
        "src_len": int(src_len),
        "grid": int(g),
        "note": proxy_note,
    }


# ============================================================
# 9) XAI: Vanilla Gradient (Pixel Saliency)
# ============================================================
def vanilla_gradient_saliency_heatmap(
    pil_img: Image.Image,
    question_en: str,
    max_new_tokens: int = 16,
    num_beams: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    vqa_model.eval()
    inputs = vqa_proc(pil_img, question_en, return_tensors="pt").to(device)

    with torch.no_grad():
        seq = vqa_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)

    pixel_values = inputs["pixel_values"].detach().clone().requires_grad_(True)

    decoder_input_ids = seq[:, :-1]
    labels = seq[:, 1:].clone()

    vqa_model.zero_grad(set_to_none=True)
    out = vqa_model(
        pixel_values=pixel_values,
        input_ids=inputs.get("input_ids", None),
        attention_mask=inputs.get("attention_mask", None),
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        return_dict=True,
    )

    loss = out.loss
    if loss is None:
        bgr = pil_to_bgr(pil_img)
        H, W = bgr.shape[:2]
        return np.zeros((H, W), dtype=np.float32), {"mode": "no_loss"}

    loss.backward()

    grad = pixel_values.grad
    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]

    if grad is None:
        return np.zeros((H, W), dtype=np.float32), {"mode": "no_grad_returned"}

    sal = grad[0].detach().abs().mean(dim=0).cpu().numpy().astype(np.float32)
    sal = normalize01(sal)
    heat01 = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    heat01 = normalize01(heat01)

    return heat01, {
        "mode": "vanilla_grad_ok",
        "loss": float(loss.detach().cpu().item()),
        "seq_len": int(seq.shape[1]),
        "pixel_tensor_hw": [int(sal.shape[0]), int(sal.shape[1])],
    }


# ============================================================
# 10) XAI: Grad-CAM for BLIP-VQA (ViT patch-token CAM)
# ============================================================
class _TensorHook:
    def __init__(self) -> None:
        self.tensor: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            t = output[0]
        else:
            t = getattr(output, "last_hidden_state", output)
        self.tensor = t
        if hasattr(self.tensor, "retain_grad"):
            self.tensor.retain_grad()


def _vit_token_gradcam_from_acts_grads(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    A = acts[0]  # (tokens, dim)
    G = grads[0]

    A = A[1:, :]  # drop CLS
    G = G[1:, :]

    N = A.shape[0]
    g = int(round(math.sqrt(N)))
    if g * g != N:
        return np.zeros((max(g, 1), max(g, 1)), dtype=np.float32)

    weights = G.mean(dim=0)
    cam_tokens = torch.relu((A * weights[None, :]).sum(dim=1))
    cam = cam_tokens.reshape(g, g).detach().float().cpu().numpy()
    cam = normalize01(cam)
    return cam.astype(np.float32)


def blip_vit_gradcam_heatmap(
    pil_img: Image.Image,
    question_en: str,
    max_new_tokens: int = 16,
    num_beams: int = 3,
    use_last_vision_layer: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    question_en = (question_en or "").strip()
    if not question_en:
        return np.zeros((384, 384), dtype=np.float32), {"mode": "no_question"}

    model_seen_384 = blip_model_seen_image(pil_img, target=384)
    inputs = vqa_proc(model_seen_384, question_en, return_tensors="pt").to(device)

    vqa_model.eval()
    with torch.no_grad():
        seq = vqa_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False)

    hook = _TensorHook()

    try:
        vision_layers = vqa_model.vision_model.encoder.layers  # type: ignore
    except Exception:
        vision_layers = None

    if not vision_layers:
        return np.zeros((384, 384), dtype=np.float32), {"mode": "no_vision_layers"}

    layer = vision_layers[-1] if use_last_vision_layer else vision_layers[0]
    handle = layer.register_forward_hook(hook)

    try:
        decoder_input_ids = seq[:, :-1]
        labels = seq[:, 1:].clone()

        vqa_model.zero_grad(set_to_none=True)
        out = vqa_model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs.get("input_ids", None),
            attention_mask=inputs.get("attention_mask", None),
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

        loss = out.loss
        if loss is None:
            return np.zeros((384, 384), dtype=np.float32), {"mode": "no_loss"}

        loss.backward()

        acts = hook.tensor
        if acts is None or acts.grad is None:
            return np.zeros((384, 384), dtype=np.float32), {"mode": "no_acts_or_grads"}

        cam_grid = _vit_token_gradcam_from_acts_grads(acts, acts.grad)
        cam_384 = cv2.resize(cam_grid, (384, 384), interpolation=cv2.INTER_CUBIC)
        cam_384 = normalize01(cam_384)

        dbg = {
            "mode": "vit_gradcam_ok",
            "loss": float(loss.detach().cpu().item()),
            "seq_len": int(seq.shape[1]),
            "tokens_patches_plus_cls": int(acts.shape[1]) if hasattr(acts, "shape") else None,
            "cam_grid": list(cam_grid.shape),
            "aligned_resolution": [384, 384],
        }
        return cam_384.astype(np.float32), dbg
    finally:
        try:
            handle.remove()
        except Exception:
            pass


# ============================================================
# 11) XAI: Attention-weighted boxes (Top-K)
# ============================================================
def attention_weighted_boxes_topk(attn_heat01: np.ndarray, objs_sorted: List[Dict[str, Any]], topk: int = 5) -> List[Dict[str, Any]]:
    if attn_heat01 is None or attn_heat01.size == 0:
        return []
    H, W = attn_heat01.shape[:2]
    scored: List[Dict[str, Any]] = []
    for o in (objs_sorted or []):
        x1, y1, x2, y2 = o["bbox"]
        x1 = max(0, min(W - 1, int(x1)))
        x2 = max(0, min(W, int(x2)))
        y1 = max(0, min(H - 1, int(y1)))
        y2 = max(0, min(H, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        patch = attn_heat01[y1:y2, x1:x2]
        score = float(np.mean(patch)) if patch.size else 0.0
        oo = dict(o)
        oo["attn_score"] = score
        scored.append(oo)
    scored.sort(key=lambda d: float(d.get("attn_score", 0.0)), reverse=True)
    return scored[: max(1, int(topk))]


def yolo_label_ar(english_label: str) -> str:
    if "yolo_label_cache" not in st.session_state:
        st.session_state.yolo_label_cache = {}
    key = (english_label or "").strip().lower()
    if not key:
        return ""
    if key in st.session_state.yolo_label_cache:
        return st.session_state.yolo_label_cache[key]
    ar = translate(english_label, "en", "ar")
    st.session_state.yolo_label_cache[key] = ar
    return ar


def _pick_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    for p in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/tahoma.ttf"]:
        try:
            return ImageFont.truetype(p, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_boxes_on_pil_ar(
    pil_img: Image.Image,
    boxes: List[Dict[str, Any]],
    score_key: str = "attn_score",
    font_path: Optional[str] = None,
    font_size: int = 22,
) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    font = _pick_font(font_path, font_size)

    W, H = img.size
    for idx, o in enumerate(boxes):
        x1, y1, x2, y2 = o["bbox"]
        x1 = max(0, min(W - 1, int(x1)))
        x2 = max(0, min(W - 1, int(x2)))
        y1 = max(0, min(H - 1, int(y1)))
        y2 = max(0, min(H - 1, int(y2)))

        name_en = o.get("name", "obj")
        name_ar = yolo_label_ar(name_en)
        name_ar_rtl = rtl_ar_text(name_ar)
        score_val = float(o.get(score_key, 0.0))
        label = f"{idx+1}: {name_ar_rtl} ({name_en}) {score_val:.3f}"

        draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

        try:
            bbox_txt = draw.textbbox((0, 0), label, font=font)
            tw = bbox_txt[2] - bbox_txt[0]
            th = bbox_txt[3] - bbox_txt[1]
        except Exception:
            tw, th = (320, 30)

        y_text = max(0, y1 - th - 4)
        draw.rectangle([x1, y_text, x1 + tw + 8, y_text + th + 6], fill=(0, 0, 0))
        draw.text((x1 + 4, y_text + 2), label, fill="yellow", font=font)

    return img


# ============================================================
# 12.5) NEW XAI: LIME for VQA (image superpixels)
# ============================================================
def lime_vqa_explanation(
    pil_img: Image.Image,
    question_en: str,
    base_answer_en: str,
    num_samples: int = 700,
    num_features: int = 10,
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    LIME for VQA (image side): perturb superpixels (turn off regions) and fit a local linear model
    to explain the score of the base answer.
    """
    if lime_image is None or slic is None:
        return None, {
            "mode": "missing_deps",
            "note": "Install: lime, scikit-image",
        }

    q = (question_en or "").strip()
    a = (base_answer_en or "").strip()
    if not q or not a:
        return None, {"mode": "no_question_or_answer"}

    np_img = np.array(pil_img.convert("RGB"))

    segments = slic(np_img, n_segments=120, compactness=10.0, sigma=1.0, start_label=0)

    def classifier_fn(images: List[np.ndarray]) -> np.ndarray:
        probs_out: List[List[float]] = []
        for im in images:
            pim = Image.fromarray(im.astype(np.uint8))
            s = blip_answer_logprob_proxy(pim, q, a)
            p_yes = float(np.clip(s, 0.0, 1.0))
            p_no = float(1.0 - p_yes)
            probs_out.append([p_no, p_yes])
        return np.array(probs_out, dtype=np.float32)

    explainer = lime_image.LimeImageExplainer(random_state=0)
    explanation = explainer.explain_instance(
        np_img,
        classifier_fn=classifier_fn,
        labels=(1,),
        top_labels=None,
        hide_color=0,
        num_samples=int(num_samples),
        segmentation_fn=lambda x: segments,
    )

    temp, mask = explanation.get_image_and_mask(
        label=1,
        positive_only=True,
        num_features=int(num_features),
        hide_rest=False,
    )

    overlay = np_img.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    alpha = 0.35
    m = (mask > 0).astype(np.float32)[..., None]
    overlay = (overlay.astype(np.float32) * (1.0 - alpha * m) + green.astype(np.float32) * (alpha * m)).astype(np.uint8)

    return Image.fromarray(overlay), {
        "mode": "ok",
        "num_samples": int(num_samples),
        "num_features": int(num_features),
        "note": "Explains which superpixels most support the base answer score under local perturbations.",
    }

# ============================================================
# 12.6) NEW XAI: SHAP for multimodal inputs (image superpixels + question token-groups)
# ============================================================
def _mask_image_by_superpixel_binary(
    np_img: np.ndarray,
    segments: np.ndarray,
    z: np.ndarray,
    hide_color: int = 0,
) -> np.ndarray:
    """
    z: shape (num_segments,), values in {0,1}.
    1 = keep segment, 0 = hide segment.
    """
    out = np_img.copy()
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)

    for seg_id in range(int(segments.max()) + 1):
        if float(z[seg_id]) < 0.5:
            out[segments == seg_id] = hide_color
    return out


def _group_tokens_simple(tokens: List[str], group_size: int = 2) -> List[List[int]]:
    groups: List[List[int]] = []
    i = 0
    while i < len(tokens):
        groups.append(list(range(i, min(i + group_size, len(tokens)))))
        i += group_size
    return groups


def _mask_question_by_groups(question_en: str, tok_groups: List[List[int]], z: np.ndarray) -> str:
    """
    z: binary keep mask for groups. If group is 0, we drop its tokens from the question.
    This is a simple and stable masking strategy for BLIP VQA.
    """
    q = (question_en or "").strip()
    if not q:
        return q

    # Simple whitespace tokenization for stability across platforms
    toks = q.split()
    keep = np.ones((len(toks),), dtype=np.int32)

    for gi, idxs in enumerate(tok_groups):
        if float(z[gi]) < 0.5:
            for ti in idxs:
                if 0 <= ti < len(keep):
                    keep[ti] = 0

    kept = [toks[i] for i in range(len(toks)) if keep[i] == 1]
    out = " ".join(kept).strip()
    return out if out else q  # avoid empty question


def shap_multimodal_explanation(
    pil_img: Image.Image,
    question_en: str,
    base_answer_en: str,
    num_samples_img: int = 250,
    num_samples_txt: int = 250,
    group_size: int = 2,
    top_groups: int = 10,
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    SHAP-style feature contributions for BLIP VQA:
      - Image side: superpixels as features (keep/hide) explaining the score of the BASE answer
      - Text side: token-groups as features (keep/drop) explaining the score of the BASE answer

    Score function is aligned to your code: blip_answer_logprob_proxy(image, question, base_answer).
    """
    if shap is None or slic is None:
        return None, {
            "mode": "missing_deps",
            "note": "Install: shap, scikit-image, scipy, scikit-learn, numba",
        }

    q = (question_en or "").strip()
    a = (base_answer_en or "").strip()
    if not q or not a:
        return None, {"mode": "no_question_or_answer"}

    np_img = np.array(pil_img.convert("RGB"))
    segments = slic(np_img, n_segments=120, compactness=10.0, sigma=1.0, start_label=0)
    num_segs = int(segments.max()) + 1

    # -------------------------
    # 1) IMAGE-SIDE SHAP (superpixels)
    # -------------------------
    def f_img(Z: np.ndarray) -> np.ndarray:
        # Z shape: (batch, num_segs) with 0/1
        outs = []
        for z in Z:
            masked = _mask_image_by_superpixel_binary(np_img, segments, z, hide_color=0)
            pim = Image.fromarray(masked.astype(np.uint8))
            s = blip_answer_logprob_proxy(pim, q, a)
            outs.append([float(s)])
        return np.array(outs, dtype=np.float32)

    # background: all segments hidden (zeros)
    bg_img = np.zeros((1, num_segs), dtype=np.float32)
    x_img = np.ones((1, num_segs), dtype=np.float32)

    expl_img = shap.KernelExplainer(f_img, bg_img)
    shap_vals_img = expl_img.shap_values(x_img, nsamples=int(num_samples_img))
    # shap_vals_img can be list or array depending on SHAP version
    if isinstance(shap_vals_img, list):
        shap_vals_img = shap_vals_img[0]
    shap_vals_img = np.array(shap_vals_img).reshape(-1)  # (num_segs,)

    # Map SHAP values to pixel heat
    heat = np.zeros((np_img.shape[0], np_img.shape[1]), dtype=np.float32)
    for seg_id in range(num_segs):
        heat[segments == seg_id] = float(shap_vals_img[seg_id])

    # Keep only positive contributions for a clean “supporting the answer” view
    heat_pos = np.clip(heat, 0.0, None)
    heat01 = normalize01(heat_pos)
    heat01 = postprocess_heatmap(heat01, blur_sigma=6.0, clip_percentile=99.0, gamma=0.7)
    overlay_img = overlay_heatmap_jet(pil_img, heat01, alpha=0.45)

    # -------------------------
    # 2) TEXT-SIDE SHAP (token groups)
    # -------------------------
    toks = q.split()
    tok_groups = _group_tokens_simple(toks, group_size=max(1, int(group_size)))
    G = len(tok_groups)

    def f_txt(Z: np.ndarray) -> np.ndarray:
        outs = []
        for z in Z:
            q_masked = _mask_question_by_groups(q, tok_groups, z)
            s = blip_answer_logprob_proxy(pil_img, q_masked, a)
            outs.append([float(s)])
        return np.array(outs, dtype=np.float32)

    bg_txt = np.zeros((1, G), dtype=np.float32)  # all groups dropped
    x_txt = np.ones((1, G), dtype=np.float32)    # all groups kept

    expl_txt = shap.KernelExplainer(f_txt, bg_txt)
    shap_vals_txt = expl_txt.shap_values(x_txt, nsamples=int(num_samples_txt))
    if isinstance(shap_vals_txt, list):
        shap_vals_txt = shap_vals_txt[0]
    shap_vals_txt = np.array(shap_vals_txt).reshape(-1)  # (G,)

    group_texts: List[str] = []
    for gi, idxs in enumerate(tok_groups):
        phrase = " ".join([toks[i] for i in idxs if 0 <= i < len(toks)]).strip()
        group_texts.append(phrase if phrase else f"group_{gi+1}")

    # Rank groups by absolute contribution
    order = np.argsort(-np.abs(shap_vals_txt))
    top = []
    for gi in order[: max(1, int(top_groups))]:
        top.append(
            {
                "group_id": int(gi),
                "text": group_texts[int(gi)],
                "shap_value": float(shap_vals_txt[int(gi)]),
            }
        )

    dbg = {
        "mode": "ok",
        "note": "SHAP KernelExplainer over (1) superpixel keep/hide for image and (2) token-group keep/drop for question. Score = exp(-loss) proxy for the BASE answer under BLIP.",
        "image": {
            "num_segments": int(num_segs),
            "nsamples": int(num_samples_img),
        },
        "text": {
            "num_groups": int(G),
            "group_size": int(group_size),
            "nsamples": int(num_samples_txt),
            "top_groups": top,
        },
    }

    return overlay_img, dbg

# ============================================================
# 14) Pages
# ============================================================
def hero(title: str, subtitle: str, badges: List[str]) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="heroGlow"></div>
          <div class="heroTitle">{title}</div>
          <p class="heroSub">{subtitle}</p>
          <div class="badgeRow">
            {''.join([f'<div class="badge">{b}</div>' for b in badges])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_home() -> None:
    # LANDING (centered, smaller, cropped logo)
    st.session_state.setdefault("landing_done", False)
    logo_uri = st.session_state.get("_logo_data_uri", None)

    if (not st.session_state.landing_done) and logo_uri:
        st.markdown('<div class="landingWrap">', unsafe_allow_html=True)
        st.markdown('<div class="landingTitle">Click the logo to start</div>', unsafe_allow_html=True)

        st.markdown('<div class="landingStart">', unsafe_allow_html=True)
        clicked = st.button(" ", key="start_logo_btn")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="landingSub">ArabicX-Viz Demo</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if clicked:
            st.session_state.landing_done = True
            set_page("Try the System")
        return

    # Normal Home page
    topbar("Home")
    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)

    hero(
        "ArabicX-Viz",
        "Arabic-first interactive system for Visual Question Answering and visual explanations: Arabic question → English proxy → BLIP VQA → Arabic answer with multiple visual explanation views.",
        badges=["Arabic-first UX", "BLIP VQA", "YOLOv8 Seg", "Visual XAI", "Retrieval XAI", "Counterfactual XAI", "LIME (optional)", "PDF Report"],
    )

    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)
    nav_buttons_row()
    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            """
            <div class="card">
              <h4>👩‍🎓 About me</h4>
              <p><b>Dania</b> | Data Science student. This demo supports my bachelor thesis on Arabic-focused visual explainability for language and vision-language models.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h4>🎯 Thesis topic</h4>
              <p><b>ArabicX-Viz:</b> Visual Explainability for Arabic Language Models (image-focused). The goal is to make outputs clear and easy to judge for Arabic users.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
              <h4>✨ What makes it “wow”</h4>
              <p>Multiple explanations in one place: object interventions, answer-aware heatmaps, attention, saliency, ViT Grad-CAM, retrieval explanations, plus <b>counterfactual answer-flip edits</b>. Optional: <b>LIME</b> superpixel explanation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)
    st.info("Optional: If you want your university logo, save an image named giu_logo.png next to app.py, then it will appear automatically.")
    if os.path.exists("src\images\giu_logo.png"):
        st.image("src\images\giu_logo.png", caption="University logo", width=180)


def app_header():
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px">
              <img src="{st.session_state.logo_small}" width="48">
              <h3 style="margin:0">ArabicX-Viz</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("About", use_container_width=True):
                set_page("About")
        with b2:
            if st.button("How to Test", use_container_width=True):
                set_page("How to Test")
        with b3:
            if st.button("System", use_container_width=True):
                set_page("System")


def page_about():
    st.markdown("## About ArabicX-Viz")

    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        st.markdown(
            """
            <div class="bigCard">
              <h3>👩‍🎓 About Me</h3>
              <p style="color:#000000">
              Dania, Data Science undergraduate at the German International University (GIU).
              This system is developed as part of my bachelor thesis.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="bigCard" style="margin-top:20px">
              <h3>🎯 About the Project</h3>
              <p>
              ArabicX-Viz is an interactive system for visual explainability in Arabic
              Vision–Language Models. It enables users to understand how models reason
              over images and Arabic questions using multiple XAI techniques.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        if os.path.exists("images/giu_logo.png"):
            st.image("images/giu_logo.png", width=180)


def page_how_to_test() -> None:
    topbar("How to Test")
    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)

    hero(
        "How to test the system",
        "Follow these short steps. The goal is to judge both the answer and whether the explanation makes sense.",
        badges=["Simple steps", "Best demo tips", "User study ready"],
    )

    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)
    nav_buttons_row()

    st.markdown(
        """
        <div class="card">
          <h4>✅ Best images for a strong demo</h4>
          <p>Use clear images with 1 to 3 main objects. Examples: a person holding something, a car with a driver, a pet doing an action.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
          <h4>✅ Best Arabic question style</h4>
          <p>Short and specific. Examples: <b>ماذا يحمل الشخص؟</b> <b>من يقود السيارة؟</b> <b>ماذا يفعل الكلب؟</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
          <h4>✅ How to evaluate</h4>
          <p>Give ratings for (1) answer correctness, (2) explanation helpfulness. If the answer is wrong, check whether the heatmap still highlights a reasonable region. Also check retrieval and counterfactuals: do the nearest examples look relevant, and do small edits flip the answer in a believable way?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_demo() -> None:
    topbar("Try the System")
    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)
    nav_buttons_row()
    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)

    init_retrieval_memory()

    st.session_state.setdefault("question_ar", "")
    st.session_state.setdefault("last_answer_ar", "")
    st.session_state.setdefault("computed", False)

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("ارفع صورة للبدء.")
        return

    pil_img = Image.open(uploaded).convert("RGB")
    img_key = uploaded_image_key(uploaded)

    with st.spinner("Generating caption..."):
        cap_en = caption_en(pil_img)
        cap_ar = translate(cap_en, "en", "ar")

    left, right = st.columns([1.25, 1.0], gap="large")
    with left:
        st.image(pil_img, caption="الصورة", use_container_width=True)
    with right:
        st.markdown("### 📝 Caption (Arabic)")
        st.info(cap_ar if cap_ar else "—")
        st.caption(f"Caption (EN): {cap_en}")

    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)

    q_col, a_col = st.columns([1.0, 1.0], gap="large")
    with q_col:
        st.markdown("### ❓ سؤال (Arabic)")
        q_ar = st.text_input(
            "اكتب سؤالك هنا",
            value=st.session_state.question_ar,
            placeholder="مثال: ماذا يفعل الكلب؟",
            label_visibility="collapsed",
        )
        ask = st.button("📤 اسأل", use_container_width=True)
        st.session_state.question_ar = q_ar

    with a_col:
        st.markdown("### ✅ الإجابة (Arabic)")
        if st.session_state.last_answer_ar:
            st.success(st.session_state.last_answer_ar)
        else:
            st.info("اكتب سؤالًا واضغط «اسأل».")
        if st.session_state.get("_base_conf") is not None and st.session_state.computed:
            st.caption(f"Confidence (proxy): {float(st.session_state._base_conf):.3f}")

    if ask:
        q_ar_clean = (st.session_state.question_ar or "").strip()
        if not q_ar_clean:
            st.warning("اكتب سؤالًا بالعربية أولاً.")
            return

        with st.spinner("Translating question + answering..."):
            q_en = translate(q_ar_clean, "ar", "en")
            base_ans_en, base_conf = blip_answer_and_conf(pil_img, q_en)
            base_ans_ar = translate(base_ans_en, "en", "ar")

        st.session_state.last_answer_ar = base_ans_ar
        st.session_state._q_en = q_en
        st.session_state._base_ans_en = base_ans_en
        st.session_state._base_conf = float(base_conf)
        st.session_state._img_key = img_key
        st.session_state._cap_en = cap_en
        st.session_state._cap_ar = cap_ar

        with st.spinner("Running YOLO-seg + intervention scoring..."):
            _, _, objs_sorted, heat01 = score_objects_by_intervention(pil_img, q_en, conf_thres=0.25, topk=10)

        st.session_state._objs_sorted = objs_sorted
        st.session_state._heat01 = heat01

        with st.spinner("Extracting question-guided attention heatmap..."):
            attn_heat01, attn_dbg = blip_question_guided_attention_heatmap(pil_img, q_en, layer_reduce="mean", token_reduce="mean")

        st.session_state._attn_heat01 = attn_heat01
        st.session_state._attn_dbg = attn_dbg

        st.session_state._cf_cache_key = f"{img_key}||{_norm_ans(q_en)}"
        st.session_state._cf_results = None

        st.session_state._lime_cache_key = f"{img_key}||{_norm_ans(q_en)}||{_norm_ans(base_ans_en)}"
        st.session_state._lime_overlay = None
        st.session_state._lime_dbg = None

        st.session_state.computed = True

    if not st.session_state.computed:
        return

    q_ar_clean = (st.session_state.question_ar or "").strip()
    q_en = st.session_state._q_en
    base_ans_en = st.session_state._base_ans_en
    base_conf = float(st.session_state._base_conf)
    base_ans_ar = st.session_state.last_answer_ar
    cap_en = st.session_state.get("_cap_en", cap_en)
    cap_ar = st.session_state.get("_cap_ar", cap_ar)

    objs_sorted: List[Dict[str, Any]] = st.session_state._objs_sorted
    heat01 = st.session_state._heat01
    attn_heat01 = st.session_state.get("_attn_heat01", None)
    attn_dbg = st.session_state.get("_attn_dbg", {})

    bgr = pil_to_bgr(pil_img)
    H, W = bgr.shape[:2]

    st.markdown('<div class="softDivider"></div>', unsafe_allow_html=True)

    SECTIONS = [
        "1) Object Intervention",
        "2) Answer-aware Heatmap",
        "3) Spatial Attention (Question-guided)",
        "4) Active Attention (Pick an object)",
        "5) Counterfactual Explanations (Answer flip)",
        "6) Arabic Text Explanation",
        "7) Attention-weighted Boxes (Top-K)",
        "8) Vanilla Gradient (Pixel Saliency)",
        "9) Grad-CAM (BLIP ViT aligned)",
        "10) LIME for VQA (superpixels, optional)",
        "11) SHAP for multimodal inputs (superpixels + token groups)",
    ]
    section = st.selectbox("📌 Choose a view", SECTIONS, index=0)

    if section == SECTIONS[0]:
        st.subheader("🧩 Object / Region Attention (Segmentation + Intervention Scoring)")
        if not objs_sorted:
            st.warning("لم يتم اكتشاف كائنات بواسطة YOLO-seg. جرّبي صورة أوضح.")
        else:
            top_obj = objs_sorted[0]
            st.markdown("#### 📌 Top objects (most influential)")
            for o in objs_sorted[:5]:
                x1, y1, x2, y2 = o["bbox"]
                st.write(
                    f"- #{o['i']} **{o['name']}** | det_conf={o['det_conf']:.2f} | "
                    f"base_conf={o['base_conf']:.3f} | only_conf={o['only_conf']:.3f} | removed_conf={o['removed_conf']:.3f} | "
                    f"drop={o['drop']:.3f} gain={o['gain']:.3f} | **score={o['score']:.3f}** | bbox=({x1},{y1})-({x2},{y2})"
                )

            st.markdown("#### 👁️ Top object preview")
            cA, cB, cC = st.columns(3, gap="large")
            with cA:
                st.image(top_obj["img_only"], caption=f"Only: {top_obj['name']}", use_container_width=True)
            with cB:
                st.image(top_obj["img_removed"], caption=f"Removed: {top_obj['name']}", use_container_width=True)
            with cC:
                st.markdown(
                    f"""
                    **Question (EN):** {q_en}  
                    **Base answer (EN):** {base_ans_en}  
                    **Base confidence:** {base_conf:.3f}  

                    **only_conf:** {top_obj['only_conf']:.3f}  
                    **removed_conf:** {top_obj['removed_conf']:.3f}  
                    **score (drop+gain):** {top_obj['score']:.3f}
                    """
                )

    elif section == SECTIONS[1]:
        st.subheader("🔥 Answer-aware Heatmap (Object masks weighted by intervention score)")
        if not objs_sorted:
            st.warning("لا يوجد كائنات مكتشفة لتكوين heatmap.")
        else:
            heat_ans = answer_aware_heatmap_from_objects(objs_sorted, H, W)
            heat_ans_pp = postprocess_heatmap(heat_ans, blur_sigma=6.0, clip_percentile=99.0, gamma=0.7)
            overlay = overlay_heatmap_jet(pil_img, heat_ans_pp, alpha=0.45)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.image(pil_img, caption="Original", use_container_width=True)
            with c2:
                st.image(overlay, caption="Answer-aware heatmap (smoothed + JET)", use_container_width=True)

    elif section == SECTIONS[2]:
        st.subheader("🧭 Spatial Attention (Question-guided Cross-Attention → Image Patches)")
        if attn_heat01 is None:
            st.warning("No attention heatmap computed yet.")
        else:
            cA, cB, cC = st.columns(3)
            with cA:
                blur_sigma = st.slider("Smoothing (sigma)", 0.0, 20.0, 8.0, 0.5)
            with cB:
                clip_p = st.slider("Clip percentile", 90.0, 100.0, 99.0, 0.5)
            with cC:
                gamma = st.slider("Gamma", 0.2, 2.0, 0.7, 0.05)

            attn_pp = postprocess_heatmap(attn_heat01, blur_sigma=blur_sigma, clip_percentile=clip_p, gamma=gamma)
            overlay_attn = overlay_heatmap_jet(pil_img, attn_pp, alpha=0.45)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.image(pil_img, caption="Original", use_container_width=True)
            with c2:
                st.image(overlay_attn, caption="Question-guided attention (smoothed + JET)", use_container_width=True)

            st.markdown("#### ✅ Model-seen square view (better alignment)")
            model_seen = blip_model_seen_image(pil_img, target=480)
            attn_seen = cv2.resize(attn_pp, (480, 480), interpolation=cv2.INTER_CUBIC)
            overlay_seen = overlay_heatmap_jet(model_seen, attn_seen, alpha=0.45)
            st.image(overlay_seen, caption="Overlay on BLIP-like square view", use_container_width=True)

            with st.expander("Debug"):
                st.write(attn_dbg)

    elif section == SECTIONS[3]:
        st.subheader("🧠 Active Attention (اختيار كائن)")
        if not objs_sorted:
            st.warning("لا يوجد كائنات للاختيار.")
        else:
            labels = [f"#{o['i']} - {o['name']} (score={o['score']:.3f})" for o in objs_sorted]
            choice = st.selectbox("اختاري كائنًا للتركيز عليه:", labels, index=0)
            obj = objs_sorted[labels.index(choice)]

            ans_only_en, conf_only = blip_answer_and_conf(obj["img_only"], q_en)
            ans_only_ar = translate(ans_only_en, "en", "ar")

            a1, a2 = st.columns(2, gap="large")
            with a1:
                st.image(obj["img_only"], caption=f"Focus only: {obj['name']}", use_container_width=True)
            with a2:
                st.markdown(f"**إجابة النموذج عند رؤية هذا الكائن فقط:** {ans_only_ar}")
                st.markdown(f"**confidence:** {conf_only:.3f}")

                if _norm_ans(ans_only_en) == _norm_ans(base_ans_en):
                    st.success("✅ نفس الإجابة: هذا الكائن يدعم الإجابة بقوة.")
                else:
                    st.warning("⚠️ إجابة مختلفة: هذا الكائن وحده غير كافٍ لتبرير الإجابة.")

    elif section == SECTIONS[4]:
        st.subheader("🧩 Counterfactual Explanations (Minimal edit that flips the BLIP answer)")
        st.caption("We apply small, controlled edits to a detected region and check when the BLIP answer changes.")

        if not objs_sorted:
            st.warning("لا يوجد كائنات مكتشفة. جرّبي صورة أوضح ليتم توليد Counterfactuals.")
            return

        cA, cB, cC, cD = st.columns([1.0, 1.0, 1.0, 1.0], gap="large")
        with cA:
            max_objs = st.slider("Max objects to test", 1, 10, 6)
        with cB:
            topk = st.slider("Top-K counterfactuals", 1, 8, 5)
        with cC:
            blur_strength = st.selectbox("Blur strength set", ["light", "medium", "strong"], index=1)
        with cD:
            rerun_cf = st.button("🔁 Recompute counterfactuals", use_container_width=True)

        if blur_strength == "light":
            blur_sigmas = [8.0, 10.0]
        elif blur_strength == "strong":
            blur_sigmas = [14.0, 18.0, 22.0]
        else:
            blur_sigmas = [10.0, 14.0, 18.0]

        if rerun_cf or st.session_state.get("_cf_results", None) is None:
            with st.spinner("Searching for minimal answer-flip edits..."):
                cf = compute_counterfactuals(
                    pil_img=pil_img,
                    question_en=q_en,
                    objs_sorted=objs_sorted,
                    max_objects=int(max_objs),
                    blur_sigmas=blur_sigmas,
                    topk=int(topk),
                )
            st.session_state._cf_results = cf
        else:
            cf = st.session_state._cf_results

        st.markdown("#### ✅ Base prediction")
        st.write({"question_en": q_en, "base_answer_en": cf.get("base_ans", base_ans_en), "base_conf": round(float(cf.get("base_conf", base_conf)), 4)})

        cands = cf.get("candidates", [])
        if not cands:
            st.warning("No answer-flip counterfactual found. Try increasing Max objects, or choose strong blur.")
            st.caption(str(cf.get("note", "")))
            return

        st.markdown("#### 🔥 Minimal counterfactuals (answer flips)")
        for idx, cand in enumerate(cands, start=1):
            obj_name = cand.get("object_name", "obj")
            edit_name = cand.get("edit_name", "")
            area = float(cand.get("area_frac", 0.0))
            ans2 = cand.get("ans_edit", "")
            conf2 = float(cand.get("conf_edit", 0.0))
            conf_drop = float(cand.get("conf_drop", 0.0))

            ans2_ar = translate(str(ans2), "en", "ar")

            c1, c2 = st.columns([1.0, 1.2], gap="large")
            with c1:
                st.image(cand["image_edit"], caption=f"#{idx} Edit: {edit_name} | region={obj_name}", use_container_width=True)
            with c2:
                st.markdown(f"**Region:** {obj_name}")
                st.markdown(f"**Edit:** {edit_name}")
                st.markdown(f"**Area fraction (smaller = more minimal):** {area:.4f}")
                st.markdown(f"**New answer (EN):** {ans2}")
                st.markdown(f"**New answer (AR):** {ans2_ar}")
                st.markdown(f"**New confidence:** {conf2:.3f}")
                st.markdown(f"**Confidence change (base - new):** {conf_drop:.3f}")
                st.info("Interpretation: a small change to this region flips the BLIP answer under the same question, so the region is causally important for the model decision in this case.")

    elif section == SECTIONS[5]:
        st.subheader("📝 تفسير نصّي (Explainability Text)")
        top_obj = objs_sorted[0] if objs_sorted else None
        expl = build_expl_ar(q_ar_clean, base_ans_ar, cap_ar, top_obj)
        st.info(expl)

    elif section == SECTIONS[6]:
        st.subheader("📦 Attention-weighted Boxes (Top-K)")
        if attn_heat01 is None:
            st.warning("لا يوجد Attention map. اسألي سؤالًا أولًا.")
            return
        if not objs_sorted:
            st.warning("لا يوجد كائنات من YOLO لحساب متوسط الـ attention داخلها.")
            return

        attn_pp = postprocess_heatmap(attn_heat01, blur_sigma=8.0, clip_percentile=99.0, gamma=0.7)
        topk_boxes = st.slider("Top-K boxes", min_value=1, max_value=10, value=5)
        top_boxes = attention_weighted_boxes_topk(attn_pp, objs_sorted, topk=int(topk_boxes))
        if not top_boxes:
            st.warning("لم أستطع حساب scores للـ boxes.")
            return

        font_path = st.text_input("Arabic font path (optional)", value="C:/Windows/Fonts/arial.ttf")
        drawn = draw_boxes_on_pil_ar(
            pil_img,
            top_boxes,
            score_key="attn_score",
            font_path=font_path.strip() if font_path.strip() else None,
            font_size=22,
        )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(pil_img, caption="Original", use_container_width=True)
        with c2:
            st.image(drawn, caption="Top-K boxes ranked by mean attention inside the box", use_container_width=True)

        st.markdown("### 🔝 Top-K list")
        for i, o in enumerate(top_boxes, start=1):
            x1, y1, x2, y2 = o["bbox"]
            name_en = o.get("name", "")
            name_ar = yolo_label_ar(name_en)
            st.write(
                f"{i}) **{name_ar}** ({name_en}) | attn_score={float(o['attn_score']):.4f} | det_conf={float(o['det_conf']):.2f} | bbox=({x1},{y1})-({x2},{y2})"
            )

    elif section == SECTIONS[7]:
        st.subheader("🧪 Vanilla Gradient (Pixel Saliency)")
        st.caption("Computed from |∂Loss/∂Pixels| for generated BLIP answer tokens.")
        with st.spinner("Computing Vanilla Gradient saliency..."):
            vg_heat01, vg_dbg = vanilla_gradient_saliency_heatmap(pil_img, q_en, max_new_tokens=16, num_beams=3)

        cA, cB, cC = st.columns(3)
        with cA:
            blur_sigma = st.slider("Smoothing (sigma)", 0.0, 20.0, 8.0, 0.5, key="vg_sigma")
        with cB:
            clip_p = st.slider("Clip percentile", 90.0, 100.0, 99.0, 0.5, key="vg_clip")
        with cC:
            gamma = st.slider("Gamma", 0.2, 2.0, 0.7, 0.05, key="vg_gamma")

        vg_pp = postprocess_heatmap(vg_heat01, blur_sigma=blur_sigma, clip_percentile=clip_p, gamma=gamma)
        overlay_vg_orig = overlay_heatmap_jet(pil_img, vg_pp, alpha=0.45)

        model_seen_384 = blip_model_seen_image(pil_img, target=384)
        vg_seen_384 = cv2.resize(vg_pp, (384, 384), interpolation=cv2.INTER_CUBIC)
        overlay_vg_seen = overlay_heatmap_jet(model_seen_384, vg_seen_384, alpha=0.45)

        st.markdown("#### ✅ Model-seen 384×384 (alignment)")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(model_seen_384, caption="BLIP model-seen (384×384 crop)", use_container_width=True)
        with c2:
            st.image(overlay_vg_seen, caption="Vanilla Gradient overlay (aligned)", use_container_width=True)

        st.markdown("#### Original image view (approx)")
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.image(pil_img, caption="Original", use_container_width=True)
        with c4:
            st.image(overlay_vg_orig, caption="Overlay on original", use_container_width=True)

        with st.expander("Debug"):
            st.write(vg_dbg)

    elif section == SECTIONS[8]:
        st.subheader("🎯 Grad-CAM (BLIP ViT aligned)")
        st.caption("Target = loss of generated answer tokens. Computed on model-seen 384×384 crop.")
        with st.spinner("Computing BLIP ViT Grad-CAM..."):
            cam384, cam_dbg = blip_vit_gradcam_heatmap(pil_img, q_en, max_new_tokens=16, num_beams=3)

        cA, cB, cC = st.columns(3)
        with cA:
            blur_sigma = st.slider("Smoothing (sigma)", 0.0, 20.0, 8.0, 0.5, key="gc_sigma")
        with cB:
            clip_p = st.slider("Clip percentile", 90.0, 100.0, 99.0, 0.5, key="gc_clip")
        with cC:
            gamma = st.slider("Gamma", 0.2, 2.0, 0.7, 0.05, key="gc_gamma")

        cam_pp = postprocess_heatmap(cam384, blur_sigma=blur_sigma, clip_percentile=clip_p, gamma=gamma)
        model_seen_384 = blip_model_seen_image(pil_img, target=384)
        overlay_seen = overlay_heatmap_jet(model_seen_384, cam_pp, alpha=0.45)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.image(model_seen_384, caption="BLIP model-seen (384×384 crop)", use_container_width=True)
        with c2:
            st.image(overlay_seen, caption="Grad-CAM overlay (ViT patch-token CAM)", use_container_width=True)

        with st.expander("Debug"):
            st.write(cam_dbg)

    
    elif section == SECTIONS[10]:
        st.subheader("🧩 LIME for VQA (local surrogate explanation)")
        st.caption("Model-agnostic, uses superpixel perturbations to explain which regions support the current answer.")
        if lime_image is None or slic is None:
            st.warning("LIME is not installed in this environment.")
            st.write("To enable it, add these to requirements.txt: `lime` and `scikit-image`.")
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            num_samples = st.slider("Perturbation samples", 200, 1500, 700, 50)
        with c2:
            num_features = st.slider("Top superpixels", 3, 25, 10, 1)
        with c3:
            rerun_lime = st.button("🔁 Recompute LIME", use_container_width=True)

        if rerun_lime or st.session_state.get("_lime_overlay", None) is None:
            with st.spinner("Running LIME (this can be slow on CPU)..."):
                overlay_lime, lime_dbg = lime_vqa_explanation(
                    pil_img=pil_img,
                    question_en=q_en,
                    base_answer_en=base_ans_en,
                    num_samples=int(num_samples),
                    num_features=int(num_features),
                )
            st.session_state._lime_overlay = overlay_lime
            st.session_state._lime_dbg = lime_dbg

        overlay_lime = st.session_state.get("_lime_overlay", None)
        lime_dbg = st.session_state.get("_lime_dbg", None)

        if overlay_lime is None:
            st.warning("LIME did not return an explanation. Try increasing samples.")
            if lime_dbg:
                st.write(lime_dbg)
            return

        cA, cB = st.columns(2, gap="large")
        with cA:
            st.image(pil_img, caption="Original", use_container_width=True)
        with cB:
            st.image(overlay_lime, caption="LIME overlay (superpixels supporting the base answer)", use_container_width=True)

        with st.expander("LIME debug"):
            st.write(lime_dbg)

        st.info("Caveat: LIME depends on how perturbations and superpixels are generated, so results can be unstable. Use it as a qualitative explanation.")

    elif section == SECTIONS[11]:
        st.subheader("🧩 SHAP for multimodal inputs (superpixels + token groups)")
        st.caption("Explains contributions to the BASE answer score: image superpixels + question token-groups.")

        if shap is None or slic is None:
            st.warning("SHAP is not installed in this environment.")
            st.write("To enable it, add these to requirements.txt: `shap`, `scikit-learn`, `scipy`, `numba`, `scikit-image`.")
            return

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ns_img = st.slider("Image SHAP samples", 80, 600, 250, 10)
        with c2:
            ns_txt = st.slider("Text SHAP samples", 80, 600, 250, 10)
        with c3:
            group_size = st.slider("Token group size", 1, 4, 2, 1)
        with c4:
            top_groups = st.slider("Top token groups", 5, 20, 10, 1)

        rerun = st.button("🔁 Recompute SHAP", use_container_width=True)

        if rerun or st.session_state.get("_shap_overlay", None) is None:
            with st.spinner("Running SHAP (can be slow on CPU)..."):
                overlay_shap, shap_dbg = shap_multimodal_explanation(
                    pil_img=pil_img,
                    question_en=q_en,
                    base_answer_en=base_ans_en,
                    num_samples_img=int(ns_img),
                    num_samples_txt=int(ns_txt),
                    group_size=int(group_size),
                    top_groups=int(top_groups),
                )
            st.session_state._shap_overlay = overlay_shap
            st.session_state._shap_dbg = shap_dbg

        overlay_shap = st.session_state.get("_shap_overlay", None)
        shap_dbg = st.session_state.get("_shap_dbg", None)

        if overlay_shap is None or shap_dbg is None:
            st.warning("SHAP did not return an explanation. Try increasing samples.")
            if shap_dbg:
                st.write(shap_dbg)
            return

        cA, cB = st.columns(2, gap="large")
        with cA:
            st.image(pil_img, caption="Original", use_container_width=True)
        with cB:
            st.image(overlay_shap, caption="SHAP image overlay (positive contributions to the base answer)", use_container_width=True)

        st.markdown("### 🧾 Token-group contributions (question side)")
        top = shap_dbg.get("text", {}).get("top_groups", [])
        if not top:
            st.info("No token-group contributions returned.")
        else:
            # Display as a simple table-like list
            for row in top:
                st.write(f"- **{row.get('text','')}** | SHAP = {float(row.get('shap_value', 0.0)):.6f}")

        with st.expander("SHAP debug"):
            st.write(shap_dbg)

        st.info("Caveat: Kernel SHAP is an approximation and can be slow. Treat it as qualitative: which regions and which question phrases support the model’s chosen answer.")


# ============================================================
# 15) App main
# ============================================================
logo_big = _find_logo_data_uri("big")
logo_small = _find_logo_data_uri("small")

st.session_state.logo_big = logo_big
st.session_state.logo_small = logo_small




logo_big = _find_logo_data_uri("big")
logo_small = _find_logo_data_uri("small")

st.session_state.logo_big = logo_big
st.session_state.logo_small = logo_small

inject_pro_ui(logo_big, logo_small)

if not st.session_state.get("started", False):
    page_landing()

elif st.session_state.get("page", "About") == "About":
    app_header()
    page_about()

elif st.session_state.page == "How to Test":
    app_header()
    page_how_to_test()

elif st.session_state.page == "System":
    app_header()
    page_demo()


