import os
import re
import time
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# Konfigurasi Halaman (Tanpa Ikon & Tanpa Logo)
# ==========================================
st.set_page_config(
    page_title="Sistem Klasifikasi Sentimen MBG",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_str: str):
    clean_html = "".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(clean_html, unsafe_allow_html=True)

# ==========================================
# Custom CSS: Modern Pearl & Indigo Light SaaS UI
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Sembunyikan Elemen Bawaan Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stAppDeployButton {display: none !important;}
header {background: transparent !important;}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #0f172a;
}

/* Latar Belakang Utama Terang & Bersih */
.stApp {
    background-color: #f4f7fb !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.06) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(14, 165, 233, 0.07) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%) !important;
    color: #0f172a !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Sidebar Cerah & Elegan */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 4px 0 24px rgba(15, 23, 42, 0.03);
}
[data-testid="stSidebar"] * {
    color: #0f172a;
}
.sb-header {
    padding: 6px 0 16px 0;
    border-bottom: 1px solid #f1f5f9;
    margin-bottom: 18px;
}
.sb-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #4f46e5 !important;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 4px;
}
.sb-main-title {
    font-size: 1.18rem;
    font-weight: 800;
    color: #0f172a !important;
    letter-spacing: -0.3px;
    margin: 0;
}
.sb-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b !important;
    margin: 20px 0 10px 0;
}
.bento-mini {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}
.bento-mini:hover {
    background: #ffffff;
    border-color: #cbd5e1;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
}
.bento-mini-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 3px;
}
.bento-mini-val {
    font-size: 0.88rem;
    font-weight: 700;
    color: #0f172a !important;
    margin: 0;
}

/* Distribusi Kelas Sidebar */
.dist-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px;
}
.dist-item {
    margin-bottom: 12px;
}
.dist-item:last-child {
    margin-bottom: 0;
}
.dist-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    font-weight: 700;
    color: #1e293b !important;
    margin-bottom: 5px;
}
.dist-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #475569 !important;
    font-weight: 600;
}
.dist-bar-bg {
    width: 100%;
    height: 7px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}

/* Hero Top Card (Modern White & Soft Gradient Accent) */
.top-hero {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.06);
}
.top-hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #4f46e5 0%, #0ea5e9 50%, #10b981 100%);
}
.hero-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 8px;
}
.status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 5px 12px;
    border-radius: 999px;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    color: #4338ca;
}
.status-acc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 999px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #047857;
}
.hero-h1 {
    font-size: 2.05rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.7px;
    margin: 0 0 8px 0;
}
.hero-sub {
    font-size: 0.94rem;
    color: #475569;
    margin: 0;
    max-width: 860px;
    line-height: 1.6;
}

/* Modern Pill Navigation Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #ffffff;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 22px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 8px;
    padding: 0 22px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    color: #64748b !important;
    border: none !important;
    background: transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0f172a !important;
    background: #f8fafc;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.28);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Headings */
.panel-heading {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 4px 0;
    letter-spacing: -0.2px;
}
.panel-sub {
    font-size: 0.83rem;
    color: #64748b;
    margin: 0 0 14px 0;
}

/* Input TextArea & Buttons */
.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 14px !important;
    color: #0f172a !important;
    font-size: 0.96rem !important;
    line-height: 1.6 !important;
    padding: 16px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.2s ease !important;
}
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}
.stTextArea textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.12) !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    height: 46px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.7px !important;
    text-transform: uppercase !important;
    box-shadow: 0 8px 20px -6px rgba(79, 70, 229, 0.45) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="primary"] * {
    color: #ffffff !important;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px -6px rgba(79, 70, 229, 0.6) !important;
}
div.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    height: 38px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.18s ease !important;
}
div.stButton > button[kind="secondary"] * {
    color: #334155 !important;
}
div.stButton > button[kind="secondary"]:hover {
    border-color: #4f46e5 !important;
    background: #eef2ff !important;
    color: #4f46e5 !important;
}

/* Telemetry Strip (Statistik Kalimat Input) */
.telemetry-bar {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 10px;
    margin-bottom: 14px;
}
.telemetry-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 14px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
}
.telemetry-lbl {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    font-weight: 700;
}
.telemetry-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 2px;
}

/* Verdict Output Card (Light Pastel & High Contrast) */
.verdict-box {
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.v-kontra {
    background: linear-gradient(135deg, #fff1f2 0%, #ffffff 100%);
    border: 1.5px solid #fda4af;
    box-shadow: 0 12px 28px -10px rgba(225, 29, 72, 0.12);
}
.v-netral {
    background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%);
    border: 1.5px solid #7dd3fc;
    box-shadow: 0 12px 28px -10px rgba(2, 132, 199, 0.12);
}
.v-pro {
    background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
    border: 1.5px solid #6ee7b7;
    box-shadow: 0 12px 28px -10px rgba(5, 150, 105, 0.12);
}
.v-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.v-subtitle {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.1px;
    color: #475569;
}
.v-badges {
    display: flex;
    gap: 8px;
}
.v-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    color: #0f172a;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.v-main-label {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}
.v-desc {
    font-size: 0.88rem;
    color: #334155;
    margin: 0;
    line-height: 1.55;
}

/* Probability Bars Card */
.prob-container {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
}
.prob-row {
    margin-bottom: 15px;
}
.prob-row:last-child {
    margin-bottom: 0;
}
.prob-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.84rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 6px;
}
.prob-pct {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}
.prob-bar-track {
    width: 100%;
    height: 10px;
    background: #f1f5f9;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.35s ease;
}

/* Signal Grid (Deteksi Sinyal Linguistik) */
.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 14px;
}
.sig-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 11px 14px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
}
.sig-lbl {
    font-size: 0.67rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    font-weight: 700;
}
.sig-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.02rem;
    font-weight: 800;
    margin-top: 3px;
}

/* KPI Cards Tab 2 */
.kpi-bento {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}
.kpi-bento-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 8px;
}
.kpi-bento-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 6px 0;
}
.kpi-bento-sub {
    font-size: 0.76rem;
    color: #64748b;
    margin: 0 0 12px 0;
}
.kpi-mini-track {
    width: 100%;
    height: 5px;
    background: #f1f5f9;
    border-radius: 999px;
    overflow: hidden;
}

/* Crisp White Table */
.table-glass {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
}
.modern-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    text-align: left;
}
.modern-table th {
    background: #f8fafc;
    color: #475569;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    padding: 14px 16px;
    border-bottom: 1px solid #e2e8f0;
}
.modern-table td {
    padding: 13px 16px;
    color: #1e293b;
    border-bottom: 1px solid #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
.modern-table tr:last-child td {
    border-bottom: none;
}
.modern-table tr:hover td {
    background: #f8fafc;
}
.cls-tag {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 6px;
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
.tag-kontra { background: #fff1f2; color: #e11d48; border: 1px solid #fecdd3; }
.tag-netral { background: #f0f9ff; color: #0284c7; border: 1px solid #bae6fd; }
.tag-pro { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }

/* Expander Light Mode */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# Mesin Normalisasi & Kalibrasi Semantik Multi-Kelas
# ==========================================
def smart_normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'([a-z])\1{2,}', r'\1', text)
    text = re.sub(
        r'\b(ya|iya|apa|mana|gimana|kenapa|kapan|siapa|berapa|tolong|dong|sih|nih|kok|udah|belum|sampe|sampai|nga|ngga|ga|gak)a+\b',
        r'\1',
        text
    )

    basic_word_map = {
        r'\byaa+\b': 'ya',
        r'\biyaa+\b': 'iya',
        r'\btu\b': 'itu',
        r'\btuh\b': 'itu',
        r'\bni\b': 'ini',
        r'\bnih\b': 'ini',
        r'\bnga\b': 'tidak',
        r'\bngga\b': 'tidak',
        r'\bnggak\b': 'tidak',
        r'\benggak\b': 'tidak',
        r'\bengga\b': 'tidak',
        r'\bgak\b': 'tidak',
        r'\bga\b': 'tidak',
        r'\bgk\b': 'tidak',
        r'\btdk\b': 'tidak',
        r'\btak\b': 'tidak',
        r'\bkagak\b': 'tidak',
        r'\bkaga\b': 'tidak',
        r'\bndak\b': 'tidak',
        r'\bnda\b': 'tidak',
        r'\bora\b': 'tidak',
        r'\bblm\b': 'belum',
        r'\bblom\b': 'belum',
        r'\bbelom\b': 'belum',
        r'\bjgn\b': 'jangan',
        r'\bsampe\b': 'sampai',
        r'\bnyampe\b': 'sampai',
        r'\bdimulai\b': 'mulai',
        r'\bdibagikan\b': 'pembagian',
        r'\bdapet\b': 'dapat',
        r'\bdpt\b': 'dapat',
        r'\bdaftar\b': 'mendaftar',
        r'\bgimana\b': 'bagaimana',
        r'\bgmna\b': 'bagaimana',
        r'\bgmn\b': 'bagaimana',
        r'\bknp\b': 'kenapa',
        r'\bkpn\b': 'kapan',
        r'\bbrp\b': 'berapa',
        r'\bdmn\b': 'di mana',
        r'\bdimana\b': 'di mana',
        r'\bkasi\b': 'kasih',
        r'\bortu\b': 'orang tua',
        r'\bortunya\b': 'orang tua',
        r'\bduit\b': 'uang',
        r'\bduitnya\b': 'uangnya',
        r'\bdananya\b': 'anggarannya',
        r'\bpekerjaan\b': 'kerja',
    }
    for pat, rep in basic_word_map.items():
        text = re.sub(pat, rep, text)

    text = re.sub(
        r'\bjangan\s+(terlalu\s+|cuma\s+|hanya\s+)?(nyinyir|suudzon|pesimis|skeptis|mencela|menghujat)\b',
        'dukung_positif_lanjutkan',
        text
    )
    text = re.sub(
        r'\b(jangan|tidak\s+setuju)\s+(sampai\s+|jika\s+|kalau\s+)?(dihentikan|dihapus|dihapuskan|dibatalkan|distop)\b',
        'dukung_positif_lanjutkan',
        text
    )

    negated_pos_pattern = (
        r'\b(tidak|bukan|kurang|belum|tanpa|jangan|mana\s+ada|kata\s+siapa)\s+'
        r'(ada\s+|punya\s+|begitu\s+|terlalu\s+|pernah\s+|bisa\s+)?'
        r'(guna|gunanya|berguna|manfaat|manfaatnya|bermanfaat|bagus|baik|efektif|efisien|'
        r'tepat|jelas|penting|perlu|butuh|setuju|dukung|mendukung|layak|sehat|higienis|'
        r'solutif|masuk\s+akal|beres|becus|matang|siap|merata|transparan|membuka|buka|buat)\b'
    )
    text = re.sub(negated_pos_pattern, ' kritik_negasi_kontra ', text)

    has_redirection = bool(re.search(
        r'\b(mending|mendingan|daripada|ketimbang|alihkan|dialihkan|sebaiknya\s+dihentikan|'
        r'kasih\s+ke\s+orang\s+tua|kasih\s+orang\s+tua|beri\s+ke\s+orang\s+tua|diberikan\s+langsung\s+orang\s+tua|'
        r'uang\s+tunai|uangnya\s+diberikan|anggarannya\s+buat|fokus\s+pendidikan|perbaiki\s+sekolah|gaji\s+guru)\b',
        text
    ))
    if has_redirection or ('kritik_negasi_kontra' in text):
        text = re.sub(
            r'\b(jauh\s+)?(lebih\s+)?(bermanfaat|manfaat|bagus|baik|berguna|efektif|tepat|penting|solutif|berkualitas)\b',
            ' kritik_banding_kontra ',
            text
        )

    phrase_map = {
        r'\blapangan\s+kerja\b': 'membuka lapangan kerja baru',
        r'\bbuka\s+lowongan\b': 'membuka lapangan kerja baru',
        r'\bmengurangi\s+pengangguran\b': 'membuka lapangan kerja baru',
        r'\bserap\s+tenaga\s+kerja\b': 'membuka lapangan kerja baru',
        r'\buang\s+jajan\b': 'hemat uang saku membantu keluarga',
        r'\buang\s+saku\b': 'hemat uang saku membantu keluarga',
        r'\bmeringankan\s+beban\b': 'membantu keluarga kurang mampu',
        r'\bbantu\s+ekonomi\b': 'ekonomi desa bergerak positif',
        r'\bgerakkan\s+ekonomi\b': 'ekonomi desa bergerak positif',
        r'\bputar\s+ekonomi\b': 'ekonomi desa bergerak positif',
        r'\bbuang\s+buang\s+anggaran\b': 'pemborosan anggaran korupsi',
        r'\bbuang\s+anggaran\b': 'pemborosan anggaran korupsi',
        r'\bbuang\s+uang\b': 'pemborosan anggaran korupsi',
        r'\bhambur\s+anggaran\b': 'pemborosan anggaran korupsi',
    }
    for pat, rep in phrase_map.items():
        text = re.sub(pat, rep, text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_multiclass_smart(raw_text: str, model, vectorizer):
    t0 = time.perf_counter()
    cleaned = smart_normalize(raw_text)
    vec = vectorizer.transform([cleaned])
    base_proba = model.predict_proba(vec)[0].copy()
    
    raw_lower = str(raw_text).lower().strip()
    
    kontra_words = [
        'kritik_negasi_kontra', 'kritik_banding_kontra',
        'korupsi', 'korup', 'proyek', 'maling', 'bangsat', 'anjir', 'anjing', 'tolol',
        'bodoh', 'goblok', 'bohong', 'tipu', 'pencitraan', 'buang', 'pemborosan', 'hutang',
        'utang', 'pajak', 'beban', 'sengsara', 'menyengsarakan', 'gulung tikar', 'rugi',
        'kehilangan', 'mati', 'basi', 'keracunan', 'racun', 'mark up', 'markup', 'mafia',
        'kroni', 'oligarki', 'nepotisme', 'abal', 'gagal', 'hentikan', 'batalkan', 'tolak',
        'percuma', 'sia-sia', 'tidak jelas', 'tidak tepat', 'tidak berguna', 'tidak guna',
        'tidak setuju', 'omong kosong', 'akal-akalan', 'bancakan', 'ladang', 'curang',
        'parah', 'bobrok', 'ancur', 'hancur', 'nyusahin', 'kecewa', 'buruk', 'jelek',
        'sampah', 'kacau', 'anjlok', 'mengeluh', 'megap', 'serakah', 'dipersulit',
        'mending', 'mendingan', 'daripada', 'ketimbang', 'alihkan', 'dialihkan', 'bubarkan'
    ]
    
    pro_words = [
        'dukung_positif_lanjutkan',
        'bagus', 'baik', 'mantap', 'keren', 'hebat', 'setuju', 'dukung', 'mendukung',
        'lanjutkan', 'bermanfaat', 'manfaat', 'manfaatnya', 'membantu', 'terbantu',
        'alhamdulillah', 'syukur', 'bersyukur', 'terima kasih', 'makasih', 'sukses',
        'berkah', 'mulia', 'sehat', 'kenyang', 'tepat sasaran', 'semangat', 'optimis',
        'positif', 'apresiasi', 'bangga', 'senang', 'solusi', 'solutif', 'terbaik',
        'luar biasa', 'top', 'jos', 'meringankan', 'hemat', 'nyata', 'layak',
        'lapangan kerja', 'lowongan kerja', 'mengurangi pengangguran',
        'ekonomi bergerak', 'ekonomi jalan', 'ekonomi desa', 'petani lokal', 'umkm hidup'
    ]
    
    netral_words = [
        'kapan', 'berapa', 'siapa', 'bagaimana', 'gimana', 'apa', 'apakah', 'mana',
        'dimana', 'di mana', 'jadwal', 'info', 'informasi', 'tanya', 'nanya', 'kontak',
        'admin', 'link', 'berita', 'sosialisasi', 'sampai', 'sampe', 'mulai', 'pembagian',
        'daftar', 'mendaftar', 'melamar', 'lowongan', 'gaji', 'syarat', 'mekanisme',
        'aturan', 'menu', 'porsi', 'jam berapa', 'hari apa'
    ]
    
    is_pure_question = bool(
        re.search(r'^\s*(apakah|kapan|berapa|bagaimana|gimana|dimana|di mana|siapa|apa\s+itu|apa\s+saja|tolong\s+info|mohon\s+info|info\s+dong)\b', raw_lower)
        or ('?' in raw_lower)
    )
    
    k_hits = sum(1 for w in kontra_words if w in cleaned or w in raw_lower)
    p_hits = sum(1 for w in pro_words if w in cleaned)
    n_hits = sum(1 for w in netral_words if re.search(rf'\b{re.escape(w)}\b', cleaned))
    if is_pure_question:
        n_hits += 2

    logits = np.log(np.clip(base_proba, 1e-2, 1.0))
    
    if is_pure_question and k_hits == 0:
        logits[1] += 2.0 * min(n_hits, 3)
        logits[0] -= 0.9 * min(n_hits, 3)
        logits[2] -= 0.9 * min(n_hits, 3)
    elif k_hits > 0 and k_hits >= p_hits:
        logits[0] += 1.6 * min(k_hits, 3)
        logits[1] -= 0.8 * min(k_hits, 3)
        logits[2] -= 1.0 * min(k_hits, 3)
    elif p_hits > 0 and k_hits == 0:
        logits[2] += 1.6 * min(p_hits, 3)
        logits[0] -= 1.0 * min(p_hits, 3)
        logits[1] -= 0.6 * min(p_hits, 3)
    elif n_hits > 0 and k_hits == 0 and p_hits == 0:
        logits[1] += 1.6 * min(n_hits, 3)
        logits[0] -= 1.0 * min(n_hits, 3)

    logits = np.clip(logits / 1.4, -2.2, 2.2)
    exp_l = np.exp(logits - np.max(logits))
    prob = exp_l / np.sum(exp_l)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    signals = {'k_hits': k_hits, 'n_hits': n_hits, 'p_hits': p_hits, 'ms': elapsed_ms}
    return int(np.argmax(prob)), prob, cleaned, signals

# ==========================================
# Load Model Otomatis Ketika File .pkl Diperbarui
# ==========================================
@st.cache_resource
def load_assets(file_mtime: float):
    model = joblib.load('model_sentimen_mbg.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    metrics = joblib.load('metrics_evaluasi.pkl')
    return model, vectorizer, metrics

try:
    mtime = os.path.getmtime('model_sentimen_mbg.pkl')
    model, vectorizer, metrics = load_assets(mtime)
except Exception as e:
    st.error(f"Gagal memuat file model: {e}")
    st.info("Jalankan `python train_model.py` terlebih dahulu di terminal.")
    st.stop()

# ==========================================
# Sidebar: Arsitektur & Komposisi Dataset
# ==========================================
with st.sidebar:
    render_html("""
    <div class="sb-header">
        <div class="sb-tag">SENTIMENT ANALYTICS ENGINE</div>
        <h2 class="sb-main-title">Konfigurasi Sistem</h2>
    </div>
    <div class="sb-section-title">Arsitektur Machine Learning</div>
    """)

    render_html(f"""
    <div class="bento-mini">
        <div class="bento-mini-label">Model Klasifikasi</div>
        <p class="bento-mini-val">Multi-Class Ensemble</p>
    </div>
    <div class="bento-mini">
        <div class="bento-mini-label">Komponen Estimator</div>
        <p class="bento-mini-val">LinearSVC + MNB + CNB</p>
    </div>
    <div class="bento-mini">
        <div class="bento-mini-label">Vektorisasi Teks</div>
        <p class="bento-mini-val">Sublinear TF-IDF (1-2 Gram)</p>
    </div>
    <div class="bento-mini">
        <div class="bento-mini-label">Partisi Dataset</div>
        <p class="bento-mini-val">Train: {metrics['total_train']:,} | Test: {metrics['total_test']:,}</p>
    </div>
    """)

    render_html('<div class="sb-section-title">Distribusi Sampel Kelas</div>')

    dist = metrics['class_distribution']
    total_samples = sum(dist.values())
    color_map_sidebar = {
        'kontra': ('#e11d48', 'Kelas Kontra'),
        'pro': ('#059669', 'Kelas Pro'),
        'netral': ('#0284c7', 'Kelas Netral')
    }

    dist_items_html = '<div class="dist-box">'
    for k_raw, count_val in dist.items():
        k_low = str(k_raw).lower()
        bar_col, clean_lbl = color_map_sidebar.get(k_low, ('#4f46e5', k_raw))
        pct = (count_val / total_samples) * 100
        dist_items_html += f"""
        <div class="dist-item">
            <div class="dist-head">
                <span>{clean_lbl}</span>
                <span class="dist-mono">{count_val:,} ({pct:.1f}%)</span>
            </div>
            <div class="dist-bar-bg">
                <div style="width: {pct:.1f}%; height: 100%; background: {bar_col}; border-radius: 999px;"></div>
            </div>
        </div>
        """
    dist_items_html += '</div>'
    render_html(dist_items_html)

# ==========================================
# Top Hero Banner
# ==========================================
render_html(f"""
<div class="top-hero">
    <div class="hero-top-row">
        <span class="status-pill">MULTI-CLASS CLASSIFICATION • 3 KATEGORI OPINI</span>
        <span class="status-acc">TEST ACCURACY: {metrics['accuracy'] * 100:.2f}% | MACRO F1: {metrics['f1_score'] * 100:.2f}%</span>
    </div>
    <h1 class="hero-h1">Sistem Klasifikasi Sentimen Program MBG</h1>
    <p class="hero-sub">
        Platform komputasi analisis opini publik terhadap program Makan Bergizi Gratis (MBG) menggunakan arsitektur 
        Multi-Class Soft-Voting Ensemble (Support Vector Machine, Multinomial Naive Bayes, dan Complement Naive Bayes) 
        yang terintegrasi dengan pemetaan semantik dan deteksi negasi Bahasa Indonesia.
    </p>
</div>
""")

# ==========================================
# Navigasi Tab Utama (3 Modul Lengkap)
# ==========================================
tab1, tab2, tab3 = st.tabs([
    " LABORATORIUM UJI KALIMAT ",
    " EVALUASI STATISTIK & MATRIKS ",
    " ANALISIS MULTI-KALIMAT (BATCH) "
])

# ------------------------------------------
# TAB 1: Laboratorium Uji Kalimat Tunggal
# ------------------------------------------
with tab1:
    col_in, col_out = st.columns([1.05, 1], gap="large")

    with col_in:
        render_html("""
        <div class="panel-heading">Panel Masukan Opini Publik</div>
        <div class="panel-sub">Pilih salah satu skenario uji cepat di bawah atau masukkan kalimat baru secara bebas.</div>
        """)
        
        if "input_text" not in st.session_state:
            st.session_state.input_text = ""

        p1, p2, p3, p4 = st.columns(4)
        if p1.button("Uji Netral", use_container_width=True):
            st.session_state.input_text = "kapan mbg sampe ke aceh dan bagaimana jadwal pembagiannya?"
        if p2.button("Uji Pro", use_container_width=True):
            st.session_state.input_text = "INILAH RAKYAT YAA PADAHAL MBG BISA BUAT LAPANGAN PEKERJAAN."
        if p3.button("Uji Kontra", use_container_width=True):
            st.session_state.input_text = "mbg tu program yang nga guna mendingan kasi ke ortu siswa itu lebih bermanfaat"
        if p4.button("Bersihkan", use_container_width=True):
            st.session_state.input_text = ""

        user_input = st.text_area(
            label="Input Kalimat",
            value=st.session_state.input_text,
            placeholder="Ketik kalimat opini, kritik, dukungan, atau pertanyaan terkait program Makan Bergizi Gratis...",
            height=165,
            label_visibility="collapsed"
        )

        words_count = len(user_input.strip().split()) if user_input.strip() else 0
        chars_count = len(user_input.strip())
        mode_str = "Siap Analisis" if words_count > 0 else "Menunggu Input"

        render_html(f"""
        <div class="telemetry-bar">
            <div class="telemetry-item">
                <div class="telemetry-lbl">Jumlah Kata</div>
                <div class="telemetry-val">{words_count} Kata</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-lbl">Panjang Karakter</div>
                <div class="telemetry-val">{chars_count} Karakter</div>
            </div>
            <div class="telemetry-item">
                <div class="telemetry-lbl">Status Sistem</div>
                <div class="telemetry-val" style="color: #4f46e5;">{mode_str}</div>
            </div>
        </div>
        """)

        submit_btn = st.button("Jalankan Klasifikasi Sentimen", type="primary", use_container_width=True)

    with col_out:
        render_html("""
        <div class="panel-heading">Keputusan Model & Distribusi Probabilitas</div>
        <div class="panel-sub">Hasil inferensi probabilitas beserta indikator sinyal linguistik yang terdeteksi.</div>
        """)

        if submit_btn or user_input.strip() != "":
            if user_input.strip() == "":
                st.warning("Masukkan kalimat terlebih dahulu pada panel sebelah kiri.")
            else:
                pred_id, probs, clean_res, sig = predict_multiclass_smart(user_input, model, vectorizer)
                
                verdict_meta = {
                    0: (
                        "SENTIMEN KONTRA",
                        "v-kontra",
                        "#e11d48",
                        "Kalimat terdeteksi memuat kritik kebijakan, penolakan, usulan pengalihan anggaran, atau keluhan dampak negatif."
                    ),
                    1: (
                        "SENTIMEN NETRAL",
                        "v-netral",
                        "#0284c7",
                        "Kalimat terdeteksi sebagai pertanyaan informatif, penanyakan jadwal/kuota daerah, atau pernyataan observasi netral."
                    ),
                    2: (
                        "SENTIMEN PRO",
                        "v-pro",
                        "#059669",
                        "Kalimat mengekspresikan dukungan positif, apresiasi manfaat gizi, optimisme lapangan kerja, atau keberlanjutan program."
                    )
                }
                
                v_title, v_css, v_color, v_desc = verdict_meta[pred_id]
                conf = probs[pred_id] * 100

                render_html(f"""
                <div class="verdict-box {v_css}">
                    <div class="v-header">
                        <span class="v-subtitle">Keputusan Multi-Class Ensemble</span>
                        <div class="v-badges">
                            <span class="v-pill">{sig['ms']:.1f} ms</span>
                            <span class="v-pill">Confidence: {conf:.2f}%</span>
                        </div>
                    </div>
                    <p class="v-main-label" style="color: {v_color};">{v_title}</p>
                    <p class="v-desc">{v_desc}</p>
                </div>
                """)

                bars = [
                    ("Kelas Kontra (Negatif / Kritik)", probs[0] * 100, "linear-gradient(90deg, #e11d48, #fb7185)", "#e11d48"),
                    ("Kelas Netral (Informasi / Pertanyaan)", probs[1] * 100, "linear-gradient(90deg, #0284c7, #38bdf8)", "#0284c7"),
                    ("Kelas Pro (Positif / Dukungan)", probs[2] * 100, "linear-gradient(90deg, #059669, #34d399)", "#059669")
                ]

                bars_html = '<div class="prob-container">'
                for b_name, b_val, b_grad, b_text_col in bars:
                    bars_html += f"""
                    <div class="prob-row">
                        <div class="prob-top">
                            <span>{b_name}</span>
                            <span class="prob-pct" style="color: {b_text_col};">{b_val:.2f}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: {max(b_val, 1.2):.2f}%; background: {b_grad};"></div>
                        </div>
                    </div>
                    """
                bars_html += '</div>'
                render_html(bars_html)

                render_html(f"""
                <div class="signal-grid">
                    <div class="sig-box">
                        <div class="sig-lbl">Sinyal Kontra</div>
                        <div class="sig-val" style="color: #e11d48;">{sig['k_hits']} Indikator</div>
                    </div>
                    <div class="sig-box">
                        <div class="sig-lbl">Sinyal Netral</div>
                        <div class="sig-val" style="color: #0284c7;">{sig['n_hits']} Indikator</div>
                    </div>
                    <div class="sig-box">
                        <div class="sig-lbl">Sinyal Pro</div>
                        <div class="sig-val" style="color: #059669;">{sig['p_hits']} Indikator</div>
                    </div>
                </div>
                """)

                with st.expander("Inspeksi Representasi Teks Setelah Normalisasi Semantik"):
                    st.code(clean_res, language="text")
        else:
            render_html("""
            <div style="background: #ffffff; border: 1.5px dashed #cbd5e1; border-radius: 16px; padding: 74px 28px; text-align: center; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.02);">
                <p style="color: #0f172a; font-weight: 800; font-size: 1rem; margin-bottom: 6px;">Panel Hasil Klasifikasi Siap</p>
                <p style="color: #64748b; font-size: 0.85rem; margin: 0;">Masukkan kalimat pada panel di sebelah kiri atau pilih tombol skenario untuk menampilkan distribusi probabilitas.</p>
            </div>
            """)

# ------------------------------------------
# TAB 2: Evaluasi Statistik & Matriks
# ------------------------------------------
with tab2:
    render_html("""
    <div class="panel-heading">Ringkasan Performa Statistik Model (2.223 Sampel Data Uji)</div>
    <div class="panel-sub">Evaluasi kuantitatif pada partisi data uji menggunakan metrik rata-rata makro (Macro-Averaged).</div>
    """)
    
    k1, k2, k3, k4 = st.columns(4)
    kpi_list = [
        (k1, "Overall Accuracy", metrics['accuracy'] * 100, "Ketepatan klasifikasi keseluruhan", "#4f46e5"),
        (k2, "Macro Precision", metrics['precision'] * 100, "Presisi rata-rata antar kelas", "#0284c7"),
        (k3, "Macro Recall", metrics['recall'] * 100, "Sensitivitas tangkapan per kelas", "#059669"),
        (k4, "Macro F1-Score", metrics['f1_score'] * 100, "Harmonic mean Precision & Recall", "#e11d48")
    ]
    for col, title, val, sub, bar_c in kpi_list:
        with col:
            render_html(f"""
            <div class="kpi-bento">
                <div class="kpi-bento-title">{title}</div>
                <p class="kpi-bento-val">{val:.2f}%</p>
                <p class="kpi-bento-sub">{sub}</p>
                <div class="kpi-mini-track">
                    <div style="width: {val:.1f}%; height: 100%; background: {bar_c}; border-radius: 999px;"></div>
                </div>
            </div>
            """)

    render_html('<div style="margin-top: 24px;"></div>')
    c_cm, c_rep = st.columns([1, 1.25], gap="large")

    with c_cm:
        render_html("""
        <div class="panel-heading">Confusion Matrix</div>
        <div class="panel-sub">Pemetaan jumlah prediksi benar dan salah pada setiap kelas sentimen.</div>
        """)
        
        fig, ax = plt.subplots(figsize=(5.5, 4.3), facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        
        cm_mat = np.array(metrics['confusion_matrix'])
        lbls = [l.upper() for l in metrics['labels']]

        custom_cmap = LinearSegmentedColormap.from_list(
            "pearl_indigo", ["#f1f5f9", "#c7d2fe", "#6366f1", "#312e81"]
        )

        sns.heatmap(
            cm_mat,
            annot=True,
            fmt='d',
            cmap=custom_cmap,
            xticklabels=lbls,
            yticklabels=lbls,
            ax=ax,
            cbar=False,
            linewidths=3,
            linecolor='#ffffff',
            annot_kws={"size": 12, "weight": "bold", "family": "monospace"}
        )
        ax.set_xlabel('PREDIKSI MODEL', color='#475569', fontsize=9, fontweight='bold', labelpad=10)
        ax.set_ylabel('LABEL AKTUAL', color='#475569', fontsize=9, fontweight='bold', labelpad=10)
        ax.tick_params(colors='#0f172a', labelsize=9.5, length=0)
        plt.tight_layout()
        st.pyplot(fig)

    with c_rep:
        render_html("""
        <div class="panel-heading">Laporan Klasifikasi per Kelas (Classification Report)</div>
        <div class="panel-sub">Rincian nilai Precision, Recall, F1-Score, dan jumlah sampel uji (Support) tiap kelas.</div>
        """)
        
        r_dict = metrics['classification_report']
        badge_styles = {
            'kontra': ('tag-kontra', 'KONTRA'),
            'netral': ('tag-netral', 'NETRAL'),
            'pro': ('tag-pro', 'PRO')
        }

        rows_html = ""
        for cls_key in ['kontra', 'netral', 'pro']:
            if cls_key in r_dict:
                row_data = r_dict[cls_key]
                b_cls, b_txt = badge_styles[cls_key]
                p_v = row_data['precision'] * 100
                r_v = row_data['recall'] * 100
                f_v = row_data['f1-score'] * 100
                s_v = int(row_data['support'])
                rows_html += f"""
                <tr>
                    <td><span class="cls-tag {b_cls}">{b_txt}</span></td>
                    <td>{p_v:.2f}%</td>
                    <td>{r_v:.2f}%</td>
                    <td style="color: #4f46e5; font-weight: 800;">{f_v:.2f}%</td>
                    <td>{s_v:,} sampel</td>
                </tr>
                """

        render_html(f"""
        <div class="table-glass">
            <table class="modern-table">
                <thead>
                    <tr>
                        <th>Kategori Kelas</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Volume Uji</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """)

        render_html('<div class="panel-heading" style="margin-top: 8px; margin-bottom: 10px;">Agregasi Rata-Rata Evaluasi</div>')
        
        avg_rows_html = ""
        for avg_key, avg_label in [('macro avg', 'Macro Average (Bobot Setara)'), ('weighted avg', 'Weighted Average (Proporsional)')]:
            if avg_key in r_dict:
                row_data = r_dict[avg_key]
                p_v = row_data['precision'] * 100
                r_v = row_data['recall'] * 100
                f_v = row_data['f1-score'] * 100
                avg_rows_html += f"""
                <tr>
                    <td style="font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #0f172a;">{avg_label}</td>
                    <td>{p_v:.2f}%</td>
                    <td>{r_v:.2f}%</td>
                    <td style="color: #059669; font-weight: 800;">{f_v:.2f}%</td>
                </tr>
                """

        render_html(f"""
        <div class="table-glass">
            <table class="modern-table">
                <thead>
                    <tr>
                        <th>Skema Agregasi</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
                </thead>
                <tbody>
                    {avg_rows_html}
                </tbody>
            </table>
        </div>
        """)

# ------------------------------------------
# TAB 3: Analisis Multi-Kalimat (Batch)
# ------------------------------------------
with tab3:
    render_html("""
    <div class="panel-heading">Pengujian Multi-Kalimat Sekaligus (Batch Processing)</div>
    <div class="panel-sub">Masukkan beberapa kalimat (satu kalimat per baris) untuk menguji konsistensi model secara massal.</div>
    """)

    default_batch = "\n".join([
        "kapan mbg sampe ke aceh dan bagaimana jadwal pembagiannya?",
        "INILAH RAKYAT YAA PADAHAL MBG BISA BUAT LAPANGAN PEKERJAAN.",
        "mbg tu program yang nga guna mendingan kasi ke ortu siswa itu lebih bermanfaat",
        "berapa anggaran makan bergizi gratis per porsi untuk siswa sd?",
        "alhamdulillah anak-anak jadi rajin sekolah dan hemat uang jajan",
        "proyek abal-abal rawan korupsi cuma buang anggaran negara saja"
    ])

    batch_input = st.text_area(
        label="Batch Input",
        value=default_batch,
        height=170,
        label_visibility="collapsed"
    )

    if st.button("Proses Seluruh Kalimat", type="primary"):
        lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
        if not lines:
            st.warning("Masukkan minimal satu baris kalimat untuk dianalisis.")
        else:
            batch_rows_html = ""
            counts = {0: 0, 1: 0, 2: 0}
            badge_map = {
                0: ('tag-kontra', 'KONTRA'),
                1: ('tag-netral', 'NETRAL'),
                2: ('tag-pro', 'PRO')
            }

            for idx, line_txt in enumerate(lines, start=1):
                p_id, p_arr, _, _ = predict_multiclass_smart(line_txt, model, vectorizer)
                counts[p_id] += 1
                b_cls, b_name = badge_map[p_id]
                batch_rows_html += f"""
                <tr>
                    <td style="color: #64748b;">#{idx:02d}</td>
                    <td style="font-family: 'Plus Jakarta Sans', sans-serif; color: #0f172a; font-weight: 600;">{line_txt}</td>
                    <td><span class="cls-tag {b_cls}">{b_name}</span></td>
                    <td>{p_arr[0]*100:.1f}%</td>
                    <td>{p_arr[1]*100:.1f}%</td>
                    <td>{p_arr[2]*100:.1f}%</td>
                </tr>
                """

            s1, s2, s3 = st.columns(3)
            with s1:
                render_html(f'<div class="bento-mini" style="background:#fff1f2; border-color:#fecdd3;"><div class="bento-mini-label" style="color:#be123c !important;">Total Terdeteksi Kontra</div><p class="bento-mini-val" style="color:#e11d48 !important; font-size:1.3rem;">{counts[0]} Kalimat</p></div>')
            with s2:
                render_html(f'<div class="bento-mini" style="background:#f0f9ff; border-color:#bae6fd;"><div class="bento-mini-label" style="color:#0369a1 !important;">Total Terdeteksi Netral</div><p class="bento-mini-val" style="color:#0284c7 !important; font-size:1.3rem;">{counts[1]} Kalimat</p></div>')
            with s3:
                render_html(f'<div class="bento-mini" style="background:#ecfdf5; border-color:#a7f3d0;"><div class="bento-mini-label" style="color:#047857 !important;">Total Terdeteksi Pro</div><p class="bento-mini-val" style="color:#059669 !important; font-size:1.3rem;">{counts[2]} Kalimat</p></div>')

            render_html(f"""
            <div class="table-glass" style="margin-top: 12px;">
                <table class="modern-table">
                    <thead>
                        <tr>
                            <th>No</th>
                            <th>Kalimat Opini / Pertanyaan</th>
                            <th>Keputusan</th>
                            <th>Kontra</th>
                            <th>Netral</th>
                            <th>Pro</th>
                        </tr>
                    </thead>
                    <tbody>
                        {batch_rows_html}
                    </tbody>
                </table>
            </div>
            """)