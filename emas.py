# =====================================================================
# GoldVision AI — Sistem Peramalan Harga Emas Multivariat
# Arsitektur: Hybrid CNN-LSTM
# Fitur: USD Index, Minyak Mentah, S&P 500, Harga Emas
#
# [UI OVERHAUL TOTAL] Redesign premium oleh Lead UI/UX Engineer.
# Tema: Premium Light — Bloomberg / Yahoo Finance Premium aesthetic.
# Palet: Background #F7F8FC (abu-abu super terang), Navy #0B1F3A,
#         Gold #D4AF37, Emerald #22A06B (badge akurat), Merah #E53E3E.
#
# ⚠️  LOGIKA INTI YANG TIDAK DIUBAH SAMA SEKALI:
#     - fetch_live_data()      → yfinance tickers & windowing
#     - recursive_forecast()   → Random Walk + Dummy Matrix trick
#     - load_prediction_model() / load_scaler() → caching & loading
# =====================================================================

# --- Fix TensorFlow DLL issue (HARUS sebelum import tensorflow) ---
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# =====================================================================
# [UI] 1. KONFIGURASI HALAMAN — Tab title, icon, layout wide
# =====================================================================

st.set_page_config(
    page_title="GoldVision AI — Peramalan Harga Emas",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",   # [UI] Sidebar expanded secara default
)


# =====================================================================
# [UI] 2. INJEKSI CSS PREMIUM — TEMA LIGHT PROFESIONAL WALL STREET
#        Seluruh blok ini murni estetika; tidak ada logika bisnis.
#        Teknik: CSS custom properties (variables) + utility classes.
# =====================================================================

st.markdown("""
<style>
    /* ================================================================
       [UI] IMPORT FONT — Inter (body) + Playfair Display (display/header)
       Pair ini sengaja dipilih: Playfair memberi kesan jurnal finansial
       premium (FT, WSJ), sementara Inter menjaga keterbacaan data.
    ================================================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,600&display=swap');

    /* ================================================================
       [UI] ROOT TOKEN — Sistem warna & tipografi terpusat
    ================================================================ */
    :root {
        --bg-primary:    #0A0A0F;
        --bg-card:       #12121A;
        --bg-card-hover: #1A1A25;
        --navy:          #E8ECF1;
        --navy-muted:    #CBD5E0;
        --navy-light:    #A0AEC0;
        --navy-ghost:    #718096;
        --gold:          #D4AF37;
        --gold-bright:   #E5C548;
        --gold-pale:     rgba(212, 175, 55, 0.10);
        --gold-border:   rgba(212, 175, 55, 0.25);
        --green:         #38D989;
        --green-bg:      rgba(56, 217, 137, 0.12);
        --red:           #FF6B6B;
        --red-bg:        rgba(255, 107, 107, 0.12);
        --border:        rgba(255, 255, 255, 0.07);
        --border-strong: rgba(255, 255, 255, 0.12);
        --shadow-sm:     0 1px 4px rgba(0,0,0,0.30), 0 4px 16px rgba(0,0,0,0.20);
        --shadow-md:     0 4px 24px rgba(0,0,0,0.40), 0 1px 4px rgba(0,0,0,0.30);
        --shadow-gold:   0 6px 30px rgba(212,175,55,0.15);
        --radius-sm:     10px;
        --radius-md:     16px;
        --radius-lg:     22px;
        --font-display:  'Playfair Display', Georgia, serif;
        --font-body:     'Inter', system-ui, -apple-system, sans-serif;
        --font-mono:     'SF Mono', 'Fira Code', 'Consolas', monospace;
    }

    /* ================================================================
       [UI] RESET & BASE — Latar gelap, teks terang, font Inter
    ================================================================ */
    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: var(--navy-muted);
        background-color: var(--bg-primary);
    }

    /* [UI] Warna latar aplikasi: hitam premium (#0A0A0F) */
    .stApp {
        background-color: var(--bg-primary) !important;
        background-image: radial-gradient(ellipse at 50% 0%, rgba(212,175,55,0.03) 0%, transparent 60%);
    }

    /* [UI] Sembunyikan elemen bawaan Streamlit yang tidak estetis */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    /* header    { visibility: hidden; } */

    /* [UI] Header bar: transparan agar blend dengan dark bg */
    header[data-testid="stHeader"] {
        background: rgba(10,10,15,0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* [UI] Kurangi padding atas, batasi lebar maksimum konten */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* [UI] Streamlit label/caption text — terang di dark bg */
    .stMarkdown, .stMarkdown p, label, .stSlider label {
        color: var(--navy-muted) !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: var(--navy) !important;
    }

    /* ================================================================
       [UI] SIDEBAR PREMIUM — Hitam pekat dengan aksen emas
    ================================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08080D 0%, #0D0D15 100%) !important;
        border-right: 1px solid rgba(212,175,55,0.12) !important;
        box-shadow: 4px 0 32px rgba(0,0,0,0.40);
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider p {
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.87rem !important;
    }
    /* [UI] Slider thumb: warna emas di sidebar */
    [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        background-color: var(--gold-bright) !important;
        border-color: var(--gold-bright) !important;
        box-shadow: 0 0 0 4px rgba(212,175,55,0.20) !important;
    }
    [data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stTickBar"] {
        background: rgba(212,175,55,0.15) !important;
    }
    /* [UI] Heading dalam sidebar berwarna emas */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--gold-bright) !important;
    }
    /* [UI] Garis pemisah sidebar: putih 8% opacity */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1rem 0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.86rem !important;
        line-height: 1.7 !important;
    }

    /* ================================================================
       [UI] HERO SECTION — Card putih, accent line emas di atas
    ================================================================ */
    .hero-section {
        text-align: center;
        padding: 2.75rem 1.5rem 2.25rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    /* [UI] Garis dekoratif emas di atas hero — signature premium look */
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #D4AF37 40%, #FFD700 60%, transparent 100%);
    }
    /* [UI] Efek glare tipis di pojok kiri atas */
    .hero-section::after {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 280px; height: 280px;
        background: radial-gradient(circle at 0% 0%, rgba(212,175,55,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-family: var(--font-body);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.85rem;
    }
    .hero-title {
        font-family: var(--font-display);
        font-size: clamp(1.35rem, 2.8vw, 2.05rem);
        font-weight: 700;
        color: var(--navy);
        line-height: 1.35;
        margin-bottom: 0.9rem;
        max-width: 840px;
        margin-left: auto;
        margin-right: auto;
        text-align: center !important;
    }
    .hero-subtitle {
        font-size: 0.93rem;
        color: var(--navy-light);
        line-height: 1.7;
        max-width: 700px;
        margin: 0 auto 0.6rem;
        font-weight: 400;
        text-align: center !important;
    }
    .hero-badges {
        margin-top: 1.35rem;
        display: flex;
        justify-content: center;
        gap: 0.45rem;
        flex-wrap: wrap;
    }
    /* [UI] Pill badge dengan background emas gelap, border emas tipis */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(212,175,55,0.10);
        border: 1px solid rgba(212,175,55,0.22);
        color: var(--gold-bright);
        padding: 0.30rem 0.90rem;
        border-radius: 20px;
        font-size: 0.77rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    /* ================================================================
       [UI] LIVE INDICATOR — Titik berkedip merah (Live Market)
    ================================================================ */
    @keyframes pulse-live {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.85); }
    }
    .live-dot {
        display: inline-block;
        width: 7px; height: 7px;
        background: #22C55E;
        border-radius: 50%;
        margin-right: 5px;
        animation: pulse-live 1.6s ease-in-out infinite;
        vertical-align: middle;
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.25);
        color: #4ADE80;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 0.22rem 0.75rem;
        border-radius: 20px;
    }

    /* ================================================================
       [UI] NAVIGASI TABS — Garis bawah emas, clean tanpa box
    ================================================================ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 2px solid rgba(255,255,255,0.06) !important;
        gap: 0;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--navy-ghost) !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 0.8rem 1.4rem !important;
        font-family: var(--font-body) !important;
        font-weight: 500 !important;
        font-size: 0.89rem !important;
        margin-bottom: -2px;
        transition: color 0.18s ease, border-color 0.18s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--navy) !important;
        border-bottom-color: rgba(212,175,55,0.45) !important;
    }
    /* [UI] Tab aktif: teks terang bold + garis bawah emas penuh */
    .stTabs [aria-selected="true"] {
        color: var(--gold-bright) !important;
        font-weight: 700 !important;
        border-bottom: 2px solid var(--gold-bright) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.75rem !important;
    }

    /* ================================================================
       [UI] CARD PREMIUM — Putih, shadow tipis, radius elegan
    ================================================================ */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.2s ease;
    }
    /* [UI] Card dengan aksen garis kiri emas — untuk section highlight */
    .card-accent {
        border-left: 3px solid var(--gold-bright) !important;
        border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
    }
    .card h3 {
        font-family: var(--font-display);
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 0.85rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card p, .card li {
        color: var(--navy-light);
        font-size: 0.92rem;
        line-height: 1.75;
    }
    .card ul { padding-left: 1.3rem; }

    /* ================================================================
       [UI] METRIC CARDS — Grid 3 kolom, hover lift effect
    ================================================================ */
    .metric-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.6rem 1.25rem 1.35rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
        position: relative;
        overflow: hidden;
    }
    /* [UI] Garis dekoratif gradient emas di bawah setiap metric card */
    .metric-box::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--gold-bright) 0%, rgba(212,175,55,0.15) 100%);
    }
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
    }
    .metric-box .m-label {
        font-size: 0.71rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--navy-ghost);
        margin-bottom: 0.55rem;
    }
    /* [UI] Angka besar: Playfair Display berat, navy gelap */
    .metric-box .m-value {
        font-family: var(--font-display);
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1.05;
        margin-bottom: 0.5rem;
    }
    /* [UI] Varian warna emas untuk angka utama prediksi */
    .metric-box .m-value-gold { color: var(--gold); }
    .metric-box .m-sub {
        font-size: 0.79rem;
        color: var(--navy-ghost);
        font-weight: 500;
        margin-top: 0.25rem;
    }
    /* [UI] Badge hijau "Sangat Akurat" */
    .metric-box .m-badge-green {
        display: inline-block;
        background: var(--green-bg);
        color: var(--green);
        border: 1px solid rgba(34,160,107,0.22);
        border-radius: 20px;
        padding: 0.22rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .metric-box .m-badge-up {
        display: inline-block;
        background: var(--green-bg);
        color: var(--green);
        border: 1px solid rgba(34,160,107,0.22);
        border-radius: 20px;
        padding: 0.22rem 0.75rem;
        font-size: 0.77rem;
        font-weight: 700;
    }
    .metric-box .m-badge-down {
        display: inline-block;
        background: var(--red-bg);
        color: var(--red);
        border: 1px solid rgba(229,62,62,0.20);
        border-radius: 20px;
        padding: 0.22rem 0.75rem;
        font-size: 0.77rem;
        font-weight: 700;
    }

    /* ================================================================
       [UI] TOMBOL PREDIKSI — Gold gradient premium, hover shine
    ================================================================ */
    .stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #B8941E 100%) !important;
        color: #0A0A0F !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.85rem 2.5rem !important;
        font-family: var(--font-body) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.6px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 16px rgba(201,162,39,0.25), 0 0 40px rgba(212,175,55,0.08) !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #E5C03D 0%, #D4AF37 100%) !important;
        box-shadow: 0 6px 24px rgba(201,162,39,0.40), 0 0 60px rgba(212,175,55,0.12) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 8px rgba(201,162,39,0.25) !important;
    }

    /* ================================================================
       [UI] HARGA LIVE BESAR — Hero price block di atas grafik
    ================================================================ */
    .live-price-block {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1.5rem 2rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    /* [UI] Accent line emas di atas harga live */
    .live-price-block::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--gold-bright) 0%, rgba(212,175,55,0.15) 60%, transparent 100%);
    }
    /* [UI] Efek shimmer di pojok kanan */
    .live-price-block::after {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 150px; height: 150px;
        background: radial-gradient(circle, rgba(212,175,55,0.07) 0%, transparent 70%);
        pointer-events: none;
    }
    .live-price-label {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.3rem;
    }
    /* [UI] Harga besar: Playfair 2.8rem, navy gelap */
    .live-price-number {
        font-family: var(--font-display);
        font-size: 2.9rem;
        font-weight: 700;
        color: var(--navy);
        line-height: 1;
        letter-spacing: -0.5px;
    }
    .live-price-ticker {
        font-family: var(--font-mono);
        font-size: 0.80rem;
        color: var(--navy-ghost);
        margin-left: 0.4rem;
        vertical-align: middle;
    }
    .live-price-meta {
        font-size: 0.76rem;
        color: var(--navy-ghost);
        margin-top: 0.35rem;
    }

    /* ================================================================
       [UI] DIVIDERS — Tipis & gold gradient
    ================================================================ */
    .gold-rule {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, var(--gold-bright) 50%, transparent 100%);
        margin: 2rem 0;
        opacity: 0.45;
    }
    .section-divider {
        height: 1px;
        background: var(--border);
        margin: 1.75rem 0;
    }

    /* ================================================================
       [UI] SECTION LABELS — Eyebrow + Title + Subtitle pattern
    ================================================================ */
    .section-eyebrow {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.25rem;
    }
    .section-title {
        font-family: var(--font-display);
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 0.4rem;
    }
    .section-sub {
        font-size: 0.88rem;
        color: var(--navy-light);
        margin-bottom: 1.25rem;
        line-height: 1.65;
    }

    /* ================================================================
       [UI] ARSITEKTUR MODEL — Layer blocks dengan border kiri emas
    ================================================================ */
    .arch-block {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-left: 3px solid var(--gold-bright);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 1rem 1.5rem;
        margin: 0.65rem 0;
        transition: background 0.15s ease;
    }
    .arch-block:hover { background: rgba(255,255,255,0.05); }
    .arch-block strong { color: var(--navy); font-size: 0.92rem; }
    .arch-block code {
        background: rgba(212,175,55,0.12);
        color: var(--gold-bright);
        padding: 0.12rem 0.45rem;
        border-radius: 5px;
        font-size: 0.83rem;
        font-family: var(--font-mono);
    }
    .arch-block p {
        color: var(--navy-light);
        font-size: 0.89rem;
        margin: 0.35rem 0 0;
        line-height: 1.65;
    }

    /* ================================================================
       [UI] TECH CHIPS — Pill badge navy untuk tech stack
    ================================================================ */
    .tech-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(212,175,55,0.12);
        border: 1px solid rgba(212,175,55,0.20);
        color: var(--gold-bright) !important;
        padding: 0.34rem 0.90rem;
        border-radius: 20px;
        font-size: 0.80rem;
        font-weight: 500;
        margin: 0.28rem;
        transition: all 0.15s ease;
    }
    .tech-chip:hover { background: rgba(212,175,55,0.20); border-color: rgba(212,175,55,0.35); }

    /* ================================================================
       [UI] VARIABEL CARDS — 3 kolom korelasi dengan badge warna
    ================================================================ */
    .var-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        height: 100%;
        box-shadow: var(--shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .var-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
    }
    .var-card .var-icon { font-size: 2rem; margin-bottom: 0.65rem; }
    .var-card .var-title {
        font-family: var(--font-display);
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 0.55rem;
    }
    .var-card .var-corr {
        display: inline-block;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.22rem 0.65rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
    }
    .corr-neg { background: var(--red-bg);   color: var(--red);   border: 1px solid rgba(255,107,107,0.22); }
    .corr-pos { background: var(--green-bg); color: var(--green); border: 1px solid rgba(56,217,137,0.22); }
    .corr-inv { background: rgba(99,102,241,0.12); color: #818CF8; border: 1px solid rgba(129,140,248,0.22); }
    .var-card p {
        font-size: 0.86rem;
        color: var(--navy-light);
        line-height: 1.72;
        margin: 0;
    }

    /* ================================================================
       [UI] DATAFRAME — Header navy, stripe abu-abu, rounded border
    ================================================================ */
    .stDataFrame {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stDataFrame thead th {
        background: rgba(212,175,55,0.15) !important;
        color: var(--gold-bright) !important;
        font-weight: 600 !important;
        font-size: 0.81rem !important;
        letter-spacing: 0.5px !important;
    }
    .stDataFrame tbody tr:nth-child(even) { background: rgba(255,255,255,0.02) !important; }
    .stDataFrame tbody td {
        font-size: 0.87rem !important;
        color: var(--navy-muted) !important;
    }

    /* ================================================================
       [UI] EXPANDER — Accordion ringan & bersih
    ================================================================ */
    .stExpander {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        background: var(--bg-card) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    .stExpander summary {
        font-weight: 600 !important;
        color: var(--gold-bright) !important;
        font-size: 0.90rem !important;
    }
    .stExpander [data-testid="stExpanderDetails"] {
        background: var(--bg-card) !important;
    }

    /* ================================================================
       [UI] SPINNER — Warna emas untuk loading spinner
    ================================================================ */
    .stSpinner > div {
        border-top-color: var(--gold-bright) !important;
    }

    /* ================================================================
       [UI] TOAST / ALERT — Rounded, warna sesuai tema
    ================================================================ */
    .stAlert {
        border-radius: var(--radius-sm) !important;
        border-left-width: 3px !important;
    }

    /* ================================================================
       [UI] FORECAST TABLE HIGHLIGHT — Baris prediksi dengan teks emas
    ================================================================ */
    .forecast-row-gold {
        color: var(--gold) !important;
        font-weight: 600 !important;
    }

    /* ================================================================
       [UI] FOOTER PREMIUM — Minimalis, centered, navy/gold
    ================================================================ */
    .footer-premium {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        margin-top: 3.5rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .footer-premium .footer-brand {
        font-family: var(--font-display);
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--gold-bright);
        margin-bottom: 0.4rem;
    }
    .footer-premium .footer-sub {
        font-size: 0.80rem;
        color: var(--navy-ghost);
        line-height: 1.7;
    }
    .footer-premium .footer-dot {
        display: inline-block;
        width: 4px; height: 4px;
        background: var(--gold-bright);
        border-radius: 50%;
        margin: 0 0.55rem;
        vertical-align: middle;
    }

    /* ================================================================
       [UI] SIDEBAR PARAM ROW — Layout key-value di dalam sidebar
    ================================================================ */
    .sidebar-param-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .sidebar-param-key {
        font-size: 0.79rem;
        color: rgba(255,255,255,0.42);
        font-weight: 400;
        flex: 1;
    }
    .sidebar-param-val {
        font-size: 0.79rem;
        color: rgba(255,255,255,0.90);
        font-weight: 600;
        text-align: right;
        flex: 1;
    }

    /* ================================================================
       [UI] RECURSIVE HIGHLIGHT BOX — Penjelasan dummy matrix trick
    ================================================================ */
    .recursive-box {
        margin-top: 1.25rem;
        padding: 1rem 1.4rem;
        background: var(--gold-pale);
        border-radius: var(--radius-sm);
        border: 1px solid var(--gold-border);
    }
    .recursive-box strong { color: var(--gold); font-size: 0.92rem; }
    .recursive-box p {
        color: var(--navy-light);
        font-size: 0.87rem;
        margin: 0.5rem 0 0;
        line-height: 1.68;
    }

</style>
""", unsafe_allow_html=True)


# =====================================================================
# 3. FUNGSI HELPER: LOAD MODEL & SCALER (CACHED)
# [TIDAK DIUBAH — logika caching dan loading model asli dipertahankan]
# =====================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, LSTM, Dropout, Dense

@st.cache_resource(show_spinner="Memuat model CNN-LSTM...")
def load_prediction_model():
    """Memuat model Keras CNN-LSTM dari file .keras (cached agar tidak reload)."""
    model = Sequential([
        InputLayer(shape=(10, 4)),
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        LSTM(units=50, activation='relu'),
        Dropout(rate=0.2),
        Dense(units=1, activation='linear')
    ])
    model.load_weights("model_cnn_lstm_emas.keras")
    return model


@st.cache_resource(show_spinner="Memuat MinMaxScaler...")
def load_scaler():
    """Memuat MinMaxScaler yang sudah di-fit saat training (cached)."""
    scaler = joblib.load("scaler_emas.sav")
    return scaler


# =====================================================================
# 4. FUNGSI: AMBIL DATA LIVE DARI YAHOO FINANCE
# [TIDAK DIUBAH — logika yfinance asli dipertahankan sepenuhnya]
# =====================================================================

@st.cache_data(ttl=3600, show_spinner="Mengambil data live dari Yahoo Finance...")
def fetch_live_data():
    """
    Mengambil data bursa 10 hari terakhir menggunakan yfinance.
    Ticker:
      - DX-Y.NYB : USD Index
      - CL=F     : Minyak Mentah (Crude Oil)
      - ^GSPC    : S&P 500
      - GC=F     : Emas (Gold Futures)
    Returns: DataFrame dengan kolom [USD, Oil, SP500, Gold]
    """
    tickers = {
        "USD": "DX-Y.NYB",
        "Oil": "CL=F",
        "SP500": "^GSPC",
        "Gold": "GC=F",
    }

    # Ambil data 30 hari kalender untuk memastikan dapat 10 hari bursa
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    frames = {}
    for col_name, ticker in tickers.items():
        data = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
        if data.empty:
            st.error(f"❌ Gagal mengambil data untuk {col_name} ({ticker}).")
            return None
        # Gunakan kolom 'Close'; flatten MultiIndex jika ada
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        frames[col_name] = data["Close"]

    # Gabungkan semua ke satu DataFrame, urutan kolom: USD, Oil, SP500, Gold
    df = pd.DataFrame(frames)
    df.dropna(inplace=True)

    # Ambil 10 hari bursa terakhir
    df = df.tail(10).reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)

    return df


# =====================================================================
# 5. FUNGSI: RECURSIVE MULTI-STEP FORECASTING
# [TIDAK DIUBAH — logika Random Walk & Dummy Matrix asli dipertahankan]
# =====================================================================

def recursive_forecast(model, scaler, last_window, n_days):
    """
    Melakukan peramalan rekursif (Random Walk Hypothesis).

    Parameters
    ----------
    model      : Model Keras CNN-LSTM yang sudah terlatih.
    scaler     : MinMaxScaler yang sudah di-fit (4 fitur).
    last_window: numpy array shape (10, 4) — data 10 hari terakhir
                 yang sudah dinormalisasi.
    n_days     : Jumlah hari yang akan diramalkan (1–7).

    Returns
    -------
    predictions: List berisi hasil prediksi harga emas (USD) per hari.

    Logika:
    - Untuk setiap langkah, nilai USD, Oil, SP500 dibekukan (frozen)
      dari hari terakhir yang tersedia.
    - Hasil prediksi Emas dari model digunakan sebagai input Gold
      untuk langkah berikutnya.
    - Sliding window bergeser ke depan 1 langkah setiap iterasi.
    - Inverse transform menggunakan dummy matrix (n, 4) agar scaler
      tidak error karena jumlah fitur berbeda.
    """
    predictions = []
    current_window = last_window.copy()  # shape: (10, 4)

    for _ in range(n_days):
        # Reshape untuk input model: (1, 10, 4)
        input_data = current_window.reshape(1, 10, 4)

        # Prediksi nilai emas (normalized)
        pred_scaled = model.predict(input_data, verbose=0)[0, 0]

        # --- Inverse Transform dengan Dummy Matrix ---
        # Buat matriks dummy berukuran (1, 4), isi kolom Gold (index 3)
        # dengan hasil prediksi, kolom lain diisi 0
        dummy = np.zeros((1, 4))
        dummy[0, 3] = pred_scaled
        pred_actual = scaler.inverse_transform(dummy)[0, 3]
        predictions.append(pred_actual)

        # --- Geser Sliding Window ---
        # Bekukan fitur USD (idx 0), Oil (idx 1), SP500 (idx 2) dari baris terakhir
        frozen_features = current_window[-1, :3]  # [USD, Oil, SP500] normalized
        new_row = np.append(frozen_features, pred_scaled)  # tambah prediksi Gold
        # Geser window: buang baris pertama, tambah baris baru di akhir
        current_window = np.vstack([current_window[1:], new_row.reshape(1, -1)])

    return predictions


# =====================================================================
# [UI] 6. SIDEBAR PREMIUM — Logo emas, slider horizon, profil AI
# =====================================================================

with st.sidebar:

    # [UI] Logo + branding GoldVision di atas sidebar
    st.markdown("""
    <div style="text-align:center; padding: 1.75rem 0 1rem;">
        <div style="margin-bottom: 0.5rem; filter: drop-shadow(0 2px 8px rgba(212,175,55,0.35));">
            <img src="https://img.icons8.com/fluency/96/gold-bars.png" width="64" alt="Gold Icon">
        </div>
        <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                    font-weight:700; color:#D4AF37 !important; margin-top:0.6rem;
                    letter-spacing:0.3px;">ZaGold AI</div>
        <div style="font-size:0.68rem; color:#D4AF37 !important;
                    letter-spacing:2.5px; text-transform:uppercase;
                    margin-top:0.3rem;">Hybrid CNN-LSTM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # [UI] Label section parameter peramalan
    st.markdown("""
    <div style="font-size:0.68rem; font-weight:700; letter-spacing:2px;
                text-transform:uppercase; color:rgba(212,175,55,0.75);
                margin-bottom:0.6rem;">⚙️ &nbsp;Parameter Peramalan</div>
    """, unsafe_allow_html=True)

    # [UI] Slider dipindahkan ke dalam sidebar sesuai spesifikasi
    n_days = st.slider(
        "Horizon Peramalan (Hari)",
        min_value=1,
        max_value=7,
        value=3,
        step=1,
        help="Jumlah hari bursa ke depan yang akan diramalkan (1–7 hari).",
    )

    st.markdown("---")

    # [UI] Profil ringkas parameter model AI
    st.markdown("""
    <div style="font-size:0.68rem; font-weight:700; letter-spacing:2px;
                text-transform:uppercase; color:rgba(212,175,55,0.75);
                margin-bottom:0.75rem;">🧠 &nbsp;Profil Model AI</div>
    """, unsafe_allow_html=True)

    ai_params = {
        "Arsitektur":    "CNN-LSTM Hybrid",
        "Input Window":  "10 hari bursa",
        "Fitur Input":   "4 variabel",
        "Conv1D Filters":"64 (kernel=2)",
        "LSTM Units":    "50 unit",
        "Dropout Rate":  "20%",
        "Output":        "Prediksi 1 hari",
        "Metode":        "Recursive Forecast",
    }
    for k, v in ai_params.items():
        st.markdown(f"""
        <div class="sidebar-param-row">
            <span class="sidebar-param-key">{k}</span>
            <span class="sidebar-param-val">{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # [UI] Daftar variabel input dengan ticker kode
    st.markdown("""
    <div style="font-size:0.68rem; font-weight:700; letter-spacing:2px;
                text-transform:uppercase; color:rgba(212,175,55,0.75);
                margin-bottom:0.75rem;">📊 &nbsp;Variabel Input</div>
    """, unsafe_allow_html=True)

    for icon, label, ticker in [
        ("💵", "USD Index",     "DX-Y.NYB"),
        ("🛢️",  "Minyak Mentah", "CL=F"),
        ("📈", "S&P 500",       "^GSPC"),
        ("🥇", "Gold Futures",  "GC=F"),
    ]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.55rem;">
            <span style="font-size:1.1rem; flex-shrink:0;">{icon}</span>
            <div>
                <div style="font-size:0.83rem; color:rgba(255,255,255,0.88);
                             font-weight:500; line-height:1.2;">{label}</div>
                <div style="font-size:0.70rem; color:rgba(255,255,255,0.32);
                             font-family:'SF Mono',monospace; margin-top:0.1rem;">{ticker}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # [UI] Keterangan refresh data di bagian bawah sidebar
    st.markdown("""
    <div style="font-size:0.70rem; color:rgba(255,255,255,0.28);
                text-align:center; line-height:1.7;">
        <span style="display:inline-block; width:7px; height:7px; background:#22C55E;
                     border-radius:50%; margin-right:4px; vertical-align:middle;"></span>
        Data diperbarui tiap 60 menit<br>via Yahoo Finance API
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# [UI] 7. HERO SECTION — Judul akademis, badge, subtitle ringkas
#         Judul PERSIS sesuai spesifikasi, tanpa tambahan kata apapun.
# =====================================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">Zagold AI &nbsp;·&nbsp; HYBRID CNN-LSTM</div>
    <div class="hero-title">
        Peramalan Harga Emas Menggunakan Arsitektur Hybrid CNN-LSTM<br>
        Berbasis Data Time Series Multivariat
    </div>
    <div class="hero-subtitle">
        Sistem peramalan cerdas berbasis Deep Learning yang menggabungkan
        Convolutional Neural Network &amp; Long Short-Term Memory
        untuk analisis multivariat pasar komoditas emas global.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">🧠 Deep Learning</span>
        <span class="hero-badge">📊 Time Series Multivariat</span>
        <span class="hero-badge">🔄 Recursive Forecasting</span>
        <span class="hero-badge">🌐 Live Market Data</span>
        <span class="hero-badge">✅ MAPE 1.61%</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# [UI] 8. NAVIGASI 4 TABS — Label premium, ikon konsisten
# =====================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Prediksi Live",
    "📖  Tentang Sistem",
    "⚙️  Teknologi & Arsitektur",
    "📊  Rapor Evaluasi Model",
])


# =====================================================================
# [UI] TAB 1: PREDIKSI LIVE MARKET
# =====================================================================

with tab1:

    # [UI] Panel kontrol — card putih dengan instruksi ringkas
    st.markdown("""
    <div class="card">
        <p class="section-eyebrow">Live Market Intelligence</p>
        <p class="section-title" style="font-family:'Playfair Display',serif;
           font-size:1.2rem; font-weight:700; color:#E8ECF1; margin-bottom:0.4rem;">
           Panel Kontrol Prediksi
        </p>
        <p class="section-sub" style="margin-bottom:0;">
            Atur <strong>horizon peramalan (1–7 hari)</strong> melalui slider di
            <strong>Sidebar ←</strong>. Tekan tombol di bawah untuk mengeksekusi
            model Hybrid CNN-LSTM dengan data pasar terkini.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # [UI] Tombol utama gold gradient — full width
    predict_clicked = st.button("🚀  Jalankan Prediksi Sekarang", use_container_width=True)

    if predict_clicked:

        # -- Load model & scaler [logika asli, tidak diubah] --
        model  = load_prediction_model()
        scaler = load_scaler()

        # -- Ambil data live [logika asli, tidak diubah] --
        df_live = fetch_live_data()

        if df_live is not None and len(df_live) >= 10:

            # ── [UI] HIGHLIGHT HARGA PENUTUPAN TERKINI ──────────────────
            last_gold       = df_live["Gold"].iloc[-1]
            last_date_label = pd.to_datetime(df_live["Date"].iloc[-1]).strftime("%d %b %Y")
            refresh_ts      = datetime.now().strftime("%H:%M WIB, %d %b %Y")

            st.markdown(f"""
            <div class="live-price-block">
                <div style="flex:1;">
                    <div class="live-price-label">
                        🥇 Harga Emas Penutupan Terkini
                    </div>
                    <div>
                        <span class="live-price-number">${last_gold:,.2f}</span>
                        <span class="live-price-ticker">USD / troy oz</span>
                    </div>
                    <div class="live-price-meta">
                        GC=F &nbsp;·&nbsp; Data per {last_date_label} &nbsp;·&nbsp; Yahoo Finance
                    </div>
                </div>
                <div style="text-align:right; flex-shrink:0;">
                    <div style="margin-bottom:0.6rem;">
                        <span class="live-badge">
                            <span class="live-dot"></span>LIVE
                        </span>
                    </div>
                    <div style="font-size:0.68rem; text-transform:uppercase;
                                letter-spacing:2px; color:#E5C548; font-weight:700;
                                margin-bottom:0.2rem;">Horizon</div>
                    <div style="font-family:'Playfair Display',serif;
                                font-size:2.3rem; font-weight:700; color:#E8ECF1;
                                line-height:1;">
                        {n_days}
                        <span style="font-size:0.9rem; color:#718096; font-family:'Inter',sans-serif;">
                            hari
                        </span>
                    </div>
                    <div style="font-size:0.70rem; color:#718096; margin-top:0.3rem;">
                        Diperbarui {refresh_ts}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Normalisasi Data [logika asli, tidak diubah] ────────────
            feature_values = df_live[["USD", "Oil", "SP500", "Gold"]].values  # (10, 4)
            normalized     = scaler.transform(feature_values)                 # (10, 4)
            last_window    = normalized

            # ── Recursive Forecasting [logika asli, tidak diubah] ───────
            with st.spinner("⏳  Model CNN-LSTM sedang menjalankan recursive forecasting..."):
                predictions = recursive_forecast(model, scaler, last_window, n_days)

            # ── Buat DataFrame hasil prediksi ────────────────────────────
            last_date      = pd.to_datetime(df_live["Date"].iloc[-1])
            forecast_dates = pd.bdate_range(
                start=last_date + timedelta(days=1), periods=n_days
            )
            df_forecast = pd.DataFrame({
                "Tanggal":                  forecast_dates.strftime("%d %b %Y"),
                "Hari Ke-":                 [f"H+{i+1}" for i in range(n_days)],
                "Prediksi Harga Emas (USD)":[f"${p:,.2f}" for p in predictions],
            })

            # ── [UI] 3 METRIC CARDS: Aktual | Prediksi H+N | Selisih ───
            last_pred  = predictions[-1]
            delta      = last_pred - last_gold
            delta_pct  = (delta / last_gold) * 100
            is_up      = delta >= 0
            trend_icon = "▲" if is_up else "▼"
            badge_cls  = "m-badge-up" if is_up else "m-badge-down"

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="m-label">Harga Penutupan Aktual</div>
                    <div class="m-value">${last_gold:,.2f}</div>
                    <div class="m-sub">USD &nbsp;·&nbsp; Per {last_date_label}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="m-label">Prediksi H+{n_days}</div>
                    <div class="m-value m-value-gold">${last_pred:,.2f}</div>
                    <span class="{badge_cls}">{trend_icon} {delta_pct:+.2f}%</span>
                    <div class="m-sub">USD &nbsp;·&nbsp; Proyeksi CNN-LSTM</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="m-label">Selisih Proyeksi</div>
                    <div class="m-value">{trend_icon} ${abs(delta):,.2f}</div>
                    <span class="{badge_cls}">{'Naik' if is_up else 'Turun'} &nbsp;USD</span>
                    <div class="m-sub">Dari harga penutupan terakhir</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1.25rem;'></div>", unsafe_allow_html=True)

            # ── [UI] TABEL HASIL PREDIKSI ────────────────────────────────
            st.markdown("""
            <p class="section-eyebrow">Output Model</p>
            <p class="section-title" style="font-family:'Playfair Display',serif;
               font-size:1.1rem; font-weight:700; color:#E8ECF1; margin-bottom:0.5rem;">
               Tabel Hasil Prediksi
            </p>
            """, unsafe_allow_html=True)

            st.dataframe(df_forecast, use_container_width=True, hide_index=True)

            st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

            # ── [UI] GRAFIK PROYEKSI PREMIUM ─────────────────────────────
            st.markdown("""
            <p class="section-eyebrow">Visualisasi Proyeksi</p>
            <p class="section-title" style="font-family:'Playfair Display',serif;
               font-size:1.15rem; font-weight:700; color:#E8ECF1; margin-bottom:0.25rem;">
               Grafik Proyeksi Harga Emas
            </p>
            <p class="section-sub">
                Garis abu-abu &nbsp;<strong>—●</strong>&nbsp; = data aktual 10 hari bursa terakhir. &nbsp;
                Garis berwarna &nbsp;<strong>- - ◆</strong>&nbsp; = hasil prediksi model CNN-LSTM (Hijau jika Naik, Merah jika Turun).
            </p>
            """, unsafe_allow_html=True)

            # [UI] Matplotlib: latar gelap, spine bersih, grid halus
            fig, ax = plt.subplots(figsize=(13, 4.8))
            fig.patch.set_facecolor('#12121A')  # [UI] Latar figur gelap
            ax.set_facecolor("#12121A")          # [UI] Latar plot area gelap

            hist_dates_dt = pd.to_datetime(df_live["Date"]).dt.to_pydatetime()
            hist_prices   = df_live["Gold"].values

            # [UI] Plot historis — garis terang, marker putih
            ax.plot(
                hist_dates_dt, hist_prices,
                color="#A0AEC0", linewidth=2.0,
                marker="o", markersize=5.5,
                markerfacecolor="#12121A", markeredgecolor="#A0AEC0",
                markeredgewidth=1.5,
                label="Harga Aktual",
                zorder=3,
            )

            # Hubungkan garis historis terakhir ke garis prediksi
            forecast_dates_dt = forecast_dates.to_pydatetime()
            all_pred_dates  = np.concatenate([[hist_dates_dt[-1]], forecast_dates_dt])
            all_pred_values = np.concatenate([[hist_prices[-1]], predictions])

            # Tentukan warna tren prediksi (Naik = Hijau, Turun = Merah)
            is_up_trend = predictions[-1] >= hist_prices[-1]
            pred_color = "#4ADE80" if is_up_trend else "#FF6B6B"
            trend_label = "Naik" if is_up_trend else "Turun"

            # [UI] Garis prediksi — putus-putus tebal warna sesuai tren
            ax.plot(
                all_pred_dates, all_pred_values,
                color=pred_color, linewidth=2.8,
                marker="D", markersize=7,
                markerfacecolor=pred_color, markeredgecolor="#12121A",
                markeredgewidth=1.5,
                label=f"Prediksi ({trend_label})",
                linestyle="--",
                zorder=4,
            )

            # [UI] Area fill prediksi
            ax.fill_between(
                all_pred_dates, all_pred_values,
                alpha=0.15, color=pred_color, zorder=2,
            )

            # [UI] Garis pemisah vertikal antara aktual & prediksi
            ax.axvline(
                hist_dates_dt[-1],
                color="#A0AEC0", linestyle=":", alpha=0.55, linewidth=1.3,
            )

            # [UI] Auto-zoom Y Axis agar pergerakan terlihat jelas (Tidak mulai dari 0)
            all_prices = np.concatenate([hist_prices, predictions])
            min_price = min(all_prices)
            max_price = max(all_prices)
            padding = (max_price - min_price) * 0.15
            if padding == 0: padding = max_price * 0.01
            ax.set_ylim(min_price - padding, max_price + padding * 1.5)

            # [UI] Anotasi "PREDIKSI →" di atas garis pemisah
            ymax = ax.get_ylim()[1]
            ax.text(
                hist_dates_dt[-1], ymax,
                "  PREDIKSI →",
                fontsize=8, color=pred_color, fontweight="bold",
                va="top", ha="left", style="italic",
            )

            # [UI] Anotasi Harga pada titik prediksi (tanggal dihapus agar rapi)
            for d, p in zip(forecast_dates_dt, predictions):
                ax.annotate(
                    f"${p:,.2f}", 
                    (d, p),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center', va='bottom',
                    color=pred_color, fontsize=8.5, fontweight="bold",
                    bbox=dict(facecolor='#12121A', edgecolor=pred_color, alpha=0.8, boxstyle='round,pad=0.3')
                )

            # [UI] Hilangkan spine atas & kanan — look profesional minimal
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color((1, 1, 1, 0.08))
            ax.spines["bottom"].set_color((1, 1, 1, 0.08))

            # [UI] Grid putus-putus tipis
            ax.grid(axis="y", color=(1, 1, 1, 0.06), linewidth=0.6, linestyle="--")
            ax.grid(axis="x", color=(1, 1, 1, 0.04), linewidth=0.4, linestyle=":")

            # [UI] Label sumbu — warna terang, font kecil
            ax.set_xlabel("Tanggal", color="#A0AEC0", fontsize=9.5, fontweight="500")
            ax.set_ylabel("Harga Emas (USD)", color="#A0AEC0", fontsize=9.5, fontweight="500")
            ax.tick_params(colors="#718096", labelsize=8.5)
            
            # [UI] X-Ticks eksplisit agar semua tanggal terlihat
            all_dates_list = list(hist_dates_dt) + list(forecast_dates_dt)
            ax.set_xticks(all_dates_list)
            ax.set_xticklabels([d.strftime("%d %b") for d in all_dates_list], rotation=45, ha="right")

            # [UI] Legend premium — bingkai tipis, background gelap
            ax.legend(
                loc="upper left", fontsize=9, frameon=True,
                facecolor="#1A1A25", edgecolor=(1, 1, 1, 0.10),
                labelcolor="#CBD5E0", framealpha=0.95,
            )

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # ── [UI] RAW DATA disembunyikan dalam expander ───────────────
            # [UI] Fokus utama halaman tetap pada grafik; data historis
            #      tersedia tapi tidak memenuhi layar — accordion clean.
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            with st.expander("🗂️  Lihat Raw Data Pasar — 10 Hari Bursa Terakhir"):
                df_display = df_live.copy()
                df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime("%d %b %Y")
                df_display.columns = ["Tanggal", "USD Index", "Minyak Mentah (USD)", "S&P 500", "Emas (USD)"]

                st.dataframe(
                    df_display.style.format({
                        "USD Index":           "{:.2f}",
                        "Minyak Mentah (USD)": "${:.2f}",
                        "S&P 500":             "{:,.2f}",
                        "Emas (USD)":          "${:,.2f}",
                    }).set_table_styles([
                        {"selector": "thead th",
                         "props": [
                             ("background-color", "rgba(212,175,55,0.15)"),
                             ("color",            "#E5C548"),
                             ("font-weight",      "600"),
                             ("font-size",        "0.82rem"),
                         ]},
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

        else:
            st.error(
                "⚠️ Data tidak mencukupi. Pastikan koneksi internet aktif "
                "dan bursa sedang buka (hari kerja)."
            )


# =====================================================================
# [UI] TAB 2: TENTANG SISTEM — Card narasi + var cards korelasi
# =====================================================================

with tab2:

    # [UI] Card utama dengan kutipan judul penelitian bergaya italic emas
    st.markdown("""
    <div class="card card-accent">
        <h3>📖 Tentang Sistem</h3>
        <p style="color:#C9A227; font-style:italic;
                  font-family:'Playfair Display',serif;
                  font-size:1.02rem; font-weight:600;
                  margin-bottom:0.9rem; line-height:1.55;">
            "Prediksi Harga Emas Menggunakan Arsitektur Hybrid CNN-LSTM
            Berbasis Data Time Series Multivariate"
        </p>
        <p>
            Sistem <strong>GoldVision AI</strong> merupakan aplikasi peramalan harga emas
            yang dikembangkan sebagai bagian dari penelitian skripsi akhir.
            Model menggabungkan kekuatan
            <strong>Convolutional Neural Network (CNN)</strong> untuk
            ekstraksi fitur lokal dan
            <strong>Long Short-Term Memory (LSTM)</strong> untuk
            menangkap dependensi temporal jangka panjang dalam data time series.
        </p>
        <p style="margin-top:0.75rem;">
            Prediksi dilakukan secara <em>multi-step recursive</em> dengan menggunakan
            data bursa real-time yang diambil langsung dari Yahoo Finance API,
            mencakup empat instrumen pasar global sebagai fitur multivariat.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # [UI] Header sub-section variabel pengaruh
    st.markdown("""
    <p class="section-eyebrow">Analisis Multivariat</p>
    <p class="section-title" style="font-family:'Playfair Display',serif;
       font-size:1.2rem; font-weight:700; color:#E8ECF1; margin-bottom:1rem;">
       Variabel Pengaruh Terhadap Harga Emas
    </p>
    """, unsafe_allow_html=True)

    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:
        st.markdown("""
        <div class="var-card">
            <div class="var-icon">💵</div>
            <div class="var-title">USD Index (DXY)</div>
            <span class="var-corr corr-neg">● Korelasi Negatif</span>
            <p>
                Emas dihargai dalam Dolar AS, sehingga terdapat korelasi negatif yang kuat
                antara keduanya. Ketika Dolar menguat, harga emas cenderung turun karena
                lebih mahal bagi pemegang mata uang lain. Pelemahan Dolar mendorong kenaikan
                harga emas sebagai instrumen lindung nilai yang andal.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_v2:
        st.markdown("""
        <div class="var-card">
            <div class="var-icon">🛢️</div>
            <div class="var-title">Minyak Mentah (Crude Oil)</div>
            <span class="var-corr corr-pos">● Korelasi Positif</span>
            <p>
                Harga minyak mentah memiliki korelasi positif dengan harga emas.
                Kenaikan harga minyak sering menjadi indikator tekanan inflasi global.
                Dalam kondisi inflasi tinggi, investor beralih ke emas sebagai
                <em>safe haven</em>, mendorong kenaikan harga secara bersamaan.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_v3:
        st.markdown("""
        <div class="var-card">
            <div class="var-icon">📉</div>
            <div class="var-title">S&P 500</div>
            <span class="var-corr corr-inv">● Korelasi Invers</span>
            <p>
                Indeks S&P 500 merepresentasikan performa pasar saham AS. Hubungannya
                dengan emas bersifat invers: saat pasar bearish atau terjadi
                ketidakpastian ekonomi, investor beralih ke emas. Saat bullish, emas
                cenderung kehilangan daya tarik dibanding ekuitas berisiko.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# [UI] TAB 3: TEKNOLOGI & ARSITEKTUR — Tech stack + layer diagram
# =====================================================================

with tab3:

    # [UI] Tech stack dalam chip pill navy
    st.markdown("""
    <div class="card">
        <h3>⚙️ Tech Stack</h3>
        <p style="color:#A0AEC0; font-size:0.92rem; margin-bottom:1rem;">
            Teknologi utama yang digunakan dalam pengembangan sistem GoldVision AI:
        </p>
        <div>
            <span class="tech-chip">🐍 Python 3.x</span>
            <span class="tech-chip">🎈 Streamlit</span>
            <span class="tech-chip">📊 yfinance</span>
            <span class="tech-chip">🧠 TensorFlow / Keras</span>
            <span class="tech-chip">🔢 NumPy</span>
            <span class="tech-chip">🐼 Pandas</span>
            <span class="tech-chip">📈 Matplotlib</span>
            <span class="tech-chip">⚙️ Scikit-learn</span>
            <span class="tech-chip">💾 joblib</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # [UI] Arsitektur model — layer blocks visual bertingkat
    st.markdown("""
    <p class="section-eyebrow">Deep Learning Architecture</p>
    <p class="section-title" style="font-family:'Playfair Display',serif;
       font-size:1.2rem; font-weight:700; color:#E8ECF1; margin-bottom:1rem;">
       Arsitektur Model: Hybrid CNN-LSTM
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <p style="color:#A0AEC0; font-size:0.92rem; margin-bottom:1.25rem;">
            Model menggabungkan dua arsitektur deep learning dalam pipeline sekuensial
            untuk memaksimalkan kemampuan ekstraksi fitur dan prediksi temporal:
        </p>

        <div class="arch-block">
            <strong>📥 Input Layer</strong>
            <p>Shape: <code>(10, 4)</code> — Window 10 hari bursa × 4 fitur
            (USD Index, Minyak Mentah, S&P 500, Harga Emas)</p>
        </div>

        <div class="arch-block">
            <strong>🔍 Conv1D Layer</strong>
            <p><code>64 filters</code>, <code>kernel_size=2</code>, aktivasi <code>ReLU</code> —
            Mengekstraksi pola lokal (local pattern) dan fitur spasial dari data time series
            multivariat. Kernel size 2 menangkap hubungan antar dua timestep berurutan.</p>
        </div>

        <div class="arch-block">
            <strong>🧠 LSTM Layer</strong>
            <p><code>50 units</code>, <code>return_sequences=False</code> —
            Menangkap dependensi temporal jangka panjang dan memori tren sekuensial.
            LSTM mengingat pola dari masa lampau yang relevan untuk prediksi masa depan.</p>
        </div>

        <div class="arch-block">
            <strong>💧 Dropout Layer</strong>
            <p><code>rate=0.20</code> — Regularisasi untuk mencegah overfitting
            dengan menonaktifkan 20% neuron secara acak saat training.</p>
        </div>

        <div class="arch-block">
            <strong>📤 Dense Output Layer</strong>
            <p><code>1 neuron</code>, aktivasi <code>linear</code> —
            Menghasilkan satu nilai prediksi harga emas (normalized).</p>
        </div>

        <div class="recursive-box">
            <strong>🔄 Metode Peramalan: Recursive Multi-Step Forecasting</strong>
            <p>
                Model melakukan prediksi satu langkah ke depan, lalu hasil prediksi
                digunakan sebagai input untuk langkah berikutnya. Variabel eksogen
                (USD, Minyak, S&P 500) dibekukan dari nilai terakhir yang diketahui,
                mengikuti prinsip <em>Random Walk Hypothesis</em>. Inverse transform
                menggunakan <em>dummy matrix (n, 4)</em> agar scaler tidak error.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# [UI] TAB 4: RAPOR EVALUASI MODEL — Dashboard metrik & interpretasi
# =====================================================================

with tab4:

    # [UI] Header section evaluasi
    st.markdown("""
    <p class="section-eyebrow">Performance Report</p>
    <p class="section-title" style="font-family:'Playfair Display',serif;
       font-size:1.25rem; font-weight:700; color:#E8ECF1; margin-bottom:0.3rem;">
       Rapor Evaluasi Akurasi Model
    </p>
    <p class="section-sub">
        Metrik evaluasi model Hybrid CNN-LSTM yang diuji pada <em>test set</em>.
        Setiap metrik menunjukkan seberapa dekat prediksi model dengan harga emas aktual.
    </p>
    """, unsafe_allow_html=True)

    # [UI] 3 Metric Cards evaluasi — sejajar dalam 3 kolom
    col_e1, col_e2, col_e3 = st.columns(3)

    with col_e1:
        # [UI] MAPE 1.91% — nilai emas + badge hijau "Sangat Akurat"
        st.markdown("""
        <div class="metric-box">
            <div class="m-label">MAPE</div>
            <div class="m-value m-value-gold">1.61%</div>
            <span class="m-badge-green">✅ Sangat Akurat</span>
            <div class="m-sub" style="margin-top:0.65rem;">
                Mean Absolute Percentage Error
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_e2:
        # [UI] RMSE 85.39 USD
        st.markdown("""
        <div class="metric-box">
            <div class="m-label">RMSE</div>
            <div class="m-value">75.43</div>
            <div class="m-sub">USD &nbsp;—&nbsp; Root Mean Squared Error</div>
        </div>
        """, unsafe_allow_html=True)

    with col_e3:
        # [UI] MAE 66.00 USD
        st.markdown("""
        <div class="metric-box">
            <div class="m-label">MAE</div>
            <div class="m-value">56.16</div>
            <div class="m-sub">USD &nbsp;—&nbsp; Mean Absolute Error</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # [UI] Card interpretasi metrik dengan border kiri emas
    st.markdown("""
    <div class="card card-accent">
        <h3>📚 Interpretasi Metrik</h3>
        <ul>
            <li>
                <strong>MAPE = 1.91%</strong> — Rata-rata deviasi prediksi hanya
                1.61% dari harga aktual. Menurut standar Lewis (1982), MAPE di bawah
                10% dikategorikan sebagai <em>"highly accurate forecasting"</em>.
                Nilai ini sangat kuat untuk model peramalan harga komoditas global.
            </li>
            <br>
            <li>
                <strong>RMSE = 75.43 USD</strong> — Akar kuadrat dari rata-rata
                kuadrat error. RMSE lebih sensitif terhadap outlier dibandingkan MAE,
                memberikan gambaran kestabilan prediksi secara menyeluruh. Nilai ini
                relatif kecil mengingat harga emas berada di kisaran ribuan USD.
            </li>
            <br>
            <li>
                <strong>MAE = 56.16 USD</strong> — Rata-rata absolut penyimpangan
                prediksi dari nilai aktual. Secara rata-rata, prediksi model hanya
                meleset sekitar $56.16 dari harga emas sebenarnya — akurasi sangat
                tinggi untuk instrumen komoditas global yang volatil.
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gold-rule"></div>', unsafe_allow_html=True)

    # [UI] Summary 3 variabel input — secara visual ringkas di Tab 4
    st.markdown("""
    <p class="section-eyebrow">Model Input Summary</p>
    <p class="section-title" style="font-family:'Playfair Display',serif;
       font-size:1.1rem; font-weight:700; color:#E8ECF1; margin-bottom:0.75rem;">
       3 Variabel Eksogen yang Mempengaruhi Model
    </p>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)

    for col, icon, title, desc, badge, cls in [
        (
            col_s1, "💵", "USD Index",
            "Korelasi negatif terhadap emas. Saat dolar menguat, harga emas turun karena lebih mahal bagi pemegang mata uang lain.",
            "Korelasi Negatif", "corr-neg"
        ),
        (
            col_s2, "🛢️",  "Minyak Mentah",
            "Proxy inflasi global. Kenaikan harga minyak mendorong permintaan emas sebagai instrumen lindung nilai.",
            "Korelasi Positif", "corr-pos"
        ),
        (
            col_s3, "📈", "S&P 500",
            "Indikator risk appetite pasar. Kondisi bearish saham mendorong aliran modal ke emas (safe haven).",
            "Korelasi Invers", "corr-inv"
        ),
    ]:
        with col:
            st.markdown(f"""
            <div class="var-card" style="padding:1.35rem;">
                <div style="font-size:1.75rem; margin-bottom:0.5rem;">{icon}</div>
                <div class="var-title" style="font-size:0.95rem;">{title}</div>
                <span class="var-corr {cls}" style="font-size:0.69rem;
                      margin-bottom:0.65rem; display:inline-block;">{badge}</span>
                <p style="font-size:0.84rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# [UI] FOOTER PREMIUM — Centered, branding + disclaimer
# =====================================================================

st.markdown("""
<div class="footer-premium">
    <div class="footer-brand">🪙 GoldVision AI</div>
    <div class="footer-sub">
        Hybrid CNN-LSTM Gold Price Forecasting System
        <span class="footer-dot"></span>
        Skripsi Akhir Universitas
        <span class="footer-dot"></span>
        Data: Yahoo Finance API
    </div>
    <div class="footer-sub" style="margin-top:0.45rem; font-size:0.73rem; color:#4A5568;">
        Model dilatih pada data historis &nbsp;·&nbsp;
        Prediksi bukan merupakan saran investasi &nbsp;·&nbsp;
        Hanya untuk keperluan akademis
    </div>
</div>
""", unsafe_allow_html=True)