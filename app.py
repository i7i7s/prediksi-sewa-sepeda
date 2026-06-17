"""
Simulator Prediksi Penyewaan Sepeda Harian
Mata Kuliah: Sains Data (UAS)
Dataset: Capital Bikeshare - day.csv
Design System: Light Mode Professional Minimal
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Konfigurasi Halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Sewa Sepeda",
    page_icon="🚲",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS — Light Mode Minimalist
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Background & Font ── */
.stApp {
    background-color: #F8FAFC !important;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #0F172A;
}

/* ── Main App Container ── */
.main .block-container {
    padding-top: 2rem !important;
    max-width: 1200px !important;
}

/* ── Header Clean ── */
.header-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    border-top: 4px solid #2563EB;
}
.header-card h1 {
    margin: 0;
    color: #1E3A8A;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.header-card p {
    margin: 0.25rem 0 0 0;
    color: #64748B;
    font-size: 0.95rem;
}

/* ── Label Section ── */
.section-heading {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: #0F172A;
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #E2E8F0;
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: #EFF6FF;
    color: #2563EB;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-left: auto;
}
.badge-auto   { background: #DCFCE7; color: #166534; }
.badge-manual { background: #FEF3C7; color: #92400E; }
.badge-hybrid { background: #F1F5F9; color: #475569; }

/* ── Panel Putih (Card) ── */
.white-panel {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
}

/* ── Auto Info Bar ── */
.info-bar {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #3B82F6;
    padding: 0.85rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    color: #334155;
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* ── Summary Data Sheet ── */
.data-sheet {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
}
.sheet-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px dashed #E2E8F0;
    font-size: 0.85rem;
}
.sheet-row:last-child {
    border-bottom: none;
}
.sheet-label {
    color: #64748B;
}
.sheet-value {
    font-weight: 600;
    color: #0F172A;
}

/* ── Result Highlight Card ── */
.result-card {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    margin-top: 1rem;
}
.result-title {
    color: #1E3A8A;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.result-val {
    color: #1D4ED8;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}
.result-desc {
    color: #475569;
    font-size: 0.85rem;
}

/* ── Tweak Streamlit Inputs ── */
div[data-testid="stSelectbox"] label p,
div[data-testid="stRadio"] label p,
div[data-testid="stSlider"] label p {
    font-weight: 500 !important;
    color: #334155 !important;
    font-size: 0.9rem !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Mapping & Konstanta
# ─────────────────────────────────────────────
NAMA_BULAN   = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
                7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}
NAMA_MUSIM   = {1:"Semi (Spring)",2:"Panas (Summer)",3:"Gugur (Fall)",4:"Dingin (Winter)"}
NAMA_CUACA   = {1:"Cerah / Sedikit Berawan",2:"Mendung / Berkabut",3:"Hujan / Salju Ringan"}
NAMA_WEEKDAY = {0:"Minggu",1:"Senin",2:"Selasa",3:"Rabu",4:"Kamis",5:"Jumat",6:"Sabtu"}
NAMA_YR      = {0:"2011",1:"2012"}

# Mapping ke OHE
SEASON_OHE   = {1:"springer",2:"summer",3:"fall",4:"winter"}
MNTH_OHE     = {1:"jan",2:"feb",3:"mar",4:"apr",5:"may",6:"jun",
                7:"jul",8:"aug",9:"sep",10:"oct",11:"nov",12:"dec"}
WEEKDAY_OHE  = {0:"minggu",1:"senin",2:"selasa",3:"rabu",4:"kamis",5:"jumat",6:"sabtu"}
WEATHER_OHE  = {1:"cerah",2:"mendung",3:"hujan_ringan"}
MUSIM_BULAN  = {1:4,2:4,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3,12:4}

# ─────────────────────────────────────────────
#  Load Data & Model
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("day.csv")

@st.cache_resource
def load_model():
    try:
        import sys
        try:
            import sklearn._loss._loss as loss_cy
            sys.modules['_loss'] = loss_cy
        except ImportError:
            try:
                import sklearn._loss as loss_py
                sys.modules['_loss'] = loss_py
            except ImportError:
                pass
        model = joblib.load("model_bike_sharing.pkl")
        return model, None
    except Exception as e:
        return None, str(e)

df               = load_data()
model, model_err = load_model()

if model_err:
    st.error(f"Gagal memuat model: {model_err}")
    st.stop()

FITUR_MODEL = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else []
def modus(s): return s.mode().iloc[0]

def build_input_df(yr, season, mnth, holiday, weekday, workingday, weathersit,
                   temp, hum, windspeed):
    row = {f: 0 for f in FITUR_MODEL}
    for k, v in [("yr", yr), ("holiday", holiday), ("temp", temp),
                 ("hum", hum), ("windspeed", windspeed)]:
        if k in row: row[k] = v
    for prefix, mapping, val in [
        ("season",    SEASON_OHE,  season),
        ("mnth",      MNTH_OHE,    mnth),
        ("weekday",   WEEKDAY_OHE, weekday),
        ("weathersit",WEATHER_OHE, weathersit),
    ]:
        col = f"{prefix}_{mapping.get(val,'')}"
        if col in row: row[col] = 1
    return pd.DataFrame([row])[FITUR_MODEL]

# ─────────────────────────────────────────────
#  Header Layout
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>Analitik Prediksi Penyewaan Sepeda</h1>
    <p>Model Gradient Boosting Regressor | Capital Bikeshare Data</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Main Layout
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="white-panel">', unsafe_allow_html=True)
    
    # -- 1. BULAN & MUSIM --
    st.markdown("""
    <div class="section-heading">
        <span class="step-number">1</span> Bulan & Musim
        <span class="badge badge-auto">Auto</span>
    </div>
    """, unsafe_allow_html=True)
    
    bulan_dipilih = st.selectbox(
        "Bulan Target Prediksi:",
        options=list(NAMA_BULAN.keys()),
        format_func=lambda x: NAMA_BULAN[x],
        index=6,
        label_visibility="collapsed"
    )

    df_bulan       = df[df["mnth"] == bulan_dipilih].copy()
    auto_season    = MUSIM_BULAN[bulan_dipilih]
    auto_workingday= int(modus(df_bulan["workingday"]))

    st.markdown(f"""
    <div class="info-bar">
        Otomatis diatur untuk <b>{NAMA_BULAN[bulan_dipilih]}</b>:<br>
        Musim: <b>{NAMA_MUSIM[auto_season]}</b> &nbsp;|&nbsp; Hari Kerja Mayoritas: <b>{'Ya' if auto_workingday==1 else 'Tidak'}</b>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # -- 2. KONDISI LINGKUNGAN --
    st.markdown("""
    <div class="section-heading">
        <span class="step-number">2</span> Kondisi Cuaca & Lingkungan
        <span class="badge badge-hybrid">Histori / Manual</span>
    </div>
    """, unsafe_allow_html=True)

    hist_temp = round(float(df_bulan["temp"].mean()), 2)
    hist_hum  = round(float(df_bulan["hum"].mean()), 2)
    hist_wind = round(float(df_bulan["windspeed"].mean()), 2)

    weathersit_pilih = st.selectbox(
        "Situasi Cuaca Umum:",
        options=[1, 2, 3],
        format_func=lambda x: NAMA_CUACA[x],
        index=0,
    )

    temp_pilih = st.slider(
        "Suhu (Skala 0–1):",
        min_value=0.0, max_value=1.0, value=hist_temp, step=0.01,
    )
    hum_pilih = st.slider(
        "Kelembapan (Skala 0–1):",
        min_value=0.0, max_value=1.0, value=hist_hum, step=0.01,
    )
    wind_pilih = st.slider(
        "Kecepatan Angin (Skala 0–1):",
        min_value=0.0, max_value=1.0, value=hist_wind, step=0.01,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -- 3. PARAMETER WAKTU --
    st.markdown("""
    <div class="section-heading">
        <span class="step-number">3</span> Parameter Waktu & Kalender
        <span class="badge badge-manual">Manual</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        yr_pilih = st.selectbox("Tahun:", options=[0, 1], format_func=lambda x: NAMA_YR[x], index=1)
    with c2:
        holiday_pilih = st.selectbox("Status Libur:", options=[0, 1], format_func=lambda x: "Bukan Libur Nasional" if x==0 else "Hari Libur Nasional", index=0)
    weekday_pilih = st.selectbox("Hari dalam Seminggu:", options=list(NAMA_WEEKDAY.keys()), format_func=lambda x: NAMA_WEEKDAY[x], index=3)
    
    st.markdown('</div>', unsafe_allow_html=True) # End of left panel

with col_right:
    st.markdown("""
    <div class="section-heading">
        <span class="step-number">ℹ️</span> Ringkasan Input Model
    </div>
    """, unsafe_allow_html=True)

    def data_row(label, val):
        return f'<div class="sheet-row"><span class="sheet-label">{label}</span><span class="sheet-value">{val}</span></div>'

    rows_html = "".join([
        data_row("Tahun", f"{NAMA_YR[yr_pilih]}"),
        data_row("Bulan", f"{NAMA_BULAN[bulan_dipilih]}"),
        data_row("Musim", f"{NAMA_MUSIM[auto_season]}"),
        data_row("Hari", f"{NAMA_WEEKDAY[weekday_pilih]}"),
        data_row("Libur Nasional", f"{'Ya' if holiday_pilih==1 else 'Tidak'}"),
        data_row("Kategori Hari Kerja", f"{'Ya' if auto_workingday==1 else 'Tidak'}"),
        data_row("Cuaca", f"{NAMA_CUACA[weathersit_pilih]}"),
        data_row("Suhu (temp)", f"{temp_pilih:.2f}"),
        data_row("Kelembapan (hum)", f"{hum_pilih:.2f}"),
        data_row("Angin (windspeed)", f"{wind_pilih:.2f}"),
    ])

    st.markdown(f'<div class="data-sheet">{rows_html}</div>', unsafe_allow_html=True)

    predict_btn = st.button("Hitung Prediksi Sewa", type="primary", use_container_width=True)

    if predict_btn:
        try:
            df_input = build_input_df(
                yr=yr_pilih, season=auto_season, mnth=bulan_dipilih,
                holiday=holiday_pilih, weekday=weekday_pilih,
                workingday=auto_workingday, weathersit=weathersit_pilih,
                temp=temp_pilih, hum=hum_pilih, windspeed=wind_pilih,
            )

            pred_raw = model.predict(df_input)[0]
            prediksi = max(0, int(round(pred_raw)))
            pred_fmt = f"{prediksi:,}".replace(",", ".")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">Estimasi Jumlah Penyewaan</div>
                <div class="result-val">{pred_fmt}</div>
                <div class="result-desc">Sepeda / hari pada {NAMA_BULAN[bulan_dipilih]} {NAMA_YR[yr_pilih]}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal memproses model: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 2rem; color: #94A3B8; border: 2px dashed #E2E8F0; border-radius: 12px; margin-top: 1rem;">
            Pilih parameter di sebelah kiri, lalu klik <b>Hitung Prediksi</b> untuk melihat estimasi sewa.
        </div>
        """, unsafe_allow_html=True)
