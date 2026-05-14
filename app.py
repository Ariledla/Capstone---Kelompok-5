import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI DASAR
# ============================================================

st.set_page_config(
    page_title="Prediksi Sampah Kota Bandung",
    page_icon="♻️",
    layout="wide"
)

FILE_NAME = "jumlah_capaian_penanganan_sampah_di_kota_bandung.xlsx"

BULAN_MAP = {
    "JANUARI": 1,
    "FEBRUARI": 2,
    "MARET": 3,
    "APRIL": 4,
    "MEI": 5,
    "JUNI": 6,
    "JULI": 7,
    "AGUSTUS": 8,
    "SEPTEMBER": 9,
    "OKTOBER": 10,
    "NOVEMBER": 11,
    "DESEMBER": 12,
}

BULAN_INDO = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


# ============================================================
# FUNGSI UTAMA
# ============================================================

def format_periode(date_value):
    return f"{BULAN_INDO[date_value.month]} {date_value.year}"


def evaluate_model(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    r2 = r2_score(actual, forecast)
    return mae, rmse, mape, r2


@st.cache_data
def load_data():
    df_raw = pd.read_excel(FILE_NAME)

    df = df_raw.copy()
    df["bulan_num"] = df["bulan"].str.upper().map(BULAN_MAP)

    df["tanggal"] = pd.to_datetime({
        "year": df["tahun"],
        "month": df["bulan_num"],
        "day": 1
    })

    df = df.sort_values("tanggal")
    df = df.set_index("tanggal")

    ts = df["jumlah_sampah"].asfreq("MS")

    return df_raw, df, ts


def apply_theme(mode):
    if mode == "Terang":
        bg = "#EFE9DC"
        card = "#FFFDF7"
        text = "#172018"
        muted = "#3F4B3E"
        border = "#CFC7B8"
        accent = "#2E6F4F"
        accent_soft = "rgba(46, 111, 79, 0.11)"
        accent_hover = "#2E6F4F"
        hero = "linear-gradient(135deg, #2E6F4F 0%, #6B9A61 56%, #B88A3D 100%)"
        sidebar_bg = """
            radial-gradient(circle at 14% 8%, rgba(107, 154, 97, 0.22), transparent 24%),
            radial-gradient(circle at 90% 23%, rgba(184, 138, 61, 0.18), transparent 25%),
            linear-gradient(180deg, #EFECDD 0%, #E7E3D3 48%, #EDE3CF 100%)
        """
        sidebar_visual = "linear-gradient(135deg, #2E6F4F 0%, #527D52 60%, #8C6B31 100%)"
        plot_bg = "#FFFFFF"

        table_bg = "#FFFDF7"
        table_header = "#DDEADB"
        table_text = "#172018"
        table_border = "#CFC7B8"
        table_stripe = "#F5F1E8"

    else:
        bg = "#151A17"
        card = "#222A24"
        text = "#F5F7F2"
        muted = "#D8E0D4"
        border = "#3D4A40"
        accent = "#8BCB88"
        accent_soft = "rgba(139, 203, 136, 0.12)"
        accent_hover = "#2F7D52"
        hero = "linear-gradient(135deg, #1F4D36 0%, #4F8B59 55%, #B78335 100%)"
        sidebar_bg = """
            radial-gradient(circle at 8% 12%, rgba(139, 203, 136, 0.28), transparent 16%),
            radial-gradient(circle at 92% 28%, rgba(47, 111, 78, 0.18), transparent 17%),
            radial-gradient(circle at 18% 88%, rgba(226, 177, 93, 0.18), transparent 18%),
            linear-gradient(180deg, #1E241F 0%, #1B241C 60%, #222819 100%)
        """
        sidebar_visual = "linear-gradient(135deg, #26382C 0%, #2F6F4E 65%, #6A4A1E 100%)"
        plot_bg = "#FFFFFF"

        table_bg = "#222A24"
        table_header = "#2B342D"
        table_text = "#F5F7F2"
        table_border = "#3D4A40"
        table_stripe = "#263029"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 2% 4%, rgba(139, 203, 136, 0.06), transparent 20%),
                {bg} !important;
            color: {text} !important;
        }}

        [data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {border};
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 0rem !important;
        }}

        [data-testid="stSidebarContent"] {{
            padding-top: 0rem !important;
        }}

        [data-testid="stSidebarUserContent"] {{
            padding-top: 0rem !important;
            margin-top: -2.25rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-bottom: 260px !important;
        }}

        [data-testid="stSidebar"] * {{
            color: {text} !important;
        }}

        .theme-label {{
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
            color: {text} !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
            gap: 0.55rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            width: 100% !important;
            height: 42px !important;
            border-radius: 14px !important;
            border: 1px solid {border} !important;
            background: {card} !important;
            color: {text} !important;
            font-weight: 850 !important;
            font-size: 13px !important;
            box-shadow: 0 8px 18px rgba(0,0,0,0.08) !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {accent_hover} !important;
            color: white !important;
            border-color: {accent_hover} !important;
            transform: translateY(-1px);
        }}

        [data-testid="stSidebar"] .stButton > button:focus,
        [data-testid="stSidebar"] .stButton > button:active {{
            background: {accent_hover} !important;
            color: white !important;
            border-color: {accent_hover} !important;
            box-shadow: 0 0 0 2px {accent_soft} !important;
        }}

        [data-testid="stSidebar"] {{
            width: 304px !important;
            min-width: 304px !important;
            background: {sidebar_bg} !important;
            border-right: 1px solid {border};
        }}

        [data-testid="stSidebarContent"] {{
            width: 304px !important;
        }}        

        .sidebar-visual {{
            background: {sidebar_visual};
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 22px;
            padding: 17px 16px;
            margin: 0 !important;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
            position: fixed !important;
            left: 20px !important;
            bottom: 24px !important;
            width: 264px !important;
            max-width: 264px !important;
            box-sizing: border-box !important;
            overflow: hidden;
            z-index: 20;
        }}

        .sidebar-visual::before {{
            content: "";
            position: absolute;
            width: 95px;
            height: 95px;
            right: -25px;
            top: -28px;
            background: rgba(255, 255, 255, 0.14);
            border-radius: 50%;
        }}

        .sidebar-visual::after {{
            display: none !important;
            content: none !important;
        }}

        .sidebar-emoji {{
            font-size: 42px;
            line-height: 1;
            margin-bottom: 8px;
            position: relative;
            z-index: 2;
        }}

        .sidebar-visual-title {{
            font-size: 18px;
            font-weight: 850;
            position: relative;
            z-index: 2;
            color: white !important;
        }}

        .sidebar-visual-subtitle {{
            font-size: 12.5px;
            color: white !important;
            margin-top: 5px;
            line-height: 1.45;
            position: relative;
            z-index: 2;
            font-weight: 500;
        }}

        .team-name {{
            margin-top: 10px;
            padding: 8px 10px;
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.24);
            font-size: 12.5px;
            font-weight: 750;
            position: relative;
            z-index: 2;
            color: white !important;
            line-height: 1.45;
            text-align: center !important;
            display: flex;
            justify-content: center;
            align-items: center;
        }}

        .block-container {{
            max-width: 1420px !important;
            padding-top: 4rem !important;
            padding-left: 0.95rem !important;
            padding-right: 0.95rem !important;
            padding-bottom: 1.5rem !important;
        }}

        header, footer, #MainMenu {{
            visibility: hidden;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: inherit;
        }}

        .hero {{
            background: {hero};
            color: white !important;
            padding: 25px 31px;
            border-radius: 24px;
            margin-bottom: 26px;
            box-shadow: 0 18px 42px rgba(31, 41, 51, 0.18);
        }}

        .hero * {{
            color: white !important;
        }}

        .hero-title {{
            font-size: 35px;
            font-weight: 850;
            line-height: 1.12;
            margin-bottom: 8px;
        }}

        .hero-subtitle {{
            font-size: 15px;
            max-width: 1120px;
            opacity: 0.96;
        }}

        .section-title {{
            font-size: 25px;
            font-weight: 850;
            color: {text} !important;
            margin-bottom: 7px;
        }}

        .section-desc {{
            color: {muted} !important;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 22px;
        }}

        .small-title {{
            font-size: 16.5px;
            font-weight: 800;
            color: {text} !important;
            margin-bottom: 12px;
        }}

        .card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 20px;
            padding: 19px 21px;
            margin-bottom: 20px;
            box-shadow: 0 8px 26px rgba(31, 41, 51, 0.06);
        }}

        .text-muted {{
            color: {muted} !important;
            font-size: 14px;
            font-weight: 500;
            line-height: 1.7;
        }}

        .text-muted li {{
            color: {muted} !important;
            margin-bottom: 8px;
        }}

        .kpi-card {{
            background: {card};
            border: 1px solid {border};
            border-radius: 21px;
            padding: 20px 22px;
            min-height: 112px;
            box-shadow: 0 8px 26px rgba(31, 41, 51, 0.06);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            gap: 10px;
            margin-bottom: 24px;
        }}

        .kpi-label {{
            color: {muted} !important;
            font-size: 17px;
            font-weight: 900;
            line-height: 1.2;
            text-align: center;
            white-space: nowrap;
        }}

        .kpi-value {{
            color: {text} !important;
            font-size: 27px;
            font-weight: 900;
            line-height: 1.12;
            text-align: center;
            white-space: nowrap;
        }}

        .kpi-value.small {{
            font-size: 26px;
        }}

        .model-card {{
            background:
                linear-gradient(145deg, {card} 0%, {accent_soft} 100%);
            border: 1px solid {border};
            border-radius: 24px;
            padding: 22px 20px;
            min-height: 180px;
            box-shadow: 0 10px 28px rgba(31, 41, 51, 0.07);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            text-align: center;
            gap: 11px;
            margin-bottom: 34px;
            position: relative;
            overflow: hidden;
        }}

        .model-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 20%;
            right: 20%;
            height: 5px;
            background: {accent};
            border-radius: 0 0 999px 999px;
        }}

        .model-icon {{
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: {accent_soft};
            border: 1px solid {border};
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-top: 5px;
        }}

        .model-title {{
            font-size: 16px;
            font-weight: 900;
            color: {text} !important;
        }}

        .model-text {{
            font-size: 13.5px;
            line-height: 1.6;
            color: {muted} !important;
            font-weight: 500;
        }}

        div[data-baseweb="select"] > div {{
            background: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 14px !important;
            color: {text} !important;
            min-height: 42px !important;
            cursor: pointer !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="select"],
        div[data-baseweb="select"] *,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] *,
        div[data-baseweb="popover"] [role="option"] {{
            cursor: pointer !important;
        }}

        div[data-baseweb="select"] input {{
            cursor: pointer !important;
            caret-color: transparent !important;
        }}

        div[data-baseweb="select"] span {{
            color: {text} !important;
            font-weight: 650 !important;
        }}

        div[data-baseweb="select"] svg {{
            color: {accent} !important;
            fill: {accent} !important;
            opacity: 1 !important;
        }}

        div[data-baseweb="popover"] {{
            z-index: 999999 !important;
        }}

        div[data-baseweb="popover"] [role="listbox"] {{
            background: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 14px !important;
            padding: 6px !important;
            overflow: hidden !important;
            box-shadow: 0 18px 45px rgba(0,0,0,0.28) !important;
        }}

        div[data-baseweb="popover"] [role="option"] {{
            background: transparent !important;
            color: {text} !important;
            min-height: 42px !important;
            padding: 0 !important;
            margin: 0 0 4px 0 !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            overflow: hidden !important;
            transition: all 0.15s ease-in-out !important;
        }}

        div[data-baseweb="popover"] [role="option"]:last-child {{
            margin-bottom: 0 !important;
        }}

        div[data-baseweb="popover"] [role="option"] > div {{
            width: 100% !important;
            min-height: 42px !important;
            padding: 10px 14px !important;
            display: flex !important;
            align-items: center !important;
            background: transparent !important;
            color: {text} !important;
            border-radius: 10px !important;
            box-sizing: border-box !important;
        }}

        div[data-baseweb="popover"] [role="option"] div,
        div[data-baseweb="popover"] [role="option"] span {{
            background: transparent !important;
            color: {text} !important;
            font-weight: 750 !important;
            line-height: 1.2 !important;
        }}

        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
            background: {accent_hover} !important;
            color: white !important;
        }}

        div[data-baseweb="popover"] [role="option"]:hover > div,
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] > div {{
            background: {accent_hover} !important;
            color: white !important;
        }}

        div[data-baseweb="popover"] [role="option"]:hover div,
        div[data-baseweb="popover"] [role="option"]:hover span,
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] div,
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] span {{
            background: transparent !important;
            color: white !important;
        }}

        .stSlider label, .stSelectbox label, .stRadio label {{
            color: {text} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] span {{
            color: {text} !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 12px !important;
            padding: 4px 8px !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {accent_soft} !important;
            transform: translateX(2px);
        }}

        .custom-table-wrapper {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid {table_border};
            border-radius: 14px;
            background: {table_bg};
            margin-bottom: 24px;
        }}

        table.custom-table {{
            width: 100%;
            border-collapse: collapse;
            background: {table_bg} !important;
            color: {table_text} !important;
            font-size: 13px;
        }}

        table.custom-table thead tr th {{
            background: {table_header} !important;
            color: {table_text} !important;
            font-weight: 800;
            padding: 11px 13px;
            border-bottom: 1px solid {table_border};
            text-align: left;
            white-space: nowrap;
        }}

        table.custom-table tbody tr td,
        table.custom-table tbody tr th {{
            background: {table_bg} !important;
            color: {table_text} !important;
            padding: 10px 13px;
            border-bottom: 1px solid {table_border};
            font-weight: 500;
            white-space: nowrap;
        }}

        table.custom-table tbody tr:nth-child(even) td,
        table.custom-table tbody tr:nth-child(even) th {{
            background: {table_stripe} !important;
        }}

        table.custom-table tbody tr:last-child td,
        table.custom-table tbody tr:last-child th {{
            border-bottom: none;
        }}

        .stPlotlyChart, .stPyplot {{
            margin-bottom: 22px !important;
        }}

        button[kind="secondary"] {{
            border-radius: 14px !important;
            border: 1px solid {border} !important;
        }}

        hr {{
            border-color: {border};
            margin-top: 1.2rem !important;
            margin-bottom: 1.2rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    return {"plot_bg": plot_bg}


def kpi_card(label, value, small=False):
    small_class = "small" if small else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {small_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def card(title, body):
    st.markdown(
        f"""
        <div class="card">
            <div class="small-title">{title}</div>
            <div class="text-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def bullet_card(title, items):
    html = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f"""
        <div class="card">
            <div class="small-title">{title}</div>
            <ul class="text-muted" style="padding-left: 20px; margin-bottom: 0;">
                {html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def model_card(icon, title, body):
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-icon">{icon}</div>
            <div class="model-title">{title}</div>
            <div class="model-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def style_plot(ax, title):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.30)
    return ax


def show_table(data):
    table_data = data.copy()
    html = table_data.to_html(
        classes="custom-table",
        border=0,
        index=False,
        escape=False
    )

    st.markdown(
        f"""
        <div class="custom-table-wrapper">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD DATA
# ============================================================

df_raw, df, ts = load_data()

train = ts.iloc[:-12]
test = ts.iloc[-12:]

provinsi = ", ".join(df_raw["nama_provinsi"].dropna().unique())
kota = ", ".join(df_raw["bps_nama_kabupaten_kota"].dropna().unique())
satuan = ", ".join(df_raw["satuan"].dropna().unique())

# ============================================================
# MODEL EVALUASI 2024
# ============================================================

naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)

seasonal_naive_forecast = train.iloc[-12:].copy()
seasonal_naive_forecast.index = test.index

arima_model = ARIMA(train, order=(0, 1, 1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))
arima_forecast.index = test.index

sarima_model = SARIMAX(
    train,
    order=(1, 2, 2),
    seasonal_order=(0, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)
sarima_forecast = sarima_fit.forecast(steps=len(test))
sarima_forecast.index = test.index

naive_eval = evaluate_model(test, naive_forecast)
seasonal_naive_eval = evaluate_model(test, seasonal_naive_forecast)
arima_eval = evaluate_model(test, arima_forecast)
sarima_eval = evaluate_model(test, sarima_forecast)

eval_df = pd.DataFrame({
    "Model": [
        "Naive Forecast",
        "Seasonal Naive Forecast",
        "ARIMA(0,1,1)",
        "SARIMA(1,2,2)(0,1,1,12)"
    ],
    "MAE": [
        naive_eval[0],
        seasonal_naive_eval[0],
        arima_eval[0],
        sarima_eval[0]
    ],
    "RMSE": [
        naive_eval[1],
        seasonal_naive_eval[1],
        arima_eval[1],
        sarima_eval[1]
    ],
    "MAPE (%)": [
        naive_eval[2],
        seasonal_naive_eval[2],
        arima_eval[2],
        sarima_eval[2]
    ],
    "R²": [
        naive_eval[3],
        seasonal_naive_eval[3],
        arima_eval[3],
        sarima_eval[3]
    ]
})

# ============================================================
# SIDEBAR
# ============================================================

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Gelap"

theme = apply_theme(st.session_state.theme_mode)
plot_bg = theme["plot_bg"]

st.sidebar.markdown('<div class="theme-label">Pilih Tampilan</div>', unsafe_allow_html=True)

theme_col1, theme_col2 = st.sidebar.columns(2)

with theme_col1:
    light_label = "   ☀️   "
    if st.session_state.theme_mode == "Terang":
        light_label = "   ☀️   "

    if st.button(light_label, use_container_width=True):
        st.session_state.theme_mode = "Terang"
        st.rerun()

with theme_col2:
    dark_label = "   🌙   "
    if st.session_state.theme_mode == "Gelap":
        dark_label = "   🌙   "

    if st.button(dark_label, use_container_width=True):
        st.session_state.theme_mode = "Gelap"
        st.rerun()

menu = st.sidebar.radio(
    "Menu Utama",
    [
        "Beranda",
        "Data yang Dipakai",
        "Pola Data",
        "Prediksi Sampah",
        "Ketepatan Prediksi",
        "Ringkasan"
    ]
)

st.sidebar.markdown(
    """
    <div class="sidebar-visual">
        <div class="sidebar-emoji">♻️ 🗑️ 🍃</div>
        <div class="sidebar-visual-title">Dashboard Sampah</div>
        <div class="sidebar-visual-subtitle">
            Analisis pola dan prediksi jumlah sampah Kota Bandung.
        </div>
        <div class="team-name">
            Kelompok 5 Capstone
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Prediksi Jumlah Sampah Kota Bandung</div>
        <div class="hero-subtitle">
            Dashboard ini membantu membaca pola jumlah sampah dari waktu ke waktu dan memperkirakan jumlah sampah pada periode berikutnya.
            Tampilan dibuat agar mudah dipahami oleh pengguna umum, bukan hanya pengguna yang memahami proses pemodelan.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# BERANDA
# ============================================================

if menu == "Beranda":
    st.markdown('<div class="section-title">Gambaran Umum</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Ringkasan awal mengenai data, tujuan dashboard, dan hasil penting.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        kpi_card("Jumlah Data", f"{len(df_raw)} baris")

    with col2:
        kpi_card("Periode", f"{ts.index.min().year}–{ts.index.max().year}")

    with col3:
        kpi_card("Wilayah", kota)

    with col4:
        kpi_card("Satuan", satuan)

    left, right = st.columns(2, gap="large")

    with left:
        card(
            "Untuk apa dashboard ini dibuat?",
            """
            Dashboard ini dibuat untuk membantu melihat perubahan jumlah sampah Kota Bandung dari waktu ke waktu.
            Selain itu, dashboard ini menyediakan fitur prediksi untuk memperkirakan jumlah sampah beberapa bulan ke depan.
            """
        )

        bullet_card(
            "Pertanyaan yang ingin dijawab",
            [
                "Apakah jumlah sampah cenderung naik atau turun?",
                "Pada tahun atau bulan mana jumlah sampah terlihat tinggi atau rendah?",
                "Model prediksi mana yang kesalahannya paling kecil?",
                "Berapa perkiraan jumlah sampah untuk beberapa bulan ke depan?"
            ]
        )

    with right:
        bullet_card(
            "Hasil singkat",
            [
                "Jumlah sampah meningkat sampai sekitar 2019–2020.",
                "Setelah itu, jumlah sampah mulai menurun.",
                "Penurunan cukup tajam terlihat pada 2023–2024.",
                "Model ARIMA menjadi model terbaik berdasarkan kesalahan prediksi paling kecil.",
                "Model masih belum sepenuhnya mampu menangkap perubahan tajam."
            ]
        )

        card(
            "Catatan",
            """
            Prediksi pada dashboard ini adalah perkiraan berbasis data historis, bukan angka pasti.
            Semakin jauh bulan yang diprediksi, hasilnya bisa semakin tidak pasti.
            """
        )

# ============================================================
# DATA
# ============================================================

elif menu == "Data yang Dipakai":
    st.markdown('<div class="section-title">Data yang Dipakai</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Bagian ini menjelaskan isi data secara sederhana sebelum dianalisis dan diprediksi.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        kpi_card("Jumlah Baris", f"{len(df_raw)}")

    with col2:
        kpi_card("Jumlah Kolom", f"{df_raw.shape[1]}")

    with col3:
        kpi_card(
            "Periode Data",
            f"{format_periode(ts.index.min())} - {format_periode(ts.index.max())}",
            small=True
        )

    left, right = st.columns(2, gap="large")

    with left:
        bullet_card(
            "Identitas data",
            [
                f"Provinsi: <b>{provinsi}</b>",
                f"Kota/Kabupaten: <b>{kota}</b>",
                f"Satuan: <b>{satuan}</b>",
                "Variabel utama: <b>jumlah_sampah</b>",
                "Data berbentuk bulanan."
            ]
        )

    with right:
        bullet_card(
            "Pemeriksaan awal",
            [
                f"Missing value: <b>{int(df_raw.isnull().sum().sum())}</b>",
                f"Data duplikat: <b>{int(df_raw.duplicated().sum())}</b>",
                "Setiap tahun memiliki 12 bulan data.",
                "Data siap digunakan untuk membaca pola dan membuat prediksi."
            ]
        )

    st.markdown('<div class="small-title">Contoh Data</div>', unsafe_allow_html=True)
    show_table(df_raw.head(8))

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="small-title">Ringkasan Jumlah Sampah</div>', unsafe_allow_html=True)
        ringkasan = df_raw[["jumlah_sampah"]].describe().T.reset_index()
        ringkasan = ringkasan.rename(columns={"index": "Variabel"})
        show_table(ringkasan)

    with col2:
        st.markdown('<div class="small-title">Jumlah Data per Tahun</div>', unsafe_allow_html=True)
        data_per_tahun = df_raw["tahun"].value_counts().sort_index().reset_index()
        data_per_tahun.columns = ["Tahun", "Jumlah Data"]
        show_table(data_per_tahun)

# ============================================================
# POLA DATA
# ============================================================

elif menu == "Pola Data":
    st.markdown('<div class="section-title">Pola Data Jumlah Sampah</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Pilih grafik yang ingin dilihat. Setiap grafik diberi penjelasan singkat agar mudah dibaca.</div>',
        unsafe_allow_html=True
    )

    pilih_col, kosong_col = st.columns([0.42, 0.58], gap="large")

    with pilih_col:
        pilihan_grafik = st.selectbox(
            "Pilih grafik",
            [
                "Perubahan jumlah sampah dari waktu ke waktu",
                "Tren jangka panjang",
                "Perbandingan sebaran per tahun",
                "Rata-rata jumlah sampah per tahun",
                "Rata-rata jumlah sampah per bulan",
                "Peta warna tahun dan bulan",
                "Sebaran nilai jumlah sampah",
                "Pemisahan pola data"
            ]
        )

    left, right = st.columns([1.15, 1], gap="large")

    if pilihan_grafik == "Perubahan jumlah sampah dari waktu ke waktu":
        with left:
            st.markdown('<div class="small-title">Perubahan Jumlah Sampah 2017–2024</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.6, 2.6), facecolor=plot_bg)
            ax.plot(ts, marker="o", linewidth=1.1, markersize=3, label="Jumlah Sampah")
            style_plot(ax, "Jumlah Sampah Kota Bandung")
            ax.set_xlabel("Tahun", fontsize=8)
            ax.set_ylabel("Ton", fontsize=8)
            ax.legend(fontsize=7)
            st.pyplot(fig, use_container_width=False)

        with right:
            bullet_card(
                "Cara membaca grafik",
                [
                    "Garis naik berarti jumlah sampah meningkat.",
                    "Garis turun berarti jumlah sampah menurun.",
                    "Grafik ini menunjukkan perubahan dari bulan ke bulan."
                ]
            )
            bullet_card(
                "Temuan utama",
                [
                    "Jumlah sampah meningkat dari 2017 sampai sekitar 2019–2020.",
                    "Setelah 2020, pola mulai menurun.",
                    "Penurunan tajam terlihat pada periode 2023–2024."
                ]
            )

    elif pilihan_grafik == "Tren jangka panjang":
        rolling_mean = ts.rolling(window=12).mean()

        with left:
            st.markdown('<div class="small-title">Tren Jangka Panjang</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.6, 2.6), facecolor=plot_bg)
            ax.plot(ts, marker="o", linewidth=0.9, markersize=2.5, label="Data Asli")
            ax.plot(rolling_mean, linewidth=2, label="Rata-rata Bergerak 12 Bulan")
            style_plot(ax, "Tren Jumlah Sampah")
            ax.set_xlabel("Tahun", fontsize=8)
            ax.set_ylabel("Ton", fontsize=8)
            ax.legend(fontsize=7)
            st.pyplot(fig, use_container_width=False)

        with right:
            bullet_card(
                "Apa maksud grafik ini?",
                [
                    "Grafik ini memperhalus data bulanan.",
                    "Tujuannya agar arah jangka panjang lebih mudah terlihat.",
                    "Garis rata-rata bergerak membantu melihat pola besar."
                ]
            )
            bullet_card(
                "Temuan utama",
                [
                    "Tren meningkat sampai sekitar 2020.",
                    "Setelah 2021, tren mulai turun.",
                    "Penurunan semakin jelas pada 2023–2024."
                ]
            )

    elif pilihan_grafik == "Perbandingan sebaran per tahun":
        df_box = df.copy()
        df_box["tahun_plot"] = df_box.index.year

        with left:
            st.markdown('<div class="small-title">Sebaran Jumlah Sampah per Tahun</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.6, 2.6), facecolor=plot_bg)
            sns.boxplot(x="tahun_plot", y="jumlah_sampah", data=df_box, ax=ax)
            style_plot(ax, "Sebaran per Tahun")
            ax.set_xlabel("Tahun", fontsize=8)
            ax.set_ylabel("Ton", fontsize=8)
            st.pyplot(fig, use_container_width=False)

        with right:
            bullet_card(
                "Apa yang dilihat?",
                [
                    "Grafik ini membandingkan tinggi-rendah jumlah sampah setiap tahun.",
                    "Bagian tengah kotak menunjukkan nilai tengah.",
                    "Kotak yang lebih besar menunjukkan variasi yang lebih besar."
                ]
            )
            bullet_card(
                "Temuan utama",
                [
                    "Nilai tengah tertinggi terlihat pada 2019–2020.",
                    "Tahun 2023 memiliki variasi yang besar.",
                    "Tahun 2024 lebih rendah dibanding tahun-tahun sebelumnya."
                ]
            )

    elif pilihan_grafik == "Rata-rata jumlah sampah per tahun":
        rata_tahun = df.groupby(df.index.year)["jumlah_sampah"].mean().reset_index()
        rata_tahun.columns = ["Tahun", "Rata-rata Jumlah Sampah"]

        with left:
            st.markdown('<div class="small-title">Rata-rata per Tahun</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.6, 2.6), facecolor=plot_bg)
            ax.bar(rata_tahun["Tahun"], rata_tahun["Rata-rata Jumlah Sampah"])
            style_plot(ax, "Rata-rata Jumlah Sampah per Tahun")
            ax.set_xlabel("Tahun", fontsize=8)
            ax.set_ylabel("Ton", fontsize=8)
            st.pyplot(fig, use_container_width=False)

        with right:
            st.markdown('<div class="small-title">Tabel Rata-rata</div>', unsafe_allow_html=True)
            show_table(rata_tahun)
            bullet_card(
                "Temuan utama",
                [
                    "Rata-rata tertinggi terjadi pada 2019 dan 2020.",
                    "Setelah 2020, rata-rata mulai menurun.",
                    "Tahun 2024 menjadi yang paling rendah."
                ]
            )

    elif pilihan_grafik == "Rata-rata jumlah sampah per bulan":
        urutan_bulan = [
            "JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI",
            "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER"
        ]

        rata_bulan_series = df.groupby("bulan")["jumlah_sampah"].mean().reindex(urutan_bulan)
        rata_bulan = rata_bulan_series.reset_index()
        rata_bulan.columns = ["Bulan", "Rata-rata Jumlah Sampah"]

        with left:
            st.markdown('<div class="small-title">Rata-rata per Bulan</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.6, 2.6), facecolor=plot_bg)
            ax.bar(rata_bulan["Bulan"], rata_bulan["Rata-rata Jumlah Sampah"])
            style_plot(ax, "Rata-rata Jumlah Sampah per Bulan")
            ax.set_xlabel("Bulan", fontsize=8)
            ax.set_ylabel("Ton", fontsize=8)
            ax.tick_params(axis="x", labelsize=6, rotation=45)
            st.pyplot(fig, use_container_width=False)

        with right:
            st.markdown('<div class="small-title">Tabel Rata-rata</div>', unsafe_allow_html=True)
            show_table(rata_bulan)
            bullet_card(
                "Temuan utama",
                [
                    "Rata-rata bulanan tidak berbeda terlalu jauh.",
                    "Bulan Juli memiliki rata-rata tertinggi.",
                    "Bulan Februari memiliki rata-rata terendah.",
                    "Pola bulanan ada, tetapi tidak terlalu kuat."
                ]
            )

    elif pilihan_grafik == "Peta warna tahun dan bulan":
        urutan_bulan = [
            "JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI",
            "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER", "DESEMBER"
        ]

        df_heatmap = df.copy()
        df_heatmap["tahun_plot"] = df_heatmap.index.year
        df_heatmap["bulan_plot"] = df_heatmap["bulan"]

        pivot_heatmap = df_heatmap.pivot_table(
            values="jumlah_sampah",
            index="tahun_plot",
            columns="bulan_plot",
            aggfunc="mean"
        )

        pivot_heatmap = pivot_heatmap[urutan_bulan]

        with left:
            st.markdown('<div class="small-title">Peta Warna Tahun-Bulan</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.2, 2.9), facecolor=plot_bg)
            sns.heatmap(
                pivot_heatmap,
                annot=True,
                fmt=".0f",
                cmap="YlOrBr",
                ax=ax,
                annot_kws={"size": 5},
                cbar_kws={"shrink": 0.7}
            )
            ax.set_title("Jumlah Sampah per Tahun dan Bulan", fontsize=10, fontweight="bold")
            ax.set_xlabel("Bulan", fontsize=8)
            ax.set_ylabel("Tahun", fontsize=8)
            ax.tick_params(axis="x", labelsize=5.5, rotation=45)
            ax.tick_params(axis="y", labelsize=7)
            st.pyplot(fig, use_container_width=False)

        with right:
            bullet_card(
                "Cara membaca warna",
                [
                    "Warna yang lebih pekat menunjukkan jumlah sampah lebih tinggi.",
                    "Warna yang lebih terang menunjukkan jumlah sampah lebih rendah."
                ]
            )
            bullet_card(
                "Temuan utama",
                [
                    "Periode 2019–2020 relatif tinggi.",
                    "Tahun 2023 mulai menunjukkan penurunan.",
                    "Tahun 2024 terlihat lebih rendah hampir di semua bulan."
                ]
            )

    elif pilihan_grafik == "Sebaran nilai jumlah sampah":
        with left:
            st.markdown('<div class="small-title">Sebaran Nilai Jumlah Sampah</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.6, 2.6), facecolor=plot_bg)
            sns.histplot(df["jumlah_sampah"], kde=True, ax=ax)
            style_plot(ax, "Sebaran Jumlah Sampah")
            ax.set_xlabel("Jumlah Sampah (Ton)", fontsize=8)
            ax.set_ylabel("Frekuensi", fontsize=8)
            st.pyplot(fig, use_container_width=False)

        with right:
            bullet_card(
                "Apa yang dilihat?",
                [
                    "Grafik ini menunjukkan nilai jumlah sampah yang paling sering muncul.",
                    "Jika sebaran tidak seimbang, berarti ada periode yang berbeda dari pola umum."
                ]
            )
            bullet_card(
                "Temuan utama",
                [
                    "Sebagian besar data berada pada rentang sekitar 36.000–42.000 ton.",
                    "Terdapat beberapa nilai rendah.",
                    "Nilai rendah berkaitan dengan penurunan besar pada periode tertentu."
                ]
            )

    elif pilihan_grafik == "Pemisahan pola data":
        with left:
            st.markdown('<div class="small-title">Pemisahan Pola Data</div>', unsafe_allow_html=True)
            decomp = seasonal_decompose(ts, model="additive", period=12)
            fig = decomp.plot()
            fig.set_size_inches(5.7, 3.7)
            for ax in fig.axes:
                ax.tick_params(axis="both", labelsize=6)
                ax.set_xlabel("")
            st.pyplot(fig, use_container_width=False)

        with right:
            bullet_card(
                "Apa maksud grafik ini?",
                [
                    "Grafik ini memisahkan data menjadi arah utama, pola bulanan, dan sisa naik-turun acak.",
                    "Tujuannya agar pola data lebih mudah dipahami."
                ]
            )
            bullet_card(
                "Temuan utama",
                [
                    "Arah utama naik sampai sekitar 2020.",
                    "Setelah itu, arah utama menurun sampai 2024.",
                    "Pola bulanan terlihat, tetapi tidak terlalu dominan.",
                    "Masih ada fluktuasi acak dan beberapa lonjakan."
                ]
            )

# ============================================================
# PREDIKSI SAMPAH
# ============================================================

elif menu == "Prediksi Sampah":
    st.markdown('<div class="section-title">Prediksi Jumlah Sampah</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Pilih cara prediksi, lalu geser jumlah bulan ke depan yang ingin diperkirakan.</div>',
        unsafe_allow_html=True
    )

    left_input, right_input = st.columns([1, 1], gap="large")

    with left_input:
        selected_model = st.selectbox(
            "Pilih cara prediksi ♻️",
            [
                "ARIMA(0,1,1)",
                "SARIMA(1,2,2)(0,1,1,12)",
                "Naive Forecast",
                "Seasonal Naive Forecast"
            ]
        )

    with right_input:
        forecast_steps = st.slider(
            "Geser jumlah bulan ke depan",
            min_value=1,
            max_value=36,
            value=12,
            step=1
        )

    st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Arti Model Prediksi</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4, gap="large")

    with m1:
        model_card(
            "📈",
            "ARIMA",
            "Model time series yang melihat pola data masa lalu, tren perubahan, dan error sebelumnya. Cocok untuk data yang tidak terlalu kuat pola musimannya."
        )

    with m2:
        model_card(
            "🔁",
            "SARIMA",
            "Pengembangan ARIMA yang menambahkan pola musiman. Model ini mencoba menangkap pola berulang, misalnya pola tahunan pada data bulanan."
        )

    with m3:
        model_card(
            "📌",
            "Naive Forecast",
            "Model pembanding sederhana. Prediksi bulan depan dianggap sama dengan nilai terakhir pada data historis."
        )

    with m4:
        model_card(
            "🗓️",
            "Seasonal Naive",
            "Model pembanding musiman. Prediksi bulan tertentu mengikuti nilai pada bulan yang sama di periode sebelumnya."
        )

    future_index = pd.date_range(
        start=ts.index.max() + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="MS"
    )

    if selected_model == "ARIMA(0,1,1)":
        final_model = ARIMA(ts, order=(0, 1, 1))
        final_fit = final_model.fit()
        future_forecast = final_fit.forecast(steps=forecast_steps)

    elif selected_model == "SARIMA(1,2,2)(0,1,1,12)":
        final_model = SARIMAX(
            ts,
            order=(1, 2, 2),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        final_fit = final_model.fit(disp=False)
        future_forecast = final_fit.forecast(steps=forecast_steps)

    elif selected_model == "Naive Forecast":
        future_forecast = pd.Series([ts.iloc[-1]] * forecast_steps)

    else:
        seasonal_values = np.resize(ts.iloc[-12:].values, forecast_steps)
        future_forecast = pd.Series(seasonal_values)

    future_forecast = pd.Series(future_forecast.values, index=future_index)

    forecast_output = pd.DataFrame({
        "Periode": [format_periode(date) for date in future_forecast.index],
        "Model": selected_model,
        "Prediksi Jumlah Sampah (Ton)": future_forecast.values
    })

    total_prediksi = forecast_output["Prediksi Jumlah Sampah (Ton)"].sum()
    rata_prediksi = forecast_output["Prediksi Jumlah Sampah (Ton)"].mean()

    start_period = format_periode(future_forecast.index.min())
    end_period = format_periode(future_forecast.index.max())

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        kpi_card("Periode Prediksi", f"{start_period} - {end_period}", small=True)

    with col2:
        kpi_card("Total Prediksi", f"{total_prediksi:,.2f} ton", small=True)

    with col3:
        kpi_card("Rata-rata per Bulan", f"{rata_prediksi:,.2f} ton", small=True)

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="small-title">Grafik Prediksi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.8, 2.8), facecolor=plot_bg)
        ax.plot(ts.tail(24), marker="o", linewidth=1.1, markersize=3, label="Data Historis")
        ax.plot(
            future_forecast,
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=f"Prediksi {selected_model}"
        )
        style_plot(ax, f"Prediksi {forecast_steps} Bulan ke Depan")
        ax.set_xlabel("Periode", fontsize=8)
        ax.set_ylabel("Ton", fontsize=8)
        ax.legend(fontsize=7)
        st.pyplot(fig, use_container_width=False)

    with right:
        st.markdown('<div class="small-title">Tabel Prediksi</div>', unsafe_allow_html=True)
        show_table(forecast_output)

    bullet_card(
        "Catatan pembacaan hasil",
        [
            "Prediksi dibuat dari data terakhir yang tersedia, yaitu Desember 2024.",
            "Input prediksi menggunakan jumlah bulan karena data asli berbentuk bulanan.",
            "Semakin jauh jumlah bulan yang diprediksi, semakin besar kemungkinan hasil meleset."
        ]
    )

# ============================================================
# KETEPATAN PREDIKSI
# ============================================================

elif menu == "Ketepatan Prediksi":
    st.markdown('<div class="section-title">Ketepatan Prediksi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Bagian ini membandingkan beberapa cara prediksi menggunakan data tahun 2024 sebagai pembanding.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="small-title">Tabel Perbandingan</div>', unsafe_allow_html=True)
        show_table(eval_df)

    with right:
        st.markdown('<div class="small-title">Perbandingan Kesalahan Persentase</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.3, 2.6), facecolor=plot_bg)
        ax.bar(eval_df["Model"], eval_df["MAPE (%)"])
        style_plot(ax, "Perbandingan MAPE")
        ax.set_xlabel("Model", fontsize=8)
        ax.set_ylabel("MAPE (%)", fontsize=8)
        ax.tick_params(axis="x", labelsize=6, rotation=20)
        st.pyplot(fig, use_container_width=False)

    bullet_card(
        "Cara membaca hasil",
        [
            "Semakin kecil MAE, RMSE, dan MAPE, semakin baik hasil prediksi.",
            "MAPE menunjukkan rata-rata kesalahan dalam bentuk persen.",
            "R² negatif berarti model belum mampu menjelaskan variasi data uji dengan baik."
        ]
    )

    bullet_card(
        "Temuan utama",
        [
            "ARIMA memiliki nilai kesalahan paling kecil.",
            "Naive Forecast berada dekat dengan ARIMA.",
            "SARIMA lebih dinamis secara visual, tetapi kesalahannya lebih besar.",
            "Seasonal Naive Forecast memiliki kesalahan paling besar."
        ]
    )

# ============================================================
# RINGKASAN
# ============================================================

elif menu == "Ringkasan":
    st.markdown('<div class="section-title">Ringkasan Hasil</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Ringkasan akhir dari data, pola, prediksi, dan hasil perbandingan model.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="large")

    with left:
        bullet_card(
            "Pembahasan",
            [
                f"Data berasal dari Provinsi <b>{provinsi}</b>.",
                f"Wilayah yang dianalisis adalah <b>{kota}</b>.",
                f"Satuan jumlah sampah adalah <b>{satuan}</b>.",
                "Jumlah sampah tidak sepenuhnya stabil dari tahun ke tahun.",
                "Pola naik terlihat sampai sekitar 2020, lalu mulai menurun.",
                "Penurunan tajam pada 2023–2024 membuat prediksi menjadi lebih sulit.",
                "ARIMA menghasilkan prediksi yang cenderung datar.",
                "SARIMA lebih dinamis, tetapi kesalahannya lebih besar."
            ]
        )

    with right:
        bullet_card(
            "Kesimpulan",
            [
                f"Data yang digunakan adalah data jumlah sampah bulanan <b>{kota}</b>.",
                f"Periode data: <b>{format_periode(ts.index.min())}</b> sampai <b>{format_periode(ts.index.max())}</b>.",
                "Model terbaik berdasarkan MAE, RMSE, dan MAPE adalah <b>ARIMA(0,1,1)</b>.",
                "ARIMA memperoleh MAPE sekitar <b>6.067%</b>.",
                "Semua model memiliki R² negatif.",
                "Model sederhana belum sepenuhnya mampu menangkap perubahan tajam pada data tahun 2024."
            ]
        )

    card(
        "Catatan akhir",
        """
        Dashboard ini dapat digunakan sebagai alat bantu awal untuk membaca pola jumlah sampah dan memperkirakan kebutuhan penanganan sampah.
        Namun, hasil prediksi tetap perlu dipertimbangkan bersama faktor nyata lain, seperti kebijakan pengelolaan sampah,
        aktivitas masyarakat, perubahan sistem pencatatan, dan kondisi khusus pada tahun tertentu.
        """
    )
