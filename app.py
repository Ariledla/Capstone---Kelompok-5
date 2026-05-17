import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Simulasi Sampah Kota Bandung",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="auto"
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

MENU_OPTIONS = ["Simulasi Pengelolaan", "Ringkasan Data & Model"]

# Konstanta simulasi operasional
# Densitas digunakan untuk mengonversi prediksi ton menjadi volume m³.
# Kapasitas truk compactor mengacu pada ukuran 6 m³ dan 12 m³; default dipakai 12 m³.
DENSITAS_SAMPAH_KG_PER_M3 = 227.297


# ============================================================
# FORMATTER
# ============================================================

def format_periode(date_value):
    return f"{BULAN_INDO[date_value.month]} {date_value.year}"


def format_rupiah(value):
    return "Rp{:,.0f}".format(value).replace(",", ".")


def format_angka(value):
    return "{:,.2f}".format(value).replace(",", "X").replace(".", ",").replace("X", ".")


def format_integer(value):
    return "{:,.0f}".format(value).replace(",", ".")


def evaluate_model(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    r2 = r2_score(actual, forecast)
    return mae, rmse, mape, r2


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df_raw = pd.read_excel(FILE_NAME)

    df = df_raw.copy()
    df["bulan_num"] = df["bulan"].astype(str).str.upper().map(BULAN_MAP)

    df["tanggal"] = pd.to_datetime({
        "year": df["tahun"],
        "month": df["bulan_num"],
        "day": 1
    })

    df = df.sort_values("tanggal")
    df = df.set_index("tanggal")

    ts = df["jumlah_sampah"].asfreq("MS")

    return df_raw, df, ts


@st.cache_data
def make_sarima_forecast(ts, forecast_steps):
    model = SARIMAX(
        ts,
        order=(1, 2, 2),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)

    future_index = pd.date_range(
        start=ts.index.max() + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq="MS"
    )

    forecast = fit.forecast(steps=forecast_steps)
    forecast = pd.Series(forecast.values, index=future_index)

    return forecast


@st.cache_data
def evaluate_sarima(ts):
    train = ts.iloc[:-12]
    test = ts.iloc[-12:]

    model = SARIMAX(
        train,
        order=(1, 2, 2),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=len(test))
    forecast.index = test.index

    mae, rmse, mape, r2 = evaluate_model(test, forecast)

    eval_df = pd.DataFrame({
        "Model": ["SARIMA(1,2,2)(0,1,1,12)"],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE (%)": [mape],
        "R²": [r2]
    })

    comparison_df = pd.DataFrame({
        "Periode": [format_periode(date) for date in test.index],
        "Aktual": test.values,
        "Prediksi": forecast.values
    })

    return eval_df, comparison_df, test, forecast


# ============================================================
# OPERASIONAL
# ============================================================

def build_simulation_table(forecast, biaya_per_ton, kapasitas_truk_compactor_m3, hari_operasional_angkut_per_minggu):
    output = pd.DataFrame({
        "Tanggal": forecast.index,
        "Periode": [format_periode(date) for date in forecast.index],
        "Prediksi Sampah (Ton)": forecast.values
    })

    output["Jumlah Hari"] = output["Tanggal"].dt.days_in_month
    output["Estimasi Anggaran"] = output["Prediksi Sampah (Ton)"] * biaya_per_ton

    output["Estimasi Hari Operasional Angkut"] = np.ceil(
        output["Jumlah Hari"] * (hari_operasional_angkut_per_minggu / 7)
    )

    output["Estimasi Volume Sampah (m³)"] = (
        output["Prediksi Sampah (Ton)"] * 1000 / DENSITAS_SAMPAH_KG_PER_M3
    )

    output["Estimasi Kebutuhan Muatan Truk"] = np.ceil(
        output["Estimasi Volume Sampah (m³)"] / kapasitas_truk_compactor_m3
    )

    output["Muatan Truk per Hari Angkut"] = np.ceil(
        output["Estimasi Kebutuhan Muatan Truk"] / output["Estimasi Hari Operasional Angkut"]
    )

    return output


def prepare_display_table(output):
    display = output.copy()

    display["Prediksi Sampah (Ton)"] = display["Prediksi Sampah (Ton)"].apply(format_angka)
    display["Estimasi Anggaran"] = display["Estimasi Anggaran"].apply(format_rupiah)
    display["Estimasi Volume Sampah (m³)"] = display["Estimasi Volume Sampah (m³)"].apply(format_angka)
    display["Estimasi Hari Operasional Angkut"] = display["Estimasi Hari Operasional Angkut"].astype(int).apply(format_integer)
    display["Estimasi Kebutuhan Muatan Truk"] = display["Estimasi Kebutuhan Muatan Truk"].astype(int).apply(format_integer)
    display["Muatan Truk per Hari Angkut"] = display["Muatan Truk per Hari Angkut"].astype(int).apply(format_integer)

    display = display[
        [
            "Periode",
            "Prediksi Sampah (Ton)",
            "Estimasi Anggaran",
            "Estimasi Volume Sampah (m³)",
            "Estimasi Hari Operasional Angkut",
            "Estimasi Kebutuhan Muatan Truk",
            "Muatan Truk per Hari Angkut"
        ]
    ]

    return display


def prepare_eval_display(eval_df):
    display = eval_df.copy()
    display["MAE"] = display["MAE"].apply(format_angka)
    display["RMSE"] = display["RMSE"].apply(format_angka)
    display["MAPE (%)"] = display["MAPE (%)"].apply(lambda x: f"{x:.2f}%")
    display["R²"] = display["R²"].apply(lambda x: f"{x:.4f}")
    return display


def prepare_comparison_display(comparison_df):
    display = comparison_df.copy()
    display["Aktual"] = display["Aktual"].apply(format_angka)
    display["Prediksi"] = display["Prediksi"].apply(format_angka)
    return display


# ============================================================
# SESSION STATE
# ============================================================

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Gelap"

if "active_menu" not in st.session_state:
    st.session_state.active_menu = "Simulasi Pengelolaan"

if st.session_state.active_menu == "Data Singkat":
    st.session_state.active_menu = "Ringkasan Data & Model"

if st.session_state.active_menu not in MENU_OPTIONS:
    st.session_state.active_menu = "Simulasi Pengelolaan"


def set_theme(mode):
    st.session_state.theme_mode = mode


# ============================================================
# THEME
# ============================================================

def apply_theme(mode):
    if mode == "Terang":
        cfg = {
            "bg": "#EFE9DC",
            "card": "#FFFDF7",
            "card2": "#F5F1E8",
            "text": "#172018",
            "muted": "#3F4B3E",
            "border": "#CFC7B8",
            "accent": "#2E6F4F",
            "accent2": "#B88A3D",
            "accent_soft": "rgba(46, 111, 79, 0.11)",
            "accent_hover": "#2E6F4F",
            "shadow": "rgba(31, 41, 51, 0.06)",
            "hero": "linear-gradient(135deg, #2E6F4F 0%, #6B9A61 56%, #B88A3D 100%)",
            "sidebar_bg": """
                radial-gradient(circle at 14% 8%, rgba(107, 154, 97, 0.22), transparent 24%),
                radial-gradient(circle at 90% 23%, rgba(184, 138, 61, 0.18), transparent 25%),
                linear-gradient(180deg, #EFECDD 0%, #E7E3D3 48%, #EDE3CF 100%)
            """,
            "sidebar_visual": "linear-gradient(135deg, #2E6F4F 0%, #527D52 60%, #8C6B31 100%)",
            "chart_bg": "#FFFDF7",
            "chart_grid": "rgba(23, 32, 24, 0.12)",
            "chart_font": "#172018",
            "chart_axis": "#172018",
            "chart_hist": "#2E6F4F",
            "chart_pred": "#B88A3D",
            "chart_hist_fill": "rgba(46, 111, 79, 0.14)",
            "chart_pred_fill": "rgba(184, 138, 61, 0.16)",
            "chart_divider": "rgba(23, 32, 24, 0.42)",
            "chart_legend_bg": "rgba(255,253,247,0.96)",
            "chart_legend_border": "rgba(63,75,62,0.25)",
            "annotation_bg": "rgba(255,253,247,0.97)",
            "annotation_border": "rgba(63,75,62,0.26)",
            "input_bg": "#FFFDF7",
            "input_btn": "#E6DECE",
            "input_btn_hover": "#2E6F4F",
        }
    else:
        cfg = {
            "bg": "#151A17",
            "card": "#222A24",
            "card2": "#263029",
            "text": "#F5F7F2",
            "muted": "#D8E0D4",
            "border": "#3D4A40",
            "accent": "#8BCB88",
            "accent2": "#E2B15D",
            "accent_soft": "rgba(139, 203, 136, 0.12)",
            "accent_hover": "#2F7D52",
            "shadow": "rgba(0, 0, 0, 0.14)",
            "hero": "linear-gradient(135deg, #1F4D36 0%, #4F8B59 55%, #B78335 100%)",
            "sidebar_bg": """
                radial-gradient(circle at 8% 12%, rgba(139, 203, 136, 0.28), transparent 16%),
                radial-gradient(circle at 92% 28%, rgba(47, 111, 78, 0.18), transparent 17%),
                radial-gradient(circle at 18% 88%, rgba(226, 177, 93, 0.18), transparent 18%),
                linear-gradient(180deg, #1E241F 0%, #1B241C 60%, #222819 100%)
            """,
            "sidebar_visual": "linear-gradient(135deg, #26382C 0%, #2F6F4E 65%, #6A4A1E 100%)",
            "chart_bg": "#111915",
            "chart_grid": "rgba(255,255,255,0.08)",
            "chart_font": "#F2F5F1",
            "chart_axis": "#F2F5F1",
            "chart_hist": "#67F0C1",
            "chart_pred": "#F1BE54",
            "chart_hist_fill": "rgba(103,240,193,0.18)",
            "chart_pred_fill": "rgba(241,190,84,0.18)",
            "chart_divider": "rgba(255,255,255,0.35)",
            "chart_legend_bg": "rgba(20,28,24,0.90)",
            "chart_legend_border": "rgba(255,255,255,0.15)",
            "annotation_bg": "rgba(20,28,24,0.92)",
            "annotation_border": "rgba(255,255,255,0.22)",
            "input_bg": "#222A24",
            "input_btn": "#2B2D3A",
            "input_btn_hover": "#2F7D52",
        }

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 2% 4%, rgba(139, 203, 136, 0.06), transparent 20%),
                {cfg["bg"]} !important;
            color: {cfg["text"]} !important;
        }}

        header, footer, #MainMenu {{
            visibility: hidden;
        }}

        .mobile-kpi-summary {{
            display: none;
        }}

        h1, h2, h3, h4, h5, h6, p, label, span, div {{
            color: inherit;
        }}

        .block-container {{
            max-width: 1420px !important;
            padding-top: 4rem !important;
            padding-left: 0.95rem !important;
            padding-right: 0.95rem !important;
            padding-bottom: 1.5rem !important;
        }}

        [data-testid="stSidebar"] {{
            background: {cfg["sidebar_bg"]} !important;
            border-right: 1px solid {cfg["border"]};
            width: 304px !important;
            min-width: 304px !important;
        }}

        [data-testid="stSidebarContent"] {{
            width: 304px !important;
            padding-top: 0rem !important;
        }}

        section[data-testid="stSidebar"] > div {{
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
            color: {cfg["text"]} !important;
        }}

        .theme-label {{
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
            color: {cfg["text"]} !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
            gap: 0.55rem !important;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            width: 100% !important;
            height: 42px !important;
            border-radius: 14px !important;
            border: 1px solid {cfg["border"]} !important;
            background: {cfg["card"]} !important;
            color: {cfg["text"]} !important;
            font-weight: 850 !important;
            font-size: 13px !important;
            box-shadow: 0 8px 18px rgba(0,0,0,0.08) !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: {cfg["accent_hover"]} !important;
            color: white !important;
            border-color: {cfg["accent_hover"]} !important;
            transform: translateY(-1px);
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:nth-child(1) .stButton > button {{
            background: {cfg["accent_hover"] if mode == "Terang" else cfg["card"]} !important;
            color: {"white" if mode == "Terang" else cfg["text"]} !important;
            border-color: {cfg["accent_hover"] if mode == "Terang" else cfg["border"]} !important;
        }}

        [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {{
            background: {cfg["accent_hover"] if mode == "Gelap" else cfg["card"]} !important;
            color: {"white" if mode == "Gelap" else cfg["text"]} !important;
            border-color: {cfg["accent_hover"] if mode == "Gelap" else cfg["border"]} !important;
        }}

        .sidebar-visual {{
            background: {cfg["sidebar_visual"]};
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

        .hero {{
            background: {cfg["hero"]};
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
            line-height: 1.6;
        }}

        .section-title {{
            font-size: 25px;
            font-weight: 850;
            color: {cfg["text"]} !important;
            margin-bottom: 7px;
        }}

        .section-desc {{
            color: {cfg["muted"]} !important;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 22px;
            line-height: 1.65;
        }}

        .small-title {{
            font-size: 16.5px;
            font-weight: 800;
            color: {cfg["text"]} !important;
            margin-bottom: 12px;
        }}

        .info-card {{
            background: {cfg["card"]};
            border: 1px solid {cfg["border"]};
            border-radius: 20px;
            padding: 19px 22px;
            margin-bottom: 20px;
            min-height: auto;
            box-shadow: 0 8px 26px {cfg["shadow"]};
        }}

        .text-muted {{
            color: {cfg["muted"]} !important;
            font-size: 14px;
            font-weight: 500;
            line-height: 1.7;
        }}

        .text-muted li {{
            color: {cfg["muted"]} !important;
            margin-bottom: 8px;
        }}

        .kpi-card {{
            background: {cfg["card"]};
            border: 1px solid {cfg["border"]};
            border-radius: 20px;
            padding: 17px 18px;
            min-height: 104px;
            box-shadow: 0 8px 24px {cfg["shadow"]};
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: flex-start;
            gap: 7px;
            margin-bottom: 22px;
            box-sizing: border-box;
            overflow: hidden;
        }}

        .kpi-label {{
            color: {cfg["muted"]} !important;
            font-size: 13px;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: -0.1px;
            margin: 0;
        }}

        .kpi-value {{
            color: {cfg["text"]} !important;
            font-size: 22px;
            font-weight: 900;
            line-height: 1.16;
            letter-spacing: -0.4px;
            word-break: normal;
            overflow-wrap: anywhere;
            margin: 0;
        }}

        .kpi-note {{
            color: {cfg["muted"]} !important;
            font-size: 11.5px;
            font-weight: 650;
            line-height: 1.35;
            margin: 0;
            opacity: 0.95;
        }}

        div[data-baseweb="select"] > div {{
            background: {cfg["card"]} !important;
            border: 1px solid {cfg["border"]} !important;
            border-radius: 14px !important;
            color: {cfg["text"]} !important;
            min-height: 42px !important;
            cursor: pointer !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="select"] span {{
            color: {cfg["text"]} !important;
            font-weight: 650 !important;
        }}

        div[data-baseweb="select"] svg {{
            color: {cfg["accent"]} !important;
            fill: {cfg["accent"]} !important;
            opacity: 1 !important;
        }}

        [data-testid="stNumberInput"] {{
            border-radius: 14px !important;
        }}

        [data-testid="stNumberInput"] > div {{
            border: 1px solid {cfg["border"]} !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            box-shadow: none !important;
            outline: none !important;
            background: {cfg["input_bg"]} !important;
        }}

        [data-testid="stNumberInput"] div[data-baseweb="input"] {{
            background: {cfg["input_bg"]} !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] input {{
            background: {cfg["input_bg"]} !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            color: {cfg["text"]} !important;
            min-height: 42px !important;
            font-weight: 650 !important;
        }}

        [data-testid="stNumberInput"] input:focus {{
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] button {{
            background: {cfg["input_btn"]} !important;
            color: {cfg["text"]} !important;
            border: none !important;
            border-left: 1px solid {cfg["border"]} !important;
            min-height: 42px !important;
            height: 42px !important;
            margin-top: -3px !important;
            padding-bottom: 3px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            outline: none !important;
        }}

        [data-testid="stNumberInput"] button:hover {{
            background: {cfg["input_btn_hover"]} !important;
            color: white !important;
        }}

        .stSlider label, .stSelectbox label, .stRadio label, .stNumberInput label {{
            color: {cfg["text"]} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] span {{
            color: {cfg["text"]} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 12px !important;
            padding: 4px 8px !important;
            transition: all 0.16s ease-in-out !important;
            cursor: pointer !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: {cfg["accent_soft"]} !important;
            transform: translateX(2px);
        }}

        .custom-table-wrapper {{
            width: 100%;
            overflow-x: auto;
            border: none !important;
            border-radius: 18px;
            background: transparent !important;
            margin-bottom: 24px;
            box-shadow: none !important;
            overflow: visible;
        }}

        table.custom-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: transparent !important;
            color: {cfg["text"]} !important;
            font-size: 13px;
            line-height: 1.2;
            margin: 0 !important;
            border: none !important;
            border-radius: 18px;
            overflow: hidden;
        }}

        table.custom-table thead tr th {{
            background: {cfg["card2"]} !important;
            color: {cfg["text"]} !important;
            font-weight: 850;
            padding: 11px 14px;
            height: 42px;
            border-top: 1px solid {cfg["border"]};
            border-bottom: 1px solid {cfg["border"]};
            border-right: 1px solid {cfg["border"]};
            text-align: left;
            white-space: nowrap;
            vertical-align: middle;
        }}

        table.custom-table thead tr th:first-child {{
            border-left: 1px solid {cfg["border"]};
            border-top-left-radius: 18px;
        }}

        table.custom-table thead tr th:last-child {{
            border-top-right-radius: 18px;
        }}

        table.custom-table tbody tr td,
        table.custom-table tbody tr th {{
            background: {cfg["card"]} !important;
            color: {cfg["text"]} !important;
            padding: 11px 14px;
            height: 42px;
            border-right: 1px solid {cfg["border"]};
            border-bottom: 1px solid {cfg["border"]};
            font-weight: 650;
            white-space: nowrap;
            vertical-align: middle;
        }}

        table.custom-table tbody tr td:first-child,
        table.custom-table tbody tr th:first-child {{
            border-left: 1px solid {cfg["border"]};
        }}

        table.custom-table tbody tr:last-child td,
        table.custom-table tbody tr:last-child th {{
            height: 42px !important;
            padding-top: 11px !important;
            padding-bottom: 11px !important;
        }}

        table.custom-table tbody tr:last-child td:first-child,
        table.custom-table tbody tr:last-child th:first-child {{
            border-bottom-left-radius: 18px;
        }}

        table.custom-table tbody tr:last-child td:last-child,
        table.custom-table tbody tr:last-child th:last-child {{
            border-bottom-right-radius: 18px;
        }}

        .stPlotlyChart {{
            background: {cfg["chart_bg"]} !important;
            border: 1.5px solid {cfg["border"]} !important;
            border-radius: 20px !important;
            padding: 4px 4px 4px 4px !important;
            margin-bottom: 22px !important;
            box-shadow: 0 8px 26px {cfg["shadow"]} !important;
            overflow: hidden !important;
        }}

        .stPlotlyChart > div {{
            border-radius: 16px !important;
            overflow: hidden !important;
        }}


        .desktop-chart {{
            display: block;
        }}

        .mobile-chart {{
            display: none;
        }}

        @media screen and (max-width: 900px) {{
            .block-container {{
                padding-top: 0.7rem !important;
                padding-left: 0.62rem !important;
                padding-right: 0.62rem !important;
                padding-bottom: 1rem !important;
                max-width: 100% !important;
            }}

            header,
            [data-testid="stHeader"] {{
                visibility: visible !important;
                background: transparent !important;
                height: 46px !important;
                min-height: 46px !important;
            }}

            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"] {{
                display: none !important;
            }}

            [data-testid="stSidebarCollapsedControl"],
            [data-testid="collapsedControl"],
            [data-testid="stSidebarNav"],
            button[title="Open sidebar"],
            button[aria-label="Open sidebar"],
            button[data-testid="stBaseButton-headerNoPadding"],
            button[data-testid="baseButton-headerNoPadding"],
            button[kind="headerNoPadding"] {{
                display: flex !important;
                visibility: visible !important;
                opacity: 1 !important;
                position: fixed !important;
                top: 10px !important;
                left: 10px !important;
                z-index: 999999 !important;
                background: {cfg["card"]} !important;
                border: 1px solid {cfg["border"]} !important;
                border-radius: 13px !important;
                box-shadow: 0 8px 22px {cfg["shadow"]} !important;
                width: 42px !important;
                height: 42px !important;
                min-width: 42px !important;
                min-height: 42px !important;
                align-items: center !important;
                justify-content: center !important;
                color: {cfg["text"]} !important;
            }}

            [data-testid="stSidebarCollapsedControl"]::before,
            [data-testid="collapsedControl"]::before,
            button[title="Open sidebar"]::before,
            button[aria-label="Open sidebar"]::before,
            button[data-testid="stBaseButton-headerNoPadding"]::before,
            button[data-testid="baseButton-headerNoPadding"]::before,
            button[kind="headerNoPadding"]::before {{
                content: "☰";
                font-size: 22px;
                line-height: 1;
                font-weight: 900;
                color: {cfg["text"]} !important;
            }}

            [data-testid="stSidebarCollapsedControl"] svg,
            [data-testid="collapsedControl"] svg,
            button[title="Open sidebar"] svg,
            button[aria-label="Open sidebar"] svg,
            button[data-testid="stBaseButton-headerNoPadding"] svg,
            button[data-testid="baseButton-headerNoPadding"] svg,
            button[kind="headerNoPadding"] svg {{
                width: 22px !important;
                height: 22px !important;
                color: {cfg["text"]} !important;
                fill: {cfg["text"]} !important;
            }}

            [data-testid="stSidebar"] {{
                width: 286px !important;
                min-width: 286px !important;
            }}

            [data-testid="stSidebarContent"] {{
                width: 286px !important;
            }}

            [data-testid="stSidebarUserContent"] {{
                padding-left: 1rem !important;
                padding-right: 1rem !important;
                padding-bottom: 260px !important;
                margin-top: 0rem !important;
            }}

            [data-testid="stSidebar"] .theme-label {{
                margin-top: 48px !important;
                margin-bottom: 8px !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
                display: grid !important;
                grid-template-columns: 1fr 1fr !important;
                gap: 0.55rem !important;
                width: 100% !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
                width: 100% !important;
                min-width: 0 !important;
                flex: none !important;
            }}

            [data-testid="stSidebar"] .stButton > button {{
                height: 38px !important;
                border-radius: 13px !important;
            }}

            [data-testid="stAppViewContainer"] {{
                overflow-x: hidden !important;
            }}

            .hero {{
                padding: 16px 14px !important;
                border-radius: 19px !important;
                margin-top: 50px !important;
                margin-bottom: 14px !important;
            }}

            .hero-title {{
                font-size: 18px !important;
                line-height: 1.25 !important;
                margin-bottom: 7px !important;
                letter-spacing: -0.2px !important;
            }}

            .hero-subtitle {{
                font-size: 12px !important;
                line-height: 1.5 !important;
            }}

            .section-title {{
                font-size: 17px !important;
                margin-bottom: 5px !important;
            }}

            .section-desc {{
                font-size: 12.2px !important;
                line-height: 1.5 !important;
                margin-bottom: 10px !important;
            }}

            .small-title {{
                font-size: 14px !important;
                margin-top: 2px !important;
                margin-bottom: 5px !important;
            }}

            div[data-testid="stHorizontalBlock"] {{
                gap: 0.45rem !important;
            }}

            div[data-testid="column"] {{
                min-width: 0 !important;
                padding-left: 0 !important;
                padding-right: 0 !important;
            }}

            /* Mobile khusus: slider full 1 baris, 3 input angka sejajar 1 baris */
            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) {{
                display: grid !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: 0.40rem !important;
                align-items: end !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"] {{
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex: none !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:first-child {{
                grid-column: 1 / -1 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:nth-child(2) {{
                grid-column: 1 / 2 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:nth-child(3) {{
                grid-column: 2 / 3 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) > div[data-testid="column"]:nth-child(4) {{
                grid-column: 3 / 4 !important;
            }}

            /* Perbaikan khusus HP: slider satu baris penuh, 3 input angka sejajar di bawahnya */
            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) {{
                display: grid !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: 0.42rem !important;
                align-items: end !important;
                width: 100% !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(1) {{
                grid-column: 1 / -1 !important;
                width: 100% !important;
                max-width: 100% !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(2) {{
                grid-column: 1 / 2 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(3) {{
                grid-column: 2 / 3 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(4) {{
                grid-column: 3 / 4 !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(2),
            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(3),
            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) > div:nth-child(4) {{
                width: 100% !important;
                min-width: 0 !important;
                max-width: 100% !important;
                flex: none !important;
            }}

            div[data-testid="stHorizontalBlock"]:has(> div:nth-child(1) [data-testid="stSlider"]):has(> div:nth-child(2) [data-testid="stNumberInput"]) label p {{
                font-size: 9.2px !important;
                line-height: 1.12 !important;
                white-space: normal !important;
            }}

            .stSlider {{
                margin-bottom: 2px !important;
            }}

            [data-testid="stSlider"] {{
                margin-bottom: 2px !important;
            }}

            [data-testid="stSlider"] span {{
                font-size: 11.5px !important;
            }}

            .stNumberInput {{
                margin-bottom: 4px !important;
            }}

            [data-testid="stNumberInput"] {{
                margin-top: 0px !important;
                margin-bottom: 4px !important;
            }}

            [data-testid="stNumberInput"] label {{
                font-size: 9.2px !important;
                margin-bottom: 4px !important;
                line-height: 1.10 !important;
                min-height: 24px !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] {{
                min-height: 36px !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] input {{
                height: 34px !important;
                min-height: 34px !important;
                font-size: 9.8px !important;
                font-weight: 800 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                padding-left: 6px !important;
                padding-right: 2px !important;
            }}

            [data-testid="stNumberInput"] > div {{
                width: 100% !important;
                min-width: 0 !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
                width: 100% !important;
                min-width: 0 !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] button {{
                height: 34px !important;
                min-height: 34px !important;
                width: 24px !important;
                min-width: 24px !important;
                max-width: 24px !important;
                margin-top: 0px !important;
                padding: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                font-size: 11px !important;
                line-height: 1 !important;
                overflow: visible !important;
                box-sizing: border-box !important;
            }}

            .mobile-kpi-summary {{
                display: block !important;
                background: {cfg["card"]};
                border: 1px solid {cfg["border"]};
                border-radius: 18px;
                padding: 14px 14px 12px 14px;
                margin: 10px 0 14px 0;
                box-shadow: 0 8px 20px {cfg["shadow"]};
            }}

            .mobile-kpi-summary-title {{
                font-size: 14px;
                font-weight: 900;
                color: {cfg["text"]} !important;
                margin-bottom: 10px;
            }}

            .mobile-kpi-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px 12px;
            }}

            .mobile-kpi-item {{
                border-bottom: 1px solid {cfg["border"]};
                padding-bottom: 8px;
                min-width: 0;
            }}

            .mobile-kpi-item:nth-last-child(-n+2) {{
                border-bottom: none;
                padding-bottom: 0;
            }}

            .mobile-kpi-label {{
                font-size: 10.5px;
                line-height: 1.25;
                font-weight: 850;
                color: {cfg["muted"]} !important;
                margin-bottom: 3px;
            }}

            .mobile-kpi-value {{
                font-size: 13.5px;
                line-height: 1.25;
                font-weight: 950;
                color: {cfg["text"]} !important;
                overflow-wrap: anywhere;
            }}

            .mobile-kpi-note {{
                font-size: 9.8px;
                line-height: 1.25;
                font-weight: 650;
                color: {cfg["muted"]} !important;
                margin-top: 2px;
            }}

            .kpi-card {{
                display: none !important;
            }}

            .info-card {{
                min-height: auto !important;
                padding: 13px 13px !important;
                border-radius: 16px !important;
                margin-bottom: 10px !important;
            }}

            .text-muted {{
                font-size: 12.2px !important;
                line-height: 1.5 !important;
            }}

            .text-muted li {{
                margin-bottom: 5px !important;
            }}

            .desktop-chart {{
                display: block !important;
            }}

            .mobile-chart {{
                display: none !important;
            }}


            .stPlotlyChart {{
                border-radius: 14px !important;
                padding: 0px !important;
                margin-bottom: 2px !important;
                max-height: 340px !important;
                overflow: hidden !important;
            }}

            .stPlotlyChart > div,
            .stPlotlyChart .js-plotly-plot,
            .stPlotlyChart .plot-container,
            .stPlotlyChart .svg-container {{
                max-height: 340px !important;
            }}

            .stPlotlyChart svg {{
                max-height: 340px !important;
            }}

            .stPlotlyChart svg .gtitle {{
                font-size: 10.5px !important;
            }}

            .stPlotlyChart svg .xtitle,
            .stPlotlyChart svg .ytitle {{
                font-size: 10px !important;
            }}

            .stPlotlyChart svg .legend text {{
                font-size: 9px !important;
            }}

            .stPlotlyChart svg .annotation-text,
            .stPlotlyChart svg .annotation text {{
                font-size: 9px !important;
            }}

            .custom-table-wrapper {{
                border-radius: 15px !important;
                margin-bottom: 10px !important;
                overflow-x: hidden !important;
                width: 100% !important;
            }}

            table.custom-table {{
                width: 100% !important;
                min-width: 0 !important;
                table-layout: fixed !important;
                font-size: 8.3px !important;
            }}

            table.custom-table thead tr th,
            table.custom-table tbody tr td,
            table.custom-table tbody tr th {{
                padding: 5px 3px !important;
                height: auto !important;
                min-height: 28px !important;
                font-size: 8.3px !important;
                line-height: 1.2 !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
                word-break: break-word !important;
                vertical-align: middle !important;
            }}

            table.custom-table thead tr th {{
                font-size: 8px !important;
                font-weight: 900 !important;
            }}

            .sidebar-visual {{
                display: block !important;
                position: fixed !important;
                left: 16px !important;
                bottom: 22px !important;
                width: 218px !important;
                max-width: 218px !important;
                padding: 14px 13px !important;
                border-radius: 20px !important;
                z-index: 25 !important;
            }}

            .sidebar-emoji {{
                font-size: 30px !important;
                margin-bottom: 6px !important;
            }}

            .sidebar-visual-title {{
                font-size: 15px !important;
            }}

            .sidebar-visual-subtitle {{
                font-size: 10.8px !important;
                line-height: 1.35 !important;
            }}

            .team-name {{
                font-size: 10.5px !important;
                padding: 7px 8px !important;
                margin-top: 8px !important;
            }}

            /* FINAL MOBILE ONLY OVERRIDES */
            [data-testid="stSidebar"] .theme-label {{
                margin-top: 0px !important;
                margin-bottom: 7px !important;
                font-size: 13px !important;
                font-weight: 900 !important;
                line-height: 1.15 !important;
            }}

            /* Mobile: segmented button ☀️ / 🌙 dibuat simetris */
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) {{
                width: 188px !important;
                max-width: 188px !important;
                height: 48px !important;
                min-height: 48px !important;
                display: grid !important;
                grid-template-columns: 94px 94px !important;
                gap: 0 !important;
                column-gap: 0 !important;
                row-gap: 0 !important;
                padding: 0 !important;
                margin: 0 !important;
                border-radius: 21px !important;
                background: {cfg["card"]} !important;
                border: 1px solid {cfg["border"]} !important;
                box-shadow: 0 8px 18px {cfg["shadow"]} !important;
                overflow: hidden !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div,
            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div[data-testid="column"] {{
                width: 94px !important;
                min-width: 94px !important;
                max-width: 94px !important;
                flex: 0 0 94px !important;
                padding: 0 !important;
                margin: 0 !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton {{
                width: 94px !important;
                min-width: 94px !important;
                max-width: 94px !important;
                height: 48px !important;
                padding: 0 !important;
                margin: 0 !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button {{
                width: 94px !important;
                min-width: 94px !important;
                max-width: 94px !important;
                height: 48px !important;
                min-height: 48px !important;
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
                font-size: 22px !important;
                line-height: 1 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
                background: transparent !important;
                transform: none !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div:nth-child(1) .stButton > button {{
                border-radius: 20px 0 0 20px !important;
                background: {cfg["accent_hover"] if mode == "Terang" else "transparent"} !important;
                color: {"white" if mode == "Terang" else cfg["text"]} !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) > div:nth-child(2) .stButton > button {{
                border-radius: 0 20px 20px 0 !important;
                background: {cfg["accent_hover"] if mode == "Gelap" else "transparent"} !important;
                color: {"white" if mode == "Gelap" else cfg["text"]} !important;
            }}

            [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:has(.stButton) .stButton > button:hover {{
                background: {cfg["accent_hover"]} !important;
                color: white !important;
                transform: none !important;
            }}

            /* mobile number input: rapikan area input dan tombol +/- */
            [data-testid="stNumberInput"] {{
                margin-top: 0 !important;
                margin-bottom: 3px !important;
            }}

            [data-testid="stNumberInput"] > div {{
                border-radius: 14px !important;
                min-height: 36px !important;
                height: 36px !important;
                overflow: hidden !important;
                background: {cfg["input_bg"]} !important;
                border: 1px solid {cfg["border"]} !important;
                box-shadow: none !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] {{
                min-height: 36px !important;
                height: 36px !important;
                background: {cfg["input_bg"]} !important;
                overflow: hidden !important;
                border: none !important;
                box-shadow: none !important;
            }}

            [data-testid="stNumberInput"] div[data-baseweb="input"] > div {{
                height: 36px !important;
                min-height: 36px !important;
                background: {cfg["input_bg"]} !important;
                border: none !important;
                box-shadow: none !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] input {{
                height: 36px !important;
                min-height: 36px !important;
                background: {cfg["input_bg"]} !important;
                color: {cfg["text"]} !important;
                border: none !important;
                box-shadow: none !important;
                outline: none !important;
                padding-left: 7px !important;
                font-size: 9.8px !important;
            }}

            [data-testid="stNumberInput"] button {{
                height: 36px !important;
                min-height: 36px !important;
                width: 25px !important;
                min-width: 25px !important;
                max-width: 25px !important;
                margin-top: 0 !important;
                padding: 0 !important;
                background: {cfg["input_btn"]} !important;
                border-radius: 0 !important;
                border-top: none !important;
                border-bottom: none !important;
                border-left: 1px solid {cfg["border"]} !important;
                border-right: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
            }}

            [data-testid="stNumberInput"] button:last-child {{
                border-top-right-radius: 14px !important;
                border-bottom-right-radius: 14px !important;
            }}

            [data-testid="stNumberInput"] button svg {{
                width: 12px !important;
                height: 12px !important;
            }}

            /* compact mobile chart-table spacing: rapat, tapi tidak sampai nabrak */
            div[data-testid="stVerticalBlock"] {{
                gap: 0.18rem !important;
            }}

            .stPlotlyChart {{
                margin-bottom: 4px !important;
                padding-bottom: 0 !important;
            }}

            div[data-testid="stElementContainer"]:has(.stPlotlyChart),
            div[data-testid="element-container"]:has(.stPlotlyChart) {{
                margin-bottom: 4px !important;
                padding-bottom: 0 !important;
            }}

            div[data-testid="stElementContainer"]:has(.stPlotlyChart) + div[data-testid="stElementContainer"],
            div[data-testid="element-container"]:has(.stPlotlyChart) + div[data-testid="element-container"] {{
                margin-top: 2px !important;
                padding-top: 0 !important;
            }}

            .custom-table-wrapper {{
                margin-top: 3px !important;
                margin-bottom: 7px !important;
                padding-top: 0 !important;
            }}

            .small-title {{
                margin-top: 8px !important;
                margin-bottom: 4px !important;
                padding-top: 0 !important;
                line-height: 1.12 !important;
            }}

            .desktop-chart {{
                margin-bottom: 0 !important;
                padding-bottom: 0 !important;
            }}

            .table-title-mobile-tight {{
                margin-top: 10px !important;
                margin-bottom: 4px !important;
                padding-top: 0 !important;
                line-height: 1.12 !important;
            }}

        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    return cfg


theme = apply_theme(st.session_state.theme_mode)


def inject_mobile_sidebar_button():
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          const isMobile = () => window.parent.innerWidth <= 900;
          function findNativeSidebarButton() {
            return doc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
                   doc.querySelector('[data-testid="stSidebarCollapsedControl"]') ||
                   doc.querySelector('[data-testid="collapsedControl"] button') ||
                   doc.querySelector('[data-testid="collapsedControl"]') ||
                   doc.querySelector('button[title="Open sidebar"]') ||
                   doc.querySelector('button[aria-label="Open sidebar"]') ||
                   doc.querySelector('button[kind="headerNoPadding"]') ||
                   doc.querySelector('button[data-testid="baseButton-headerNoPadding"]') ||
                   doc.querySelector('button[data-testid="stBaseButton-headerNoPadding"]');
          }
          function makeButton() {
            let b = doc.getElementById('custom-mobile-sidebar-button');
            if (!b) {
              b = doc.createElement('button');
              b.id = 'custom-mobile-sidebar-button';
              b.innerHTML = '☰';
              b.setAttribute('aria-label', 'Buka menu');
              b.onclick = function() {
                const native = findNativeSidebarButton();
                if (native) native.click();
              };
              doc.body.appendChild(b);
            }
            b.style.cssText = `
              display: ${isMobile() ? 'flex' : 'none'};
              position: fixed;
              top: 10px;
              left: 10px;
              z-index: 2147483647;
              width: 42px;
              height: 42px;
              align-items: center;
              justify-content: center;
              border-radius: 13px;
              border: 1px solid rgba(120,145,125,.55);
              background: rgba(20,32,25,.96);
              color: #F8FAF7;
              font-size: 23px;
              font-weight: 900;
              line-height: 1;
              box-shadow: 0 8px 22px rgba(0,0,0,.24);
              cursor: pointer;
            `;
          }
          makeButton();
          window.parent.addEventListener('resize', makeButton);
          setInterval(makeButton, 900);
        })();
        </script>
        """,
        height=0,
    )

inject_mobile_sidebar_button()


# ============================================================
# UI COMPONENTS
# ============================================================

def kpi_card(label, value, note=None):
    note_html = f'<div class="kpi-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def mobile_kpi_summary(items):
    html_items = ""
    for item in items:
        note_html = f'<div class="mobile-kpi-note">{item.get("note", "")}</div>' if item.get("note") else ""
        html_items += (
            f'<div class="mobile-kpi-item">'
            f'<div class="mobile-kpi-label">{item["label"]}</div>'
            f'<div class="mobile-kpi-value">{item["value"]}</div>'
            f'{note_html}'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="mobile-kpi-summary">
            <div class="mobile-kpi-summary-title">Ringkasan Simulasi</div>
            <div class="mobile-kpi-grid">
                {html_items}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def bullet_card(title, items):
    html = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f"""
        <div class="info-card">
            <div class="small-title">{title}</div>
            <ul class="text-muted" style="padding-left:20px; margin-bottom:0;">
                {html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_table(data):
    html = data.to_html(classes="custom-table", border=0, index=False, escape=False)
    st.markdown(
        f"""
        <div class="custom-table-wrapper">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CHART
# ============================================================

def make_forecast_chart(ts, forecast, theme):
    hist = ts.tail(24)

    forecast_start = forecast.index.min()
    max_pred_value = forecast.max()
    max_pred_date = forecast.idxmax()

    y_min = min(hist.min(), forecast.min()) * 0.72
    y_max = max(hist.max(), forecast.max()) * 1.12

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        mode="lines+markers",
        name="Data historis",
        line=dict(color=theme["chart_hist"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_hist"],
            line=dict(color=theme["chart_bg"], width=1.5)
        ),
        fill="tozeroy",
        fillcolor=theme["chart_hist_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Historis: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name="Prediksi",
        line=dict(color=theme["chart_pred"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_pred"],
            line=dict(color=theme["chart_bg"], width=1.5)
        ),
        fill="tozeroy",
        fillcolor=theme["chart_pred_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_vline(
        x=forecast_start,
        line_width=1.6,
        line_dash="dash",
        line_color=theme["chart_divider"]
    )

    fig.add_annotation(
        x=forecast_start,
        y=y_max * 0.98,
        text="<b>Mulai prediksi</b>",
        showarrow=False,
        font=dict(size=12, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=4
    )

    fig.add_annotation(
        x=max_pred_date,
        y=max_pred_value,
        text=f"<b>Prediksi tertinggi</b><br>{format_integer(max_pred_value)} ton",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.4,
        arrowcolor=theme["chart_divider"],
        ax=55,
        ay=-40,
        font=dict(size=12, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=4
    )

    fig.update_layout(
        title=dict(
            text="<b>Prediksi Jumlah Sampah Kota Bandung</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=54, r=8, t=86, b=52),
        height=470,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,
            xanchor="right",
            x=0.99,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=12, color=theme["chart_font"], family="Arial")
        )
    )

    fig.update_xaxes(
        title="<b>Periode</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%b %Y",
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title="<b>Jumlah sampah (ton)</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        range=[y_min, y_max],
        automargin=True
    )

    return fig




def make_forecast_chart_mobile(ts, forecast, theme):
    hist = ts.tail(15)

    forecast_start = forecast.index.min()
    max_pred_value = forecast.max()
    max_pred_date = forecast.idxmax()

    y_min = min(hist.min(), forecast.min()) * 0.78
    y_max = max(hist.max(), forecast.max()) * 1.08

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        mode="lines+markers",
        name="Historis",
        line=dict(color=theme["chart_hist"], width=1.8, shape="linear"),
        marker=dict(size=3.4, color=theme["chart_hist"], line=dict(color=theme["chart_bg"], width=0.7)),
        fill="tozeroy",
        fillcolor=theme["chart_hist_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Historis: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        mode="lines+markers",
        name="Prediksi",
        line=dict(color=theme["chart_pred"], width=1.8, shape="linear"),
        marker=dict(size=3.4, color=theme["chart_pred"], line=dict(color=theme["chart_bg"], width=0.7)),
        fill="tozeroy",
        fillcolor=theme["chart_pred_fill"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_vline(
        x=forecast_start,
        line_width=1,
        line_dash="dash",
        line_color=theme["chart_divider"]
    )

    fig.add_annotation(
        x=forecast_start,
        y=y_max * 0.96,
        text="<b>Mulai</b>",
        showarrow=False,
        font=dict(size=7.2, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=2
    )

    fig.add_annotation(
        x=max_pred_date,
        y=max_pred_value,
        text=f"<b>Tertinggi</b><br>{format_integer(max_pred_value)} ton",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.8,
        arrowwidth=1,
        arrowcolor=theme["chart_divider"],
        ax=22,
        ay=-24,
        font=dict(size=7.2, color=theme["chart_font"], family="Arial"),
        bgcolor=theme["annotation_bg"],
        bordercolor=theme["annotation_border"],
        borderwidth=1,
        borderpad=2
    )

    fig.update_layout(
        title=dict(
            text="<b>Prediksi Sampah</b>",
            x=0.5,
            xanchor="center",
            y=0.96,
            font=dict(size=8.5, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=28, r=4, t=30, b=30),
        height=210,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.00,
            xanchor="right",
            x=0.99,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=7.2, color=theme["chart_font"], family="Arial")
        )
    )

    fig.update_xaxes(
        title="<b>Periode</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%b %Y",
        nticks=4,
        tickfont=dict(size=7.2, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=8, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title="<b>Ton</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=7.2, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=8, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        range=[y_min, y_max],
        automargin=True
    )

    return fig


def make_eval_chart(actual, predicted, theme):
    y_max = max(actual.max(), predicted.max()) * 1.15
    y_min = min(actual.min(), predicted.min()) * 0.90

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode="lines+markers",
        name="Aktual",
        line=dict(color=theme["chart_hist"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_hist"],
            line=dict(color=theme["chart_bg"], width=1.4)
        ),
        hovertemplate="<b>%{x|%b %Y}</b><br>Aktual: %{y:,.0f} ton<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=predicted.index,
        y=predicted.values,
        mode="lines+markers",
        name="Prediksi",
        line=dict(color=theme["chart_pred"], width=3, shape="linear"),
        marker=dict(
            size=7,
            color=theme["chart_pred"],
            line=dict(color=theme["chart_bg"], width=1.4)
        ),
        hovertemplate="<b>%{x|%b %Y}</b><br>Prediksi: %{y:,.0f} ton<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="<b>Aktual vs Prediksi Data Uji</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color=theme["chart_font"], family="Arial")
        ),
        paper_bgcolor=theme["chart_bg"],
        plot_bgcolor=theme["chart_bg"],
        margin=dict(l=54, r=8, t=86, b=52),
        height=430,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.07,
            xanchor="right",
            x=0.99,
            bgcolor=theme["chart_legend_bg"],
            bordercolor=theme["chart_legend_border"],
            borderwidth=1,
            font=dict(size=12, color=theme["chart_font"], family="Arial")
        )
    )

    fig.update_xaxes(
        title="<b>Periode</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickformat="%b %Y",
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        title="<b>Jumlah sampah (ton)</b>",
        showgrid=True,
        gridcolor=theme["chart_grid"],
        tickfont=dict(size=12, color=theme["chart_axis"], family="Arial"),
        title_font=dict(size=14, color=theme["chart_axis"], family="Arial"),
        zeroline=False,
        range=[y_min, y_max],
        automargin=True
    )

    return fig


# ============================================================
# LOAD ACTUAL DATA
# ============================================================

df_raw, df, ts = load_data()

provinsi = ", ".join(df_raw["nama_provinsi"].dropna().unique())
kota = ", ".join(df_raw["bps_nama_kabupaten_kota"].dropna().unique())
satuan = ", ".join(df_raw["satuan"].dropna().unique())
periode_data = f"{format_periode(ts.index.min())} - {format_periode(ts.index.max())}"


# ============================================================
# SIDEBAR KIRI
# ============================================================

st.sidebar.markdown('<div class="theme-label">Pilih Tampilan</div>', unsafe_allow_html=True)

theme_col1, theme_col2 = st.sidebar.columns(2)

with theme_col1:
    st.button(
        "☀️",
        use_container_width=True,
        on_click=set_theme,
        args=("Terang",)
    )

with theme_col2:
    st.button(
        "🌙",
        use_container_width=True,
        on_click=set_theme,
        args=("Gelap",)
    )

menu = st.sidebar.radio(
    "Menu Utama",
    MENU_OPTIONS,
    key="active_menu"
)

st.sidebar.markdown(
    """
    <div class="sidebar-visual">
        <div class="sidebar-emoji">♻️ 🗑️ 🍃</div>
        <div class="sidebar-visual-title">Dashboard Sampah</div>
        <div class="sidebar-visual-subtitle">
            Prediksi jumlah sampah, estimasi anggaran, volume sampah, dan kebutuhan muatan truk compactor.
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
        <div class="hero-title">Simulasi Pengelolaan Sampah Kota Bandung</div>
        <div class="hero-subtitle">
            Dashboard ini difokuskan untuk membantu staf DLH melakukan simulasi kebutuhan operasional
            berdasarkan prediksi jumlah sampah bulanan.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MENU 1
# ============================================================

if menu == "Simulasi Pengelolaan":
    st.markdown('<div class="section-title">Simulasi Pengelolaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Isi operasional, lalu sistem akan menghitung prediksi sampah, anggaran, volume sampah, dan kebutuhan muatan truk compactor.</div>',
        unsafe_allow_html=True
    )

    input1, input2, input3, input4 = st.columns(4, gap="large")

    with input1:
        forecast_steps = st.slider(
            "Simulasi untuk berapa bulan ke depan?",
            min_value=1,
            max_value=12,
            value=12,
            step=1
        )

    with input2:
        biaya_per_ton = st.number_input(
            "Biaya penanganan per ton",
            min_value=0,
            value=308482,
            step=1000
        )

    with input3:
        kapasitas_truk_compactor_m3 = st.selectbox(
            "Kapasitas truk compactor (m³)",
            options=[6, 12],
            index=1
        )

    with input4:
        hari_operasional_angkut_per_minggu = st.number_input(
            "Hari operasional angkut per minggu",
            min_value=1,
            max_value=7,
            value=4,
            step=1
        )

    forecast = make_sarima_forecast(ts, forecast_steps)

    simulation_df = build_simulation_table(
        forecast=forecast,
        biaya_per_ton=biaya_per_ton,
        kapasitas_truk_compactor_m3=kapasitas_truk_compactor_m3,
        hari_operasional_angkut_per_minggu=hari_operasional_angkut_per_minggu
    )

    total_sampah = simulation_df["Prediksi Sampah (Ton)"].sum()
    total_anggaran = simulation_df["Estimasi Anggaran"].sum()
    total_volume_sampah = simulation_df["Estimasi Volume Sampah (m³)"].sum()
    total_muatan_truk = simulation_df["Estimasi Kebutuhan Muatan Truk"].sum()

    highest_row = simulation_df.loc[simulation_df["Prediksi Sampah (Ton)"].idxmax()]
    lowest_row = simulation_df.loc[simulation_df["Prediksi Sampah (Ton)"].idxmin()]

    start_period = format_periode(simulation_df["Tanggal"].min())
    end_period = format_periode(simulation_df["Tanggal"].max())

    mobile_kpi_summary([
        {"label": "Periode", "value": f"{start_period} - {end_period}", "note": f"{forecast_steps} bulan ke depan"},
        {"label": "Total Sampah", "value": f"{format_angka(total_sampah)} ton"},
        {"label": "Total Anggaran", "value": format_rupiah(total_anggaran), "note": f"{format_rupiah(biaya_per_ton)} per ton"},
        {"label": "Total Volume", "value": f"{format_angka(total_volume_sampah)} m³", "note": f"densitas {format_angka(DENSITAS_SAMPAH_KG_PER_M3)} kg/m³"},
        {"label": "Beban Tertinggi", "value": highest_row["Periode"], "note": f"{format_angka(highest_row['Prediksi Sampah (Ton)'])} ton"},
        {"label": "Beban Terendah", "value": lowest_row["Periode"], "note": f"{format_angka(lowest_row['Prediksi Sampah (Ton)'])} ton"},
        {"label": "Total Muatan Truk", "value": f"{format_integer(total_muatan_truk)} muatan", "note": f"kapasitas {kapasitas_truk_compactor_m3} m³"},
        {"label": "Muatan Maks/Hari Angkut", "value": f"{format_integer(int(simulation_df['Muatan Truk per Hari Angkut'].max()))} muatan/hari"},
        {"label": "Hari Angkut/Minggu", "value": f"{hari_operasional_angkut_per_minggu} hari/minggu", "note": "parameter jadwal angkut"},
    ])

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4, gap="large")

    with row1_col1:
        kpi_card("Periode Simulasi", f"{start_period} - {end_period}", f"{forecast_steps} bulan ke depan")

    with row1_col2:
        kpi_card("Total Prediksi Sampah", f"{format_angka(total_sampah)} ton")

    with row1_col3:
        kpi_card("Total Estimasi Anggaran", format_rupiah(total_anggaran), f" {format_rupiah(biaya_per_ton)} per ton")

    with row1_col4:
        kpi_card("Total Estimasi Volume", f"{format_angka(total_volume_sampah)} m³", f"Densitas {format_angka(DENSITAS_SAMPAH_KG_PER_M3)} kg/m³")

    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4, gap="large")

    with row2_col1:
        kpi_card("Beban Tertinggi", highest_row["Periode"], f"{format_angka(highest_row['Prediksi Sampah (Ton)'])} ton")

    with row2_col2:
        kpi_card("Beban Terendah", lowest_row["Periode"], f"{format_angka(lowest_row['Prediksi Sampah (Ton)'])} ton")

    with row2_col3:
        kpi_card("Total Kebutuhan Muatan Truk", f"{format_integer(total_muatan_truk)} muatan", f"Kapasitas {kapasitas_truk_compactor_m3} m³")

    with row2_col4:
        kpi_card("Muatan Maksimum per Hari Angkut", f"{format_integer(int(simulation_df['Muatan Truk per Hari Angkut'].max()))} muatan/hari", f"{hari_operasional_angkut_per_minggu} hari angkut/minggu")

    fig_forecast = make_forecast_chart(ts, forecast, theme)
    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True}
    )

    st.markdown('<div class="small-title table-title-mobile-tight">Tabel Simulasi Kebutuhan Operasional</div>', unsafe_allow_html=True)
    show_table(prepare_display_table(simulation_df))

    bullet_card(
        "Catatan simulasi",
        [
            f"Biaya penanganan yang digunakan adalah <b>{format_rupiah(biaya_per_ton)} per ton</b>.",
            f"Kapasitas truk compactor digunakan sebesar <b>{kapasitas_truk_compactor_m3} m³</b> per muatan.",
            f"Prediksi sampah dalam ton dikonversi menjadi volume menggunakan densitas <b>{format_angka(DENSITAS_SAMPAH_KG_PER_M3)} kg/m³</b>.",
            f"Hari operasional angkut digunakan sebesar <b>{hari_operasional_angkut_per_minggu} hari per minggu</b>.",
            "Kebutuhan muatan truk dan hari operasional angkut dibulatkan ke atas agar estimasi tidak kurang dari kebutuhan lapangan.",
            "Dashboard ini berfungsi sebagai simulasi awal untuk membantu perencanaan operasional."
        ]
    )


# ============================================================
# MENU 2
# ============================================================

elif menu == "Ringkasan Data & Model":
    st.markdown('<div class="section-title">Ringkasan Data & Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Halaman ini menampilkan ringkasan data dan evaluasi model secara singkat. Detail teori, EDA, dan pembentukan model dijelaskan pada laporan.</div>',
        unsafe_allow_html=True
    )

    eval_df, comparison_df, test_actual, test_forecast = evaluate_sarima(ts)

    left, right = st.columns(2, gap="large")

    with left:
        bullet_card(
            "Ringkasan Data",
            [
                f"Wilayah: <b>{kota}</b>",
                f"Provinsi: <b>{provinsi}</b>",
                f"Jumlah data: <b>{len(df_raw)} baris</b>",
                f"Periode: <b>{periode_data}</b>",
                f"Satuan: <b>{satuan}</b>",
                "Variabel utama: <b>jumlah_sampah</b>",
                "Bentuk data: <b>bulanan</b>",
                f"Missing value: <b>{int(df_raw.isnull().sum().sum())}</b>",
                f"Data duplikat: <b>{int(df_raw.duplicated().sum())}</b>"
            ]
        )

    with right:
        bullet_card(
            "Ringkasan Model",
            [
                "Model yang digunakan: <b>SARIMA(1,2,2)(0,1,1,12)</b>.",
                "Dashboard hanya memakai satu model agar lebih mudah dipahami user.",
                "Prediksi dibatasi maksimal <b>12 bulan ke depan</b> agar fokus pada perencanaan jangka pendek.",
                "Model digunakan untuk menghasilkan estimasi jumlah sampah, anggaran, volume sampah, dan kebutuhan muatan truk compactor.",
                "Detail teori, EDA, parameter model, dan evaluasi lengkap dijelaskan pada laporan."
            ]
        )

    st.markdown('<div class="small-title">Evaluasi Model</div>', unsafe_allow_html=True)
    show_table(prepare_eval_display(eval_df))

    st.markdown('<div class="small-title">Aktual vs Prediksi Data Uji</div>', unsafe_allow_html=True)
    fig_eval = make_eval_chart(test_actual, test_forecast, theme)
    st.plotly_chart(fig_eval, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    st.markdown('<div class="small-title table-title-mobile-tight">Tabel Aktual vs Prediksi</div>', unsafe_allow_html=True)
    show_table(prepare_comparison_display(comparison_df))

    
