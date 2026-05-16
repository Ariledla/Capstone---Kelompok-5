import warnings

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    6: "Juli" if False else "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}


# ============================================================
# FUNGSI DASAR
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


def build_simulation_table(
    forecast,
    biaya_per_ton,
    kapasitas_truk,
    rit_per_truk_per_hari
):
    output = pd.DataFrame({
        "Tanggal": forecast.index,
        "Periode": [format_periode(date) for date in forecast.index],
        "Prediksi Sampah (Ton)": forecast.values
    })

    output["Jumlah Hari"] = output["Tanggal"].dt.days_in_month

    output["Estimasi Anggaran"] = (
        output["Prediksi Sampah (Ton)"] * biaya_per_ton
    )

    output["Kebutuhan Rit Bulanan"] = np.ceil(
        output["Prediksi Sampah (Ton)"] / kapasitas_truk
    )

    output["Kebutuhan Rit per Hari"] = np.ceil(
        output["Kebutuhan Rit Bulanan"] / output["Jumlah Hari"]
    )

    output["Estimasi Armada per Hari"] = np.ceil(
        output["Kebutuhan Rit per Hari"] / rit_per_truk_per_hari
    )

    return output


def prepare_display_table(output):
    display = output.copy()

    display["Prediksi Sampah (Ton)"] = display["Prediksi Sampah (Ton)"].apply(format_angka)
    display["Estimasi Anggaran"] = display["Estimasi Anggaran"].apply(format_rupiah)
    display["Kebutuhan Rit Bulanan"] = display["Kebutuhan Rit Bulanan"].astype(int).apply(format_integer)
    display["Kebutuhan Rit per Hari"] = display["Kebutuhan Rit per Hari"].astype(int).apply(format_integer)
    display["Estimasi Armada per Hari"] = display["Estimasi Armada per Hari"].astype(int).apply(format_integer)

    display = display[
        [
            "Periode",
            "Prediksi Sampah (Ton)",
            "Estimasi Anggaran",
            "Kebutuhan Rit Bulanan",
            "Kebutuhan Rit per Hari",
            "Estimasi Armada per Hari"
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
# TEMA DAN STYLE
# ============================================================

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
            line-height: 1.6;
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
            line-height: 1.65;
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
            padding: 18px 19px;
            min-height: 116px;
            box-shadow: 0 8px 26px rgba(31, 41, 51, 0.06);
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 9px;
            margin-bottom: 24px;
        }}

        .kpi-label {{
            color: {muted} !important;
            font-size: 14px;
            font-weight: 900;
            line-height: 1.2;
        }}

        .kpi-value {{
            color: {text} !important;
            font-size: 23px;
            font-weight: 900;
            line-height: 1.12;
        }}

        .kpi-note {{
            color: {muted} !important;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.45;
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

        .stSlider label, .stSelectbox label, .stRadio label, .stNumberInput label {{
            color: {text} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stSlider"] span {{
            color: {text} !important;
        }}

        [data-testid="stNumberInput"] input {{
            background: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 14px !important;
            color: {text} !important;
            min-height: 42px !important;
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

        .stPyplot {{
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

    return {
        "plot_bg": plot_bg,
        "text": text,
        "muted": muted,
        "card": card,
        "border": border
    }


# ============================================================
# KOMPONEN UI
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


def show_table(data):
    html = data.to_html(
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


def style_plot(ax, title):
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.30)
    return ax


# ============================================================
# LOAD DATA
# ============================================================

df_raw, df, ts = load_data()

provinsi = ", ".join(df_raw["nama_provinsi"].dropna().unique())
kota = ", ".join(df_raw["bps_nama_kabupaten_kota"].dropna().unique())
satuan = ", ".join(df_raw["satuan"].dropna().unique())

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
    if st.button("   ☀️   ", use_container_width=True):
        st.session_state.theme_mode = "Terang"
        st.rerun()

with theme_col2:
    if st.button("   🌙   ", use_container_width=True):
        st.session_state.theme_mode = "Gelap"
        st.rerun()

menu = st.sidebar.radio(
    "Menu Utama",
    [
        "Simulasi Pengelolaan",
        "Data & Evaluasi"
    ]
)

st.sidebar.markdown(
    """
    <div class="sidebar-visual">
        <div class="sidebar-emoji">♻️ 🗑️ 🍃</div>
        <div class="sidebar-visual-title">Dashboard Sampah</div>
        <div class="sidebar-visual-subtitle">
            Prediksi, anggaran, rit pengangkutan, dan kebutuhan armada.
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
            Prediksi jumlah sampah, estimasi anggaran, kebutuhan rit, dan armada untuk mendukung perencanaan DLH.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HALAMAN 1: SIMULASI PENGELOLAAN
# ============================================================

if menu == "Simulasi Pengelolaan":
    st.markdown('<div class="section-title">Simulasi Pengelolaan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Isi asumsi operasional, lalu sistem akan menghitung kebutuhan sampah, anggaran, rit, dan armada.</div>',
        unsafe_allow_html=True
    )

    input_col1, input_col2, input_col3, input_col4 = st.columns(4, gap="large")

    with input_col1:
        forecast_steps = st.slider(
            "Simulasi untuk berapa bulan ke depan?",
            min_value=1,
            max_value=12,
            value=12,
            step=1
        )

    with input_col2:
        biaya_per_ton = st.number_input(
            "Biaya penanganan per ton",
            min_value=0,
            value=300000,
            step=50000
        )

    with input_col3:
        kapasitas_truk = st.number_input(
            "Kapasitas truk per rit (ton)",
            min_value=1.0,
            value=5.0,
            step=0.5
        )

    with input_col4:
        rit_per_truk_per_hari = st.number_input(
            "Rit per truk per hari",
            min_value=1,
            value=2,
            step=1
        )

    forecast = make_sarima_forecast(ts, forecast_steps)

    simulation_df = build_simulation_table(
        forecast=forecast,
        biaya_per_ton=biaya_per_ton,
        kapasitas_truk=kapasitas_truk,
        rit_per_truk_per_hari=rit_per_truk_per_hari
    )

    total_sampah = simulation_df["Prediksi Sampah (Ton)"].sum()
    total_anggaran = simulation_df["Estimasi Anggaran"].sum()
    total_rit = simulation_df["Kebutuhan Rit Bulanan"].sum()

    highest_row = simulation_df.loc[
        simulation_df["Prediksi Sampah (Ton)"].idxmax()
    ]

    lowest_row = simulation_df.loc[
        simulation_df["Prediksi Sampah (Ton)"].idxmin()
    ]

    start_period = format_periode(simulation_df["Tanggal"].min())
    end_period = format_periode(simulation_df["Tanggal"].max())

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        kpi_card(
            "Periode Simulasi",
            f"{start_period} - {end_period}",
            f"{forecast_steps} bulan ke depan"
        )

    with col2:
        kpi_card(
            "Total Prediksi Sampah",
            f"{format_angka(total_sampah)} ton"
        )

    with col3:
        kpi_card(
            "Total Estimasi Anggaran",
            format_rupiah(total_anggaran),
            f"Asumsi {format_rupiah(biaya_per_ton)} per ton"
        )

    with col4:
        kpi_card(
            "Total Kebutuhan Rit",
            f"{format_integer(total_rit)} rit",
            f"Kapasitas {kapasitas_truk} ton per rit"
        )

    col5, col6, col7, col8 = st.columns(4, gap="large")

    with col5:
        kpi_card(
            "Beban Tertinggi",
            highest_row["Periode"],
            f"{format_angka(highest_row['Prediksi Sampah (Ton)'])} ton"
        )

    with col6:
        kpi_card(
            "Beban Terendah",
            lowest_row["Periode"],
            f"{format_angka(lowest_row['Prediksi Sampah (Ton)'])} ton"
        )

    with col7:
        max_rit_harian = int(simulation_df["Kebutuhan Rit per Hari"].max())
        kpi_card(
            "Rit Maksimum per Hari",
            f"{format_integer(max_rit_harian)} rit/hari"
        )

    with col8:
        max_armada = int(simulation_df["Estimasi Armada per Hari"].max())
        kpi_card(
            "Armada Maksimum per Hari",
            f"{format_integer(max_armada)} truk",
            f"Asumsi {rit_per_truk_per_hari} rit/truk/hari"
        )

    st.markdown('<div class="small-title">Prediksi Jumlah Sampah</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10.5, 3.6), facecolor=plot_bg)

    ax.plot(
        ts.tail(24),
        marker="o",
        linewidth=1.2,
        markersize=3,
        label="Data Historis"
    )

    ax.plot(
        forecast,
        marker="o",
        linewidth=2,
        markersize=3,
        label="Prediksi"
    )

    style_plot(ax, "Prediksi Jumlah Sampah Kota Bandung")
    ax.set_xlabel("Periode", fontsize=8)
    ax.set_ylabel("Jumlah Sampah (Ton)", fontsize=8)
    ax.legend(fontsize=7)

    st.pyplot(fig, use_container_width=True)

    st.markdown('<div class="small-title">Tabel Simulasi Kebutuhan Operasional</div>', unsafe_allow_html=True)

    display_table = prepare_display_table(simulation_df)
    show_table(display_table)

    bullet_card(
        "Catatan simulasi",
        [
            f"Biaya penanganan sampah: <b>{format_rupiah(biaya_per_ton)} per ton</b>.",
            f"Kapasitas truk: <b>{kapasitas_truk} ton per rit</b>.",
            f"Rit per truk: <b>{rit_per_truk_per_hari} rit per hari</b>.",
            "Kebutuhan rit dan armada dibulatkan ke atas agar estimasi tidak kurang dari kebutuhan operasional.",
            "Hasil prediksi merupakan simulasi berbasis data historis dan tetap perlu disesuaikan dengan kondisi lapangan."
        ]
    )

# ============================================================
# HALAMAN 2: DATA & EVALUASI
# ============================================================

elif menu == "Data & Evaluasi":
    st.markdown('<div class="section-title">Data & Evaluasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Ringkasan data dan evaluasi model ditampilkan singkat. Detail teori, EDA, dan proses pemodelan dijelaskan pada laporan.</div>',
        unsafe_allow_html=True
    )

    eval_df, comparison_df, test_actual, test_forecast = evaluate_sarima(ts)

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        kpi_card("Wilayah", kota)

    with col2:
        kpi_card(
            "Periode Data",
            f"{format_periode(ts.index.min())} - {format_periode(ts.index.max())}"
        )

    with col3:
        kpi_card("Jumlah Data", f"{len(df_raw)} baris")

    with col4:
        kpi_card("Satuan", satuan)

    left, right = st.columns([1, 1], gap="large")

    with left:
        bullet_card(
            "Data yang digunakan",
            [
                f"Provinsi: <b>{provinsi}</b>",
                f"Kota/Kabupaten: <b>{kota}</b>",
                "Variabel utama: <b>jumlah_sampah</b>.",
                "Data berbentuk bulanan.",
                f"Missing value: <b>{int(df_raw.isnull().sum().sum())}</b>.",
                f"Data duplikat: <b>{int(df_raw.duplicated().sum())}</b>."
            ]
        )

    with right:
        bullet_card(
            "Model yang dipakai",
            [
                "Model utama pada dashboard: <b>SARIMA(1,2,2)(0,1,1,12)</b>.",
                "Model digunakan karena dapat menangkap pola bulanan pada data historis.",
                "Prediksi pada aplikasi dibatasi maksimal <b>12 bulan ke depan</b>.",
                "Detail pemilihan model dan pembahasan teknis dijelaskan di laporan."
            ]
        )

    st.markdown('<div class="small-title">Evaluasi Model pada Data Uji</div>', unsafe_allow_html=True)
    show_table(prepare_eval_display(eval_df))

    left_plot, right_table = st.columns([1.15, 1], gap="large")

    with left_plot:
        st.markdown('<div class="small-title">Aktual vs Prediksi Data Uji</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6.2, 3.0), facecolor=plot_bg)

        ax.plot(
            test_actual,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label="Aktual"
        )

        ax.plot(
            test_forecast,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label="Prediksi"
        )

        style_plot(ax, "Perbandingan Aktual dan Prediksi")
        ax.set_xlabel("Periode", fontsize=8)
        ax.set_ylabel("Jumlah Sampah (Ton)", fontsize=8)
        ax.legend(fontsize=7)

        st.pyplot(fig, use_container_width=True)

    with right_table:
        st.markdown('<div class="small-title">Tabel Aktual vs Prediksi</div>', unsafe_allow_html=True)
        show_table(prepare_comparison_display(comparison_df))

    st.markdown('<div class="small-title">Contoh Data</div>', unsafe_allow_html=True)
    show_table(df_raw.head(10))

    bullet_card(
        "Catatan",
        [
            "Evaluasi model ditampilkan secara ringkas agar dashboard tetap fokus pada kebutuhan pengguna.",
            "Penjelasan detail seperti EDA, parameter model, dan alasan pemilihan model dapat diletakkan pada laporan.",
            "Nilai evaluasi digunakan sebagai gambaran kesalahan prediksi, bukan sebagai satu-satunya dasar keputusan operasional."
        ]
    )
