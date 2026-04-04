import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Prediksi Jumlah Sampah Kota Bandung",
    layout="wide"
)

FILE_PATH = "jumlah_capaian_penanganan_sampah_di_kota_bandung.xlsx"

bulan_map = {
    "JANUARI": "01",
    "FEBRUARI": "02",
    "MARET": "03",
    "APRIL": "04",
    "MEI": "05",
    "JUNI": "06",
    "JULI": "07",
    "AGUSTUS": "08",
    "SEPTEMBER": "09",
    "OKTOBER": "10",
    "NOVEMBER": "11",
    "DESEMBER": "12",
}


@st.cache_data
def load_data():
    df = pd.read_excel(FILE_PATH)
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    data = df.copy()
    data["bulan_num"] = data["bulan"].astype(str).str.upper().map(bulan_map)
    data["tanggal"] = pd.to_datetime(
        data["tahun"].astype(str) + "-" + data["bulan_num"],
        errors="coerce"
    )
    data = data.dropna(subset=["tanggal"]).sort_values("tanggal")
    data = data.set_index("tanggal")
    ts = data["jumlah_sampah"].astype(float).copy()
    return data, ts


def split_series(ts: pd.Series, test_size: int = 12):
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]
    return train, test


def compute_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)
    return mae, rmse, mape, r2


@st.cache_data
def search_best_arima(train: pd.Series):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    best_aic = float("inf")
    best_order = None

    for order in pdq:
        try:
            model = ARIMA(train, order=order)
            result = model.fit()
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
        except Exception:
            continue

    return best_order, best_aic


@st.cache_data
def search_best_sarima(train: pd.Series):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

    best_aic = float("inf")
    best_order = None
    best_seasonal = None

    for order in pdq:
        for seasonal in seasonal_pdq:
            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                result = model.fit(disp=False)

                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = order
                    best_seasonal = seasonal
            except Exception:
                continue

    return best_order, best_seasonal, best_aic


try:
    raw_df = load_data()
    df, ts = preprocess_data(raw_df)
    train, test = split_series(ts, test_size=12)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()


st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Beranda",
        "Understanding Data",
        "EDA",
        "Stasioneritas",
        "Modeling",
        "Forecasting",
        "Evaluasi",
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Jumlah observasi:", len(ts))
st.sidebar.write("Periode awal:", str(ts.index.min().date()))
st.sidebar.write("Periode akhir:", str(ts.index.max().date()))


if menu == "Beranda":
    st.title("Prediksi Jumlah Sampah Kota Bandung")

    st.write(
        """
        Aplikasi ini digunakan untuk melakukan analisis time series dan prediksi
        jumlah sampah Kota Bandung berdasarkan data historis bulanan.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", len(ts))
    col2.metric("Periode Awal", str(ts.index.min().date()))
    col3.metric("Periode Akhir", str(ts.index.max().date()))

    st.subheader("Tujuan")
    st.markdown(
        """
        - Memahami pola historis jumlah sampah
        - Melakukan eksplorasi data time series
        - Membandingkan model ARIMA dan SARIMA
        - Menentukan model terbaik berdasarkan hasil evaluasi
        """
    )

    st.subheader("Preview Data")
    st.dataframe(df.head(10), use_container_width=True)


elif menu == "Understanding Data":
    st.title("Understanding Data")

    st.subheader("Sample Data")
    st.dataframe(raw_df.sample(min(5, len(raw_df))), use_container_width=True)

    st.subheader("Informasi Data")
    info_df = pd.DataFrame({
        "Kolom": raw_df.columns,
        "Tipe Data": raw_df.dtypes.astype(str).values,
        "Non-Null Count": raw_df.notnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(df[["jumlah_sampah"]].describe().T, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing values:", int(raw_df.isnull().sum().sum()))
    with col2:
        st.write("Duplikat:", int(raw_df.duplicated().sum()))

    st.subheader("Hasil Preprocessing")
    st.write(
        "Data telah dikonversi menjadi time series bulanan dengan indeks tanggal "
        "sehingga siap digunakan untuk analisis dan pemodelan."
    )
    st.dataframe(df.head(), use_container_width=True)


elif menu == "EDA":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Time Series Plot",
        "Moving Average",
        "Boxplot per Tahun",
        "Seasonal Decomposition"
    ])

    with tab1:
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, label="Jumlah Sampah")
        ax.set_title("Jumlah Sampah Kota Bandung (2017–2024)")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown(
            """
            - 2017 awal relatif rendah
            - 2017–2020 tren meningkat
            - 2020–2022 relatif stabil
            - 2023–2024 penurunan signifikan
            """
        )

    with tab2:
        st.subheader("Moving Average")
        rolling_mean = ts.rolling(window=12).mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, label="Actual")
        ax.plot(rolling_mean, label="Moving Average (12 bulan)")
        ax.set_title("Moving Average Jumlah Sampah")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.markdown(
            """
            - 2017–2020 tren meningkat
            - 2020–2021 relatif stabil
            - 2021–2022 mulai menurun
            - 2023–2024 penurunan cukup tajam
            """
        )

    with tab3:
        st.subheader("Boxplot per Tahun")
        df_box = df.copy()
        df_box["tahun_plot"] = df_box.index.year

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x="tahun_plot", y="jumlah_sampah", data=df_box, ax=ax)
        ax.set_title("Distribusi Jumlah Sampah per Tahun")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.grid(axis="y")
        st.pyplot(fig)

        st.markdown(
            """
            - 2017–2019 median meningkat
            - 2020–2022 median relatif stabil
            - 2023–2024 median turun signifikan
            """
        )

    with tab4:
        st.subheader("Seasonal Decomposition")
        decomp = seasonal_decompose(ts, model="additive", period=12)
        fig = decomp.plot()
        fig.set_size_inches(12, 8)
        st.pyplot(fig)

        st.markdown(
            """
            - Tren meningkat hingga sekitar 2020 lalu menurun
            - Pola musiman ada, tetapi tidak terlalu dominan
            - Residual menunjukkan beberapa spike
            """
        )


elif menu == "Stasioneritas":
    st.title("Uji Stasioneritas")

    st.subheader("ADF Test pada Data Train")
    adf_result = adfuller(train)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
    with col2:
        st.metric("p-value", f"{adf_result[1]:.6f}")

    if adf_result[1] < 0.05:
        st.success("Data train stasioner secara statistik.")
    else:
        st.warning("Data train tidak stasioner secara statistik.")

    st.write(
        "Secara visual masih terdapat tren, sehingga differencing tetap digunakan."
    )

    st.subheader("Differencing")
    ts_diff = train.diff().dropna()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts_diff)
    ax.set_title("Differenced Series")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Perubahan")
    ax.grid(True)
    st.pyplot(fig)

    st.write(
        """
        Fluktuasi sudah berada di sekitar nol dan tren utama telah berkurang,
        sehingga data lebih siap untuk pemodelan.
        """
    )

    st.subheader("ACF dan PACF")
    col1, col2 = st.columns(2)

    with col1:
        fig_acf, ax_acf = plt.subplots(figsize=(6, 4))
        plot_acf(ts_diff, ax=ax_acf)
        st.pyplot(fig_acf)

    with col2:
        fig_pacf, ax_pacf = plt.subplots(figsize=(6, 4))
        plot_pacf(ts_diff, ax=ax_pacf)
        st.pyplot(fig_pacf)

    st.write(
        "Spike signifikan pada lag awal menunjukkan kandidat awal model ARIMA(1,1,1)."
    )


elif menu == "Modeling":
    st.title("Pemodelan")

    sub_menu = st.radio("Pilih Analisis", ["ARIMA", "SARIMA", "Perbandingan"])

    if sub_menu == "ARIMA":
        st.subheader("Pemilihan Parameter ARIMA")
        best_order, best_aic = search_best_arima(train)

        st.write("Best ARIMA:", best_order)
        st.write("Best AIC:", round(best_aic, 3))

        model_arima = ARIMA(train, order=best_order)
        result_arima = model_arima.fit()

        st.subheader("Summary ARIMA")
        st.text(str(result_arima.summary()))

    elif sub_menu == "SARIMA":
        st.subheader("Pemilihan Parameter SARIMA")
        best_order_s, best_seasonal_s, best_aic_s = search_best_sarima(train)

        st.write("Best SARIMA:", best_order_s, best_seasonal_s)
        st.write("Best AIC:", round(best_aic_s, 3))

        model_sarima = SARIMAX(
            train,
            order=best_order_s,
            seasonal_order=best_seasonal_s,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result_sarima = model_sarima.fit(disp=False)

        st.subheader("Summary SARIMA")
        st.text(str(result_sarima.summary()))

    else:
        best_order, best_aic = search_best_arima(train)
        best_order_s, best_seasonal_s, best_aic_s = search_best_sarima(train)

        compare_df = pd.DataFrame({
            "Model": ["ARIMA", "SARIMA"],
            "Order": [str(best_order), str(best_order_s)],
            "Seasonal Order": ["-", str(best_seasonal_s)],
            "AIC": [best_aic, best_aic_s]
        })
        st.dataframe(compare_df, use_container_width=True)


elif menu == "Forecasting":
    st.title("Forecasting")

    model_option = st.radio("Pilih Model Forecast", ["ARIMA", "SARIMA"])

    best_order, _ = search_best_arima(train)
    best_order_s, best_seasonal_s, _ = search_best_sarima(train)

    if model_option == "ARIMA":
        model_arima = ARIMA(train, order=best_order)
        result_arima = model_arima.fit()
        forecast = result_arima.forecast(steps=len(test))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(train, label="Train")
        ax.plot(test, label="Actual")
        ax.plot(forecast, label="Forecast ARIMA")
        ax.set_title("Forecast ARIMA vs Actual")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    else:
        model_sarima = SARIMAX(
            train,
            order=best_order_s,
            seasonal_order=best_seasonal_s,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result_sarima = model_sarima.fit(disp=False)
        forecast_s = result_sarima.forecast(steps=len(test))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(train, label="Train")
        ax.plot(test, label="Actual")
        ax.plot(forecast_s, label="Forecast SARIMA")
        ax.set_title("Forecast SARIMA vs Actual")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


elif menu == "Evaluasi":
    st.title("Evaluasi Model")

    best_order, _ = search_best_arima(train)
    best_order_s, best_seasonal_s, _ = search_best_sarima(train)

    model_arima = ARIMA(train, order=best_order)
    result_arima = model_arima.fit()
    forecast_arima = result_arima.forecast(steps=len(test))
    mae_a, rmse_a, mape_a, r2_a = compute_metrics(test, forecast_arima)

    model_sarima = SARIMAX(
        train,
        order=best_order_s,
        seasonal_order=best_seasonal_s,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result_sarima = model_sarima.fit(disp=False)
    forecast_sarima = result_sarima.forecast(steps=len(test))
    mae_s, rmse_s, mape_s, r2_s = compute_metrics(test, forecast_sarima)

    result_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA"],
        "MAE": [mae_a, mae_s],
        "RMSE": [rmse_a, rmse_s],
        "MAPE": [mape_a, mape_s],
        "R2": [r2_a, r2_s],
    })

    st.subheader("Perbandingan Metrik")
    st.dataframe(result_df, use_container_width=True)

    if mape_a < mape_s:
        st.success("Model terbaik berdasarkan hasil evaluasi adalah ARIMA.")
        st.write(
            "ARIMA memberikan error lebih rendah dan prediksi lebih stabil pada data uji."
        )
    else:
        st.success("Model terbaik berdasarkan hasil evaluasi adalah SARIMA.")
        st.write(
            "SARIMA memberikan error lebih rendah dan lebih baik dalam menangkap pola musiman."
        )