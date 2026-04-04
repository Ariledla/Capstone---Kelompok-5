import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIG
# =========================================================
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


# =========================================================
# FUNCTIONS
# =========================================================
@st.cache_data
def load_data():
    return pd.read_excel(FILE_PATH)


@st.cache_data
def preprocess_data(df):
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


def split_train_test(ts, test_size=12):
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]
    return train, test


def calculate_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)
    return mae, rmse, mape, r2


@st.cache_data
def search_best_arima(train):
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
def search_best_sarima(train):
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


def get_fitted_model(model_name, train):
    if model_name == "ARIMA":
        best_order, best_aic = search_best_arima(train)
        fitted = ARIMA(train, order=best_order).fit()
        return fitted, best_order, None, best_aic
    else:
        best_order, best_seasonal, best_aic = search_best_sarima(train)
        fitted = SARIMAX(
            train,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        return fitted, best_order, best_seasonal, best_aic


def future_index(last_date, periods):
    return pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS"
    )


# =========================================================
# LOAD DATA
# =========================================================
try:
    raw_df = load_data()
    df, ts = preprocess_data(raw_df)
    train, test = split_train_test(ts, test_size=12)
except Exception as e:
    st.error(f"Gagal membaca file Excel: {e}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Beranda",
        "Understanding Data",
        "EDA",
        "Pemodelan",
        "Forecasting",
        "Evaluasi"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write(f"Jumlah observasi: {len(ts)}")
st.sidebar.write(f"Periode awal: {ts.index.min().date()}")
st.sidebar.write(f"Periode akhir: {ts.index.max().date()}")


# =========================================================
# BERANDA
# =========================================================
if menu == "Beranda":
    st.title("Prediksi Jumlah Sampah Kota Bandung")
    st.write(
        "Aplikasi ini digunakan untuk melakukan analisis time series dan prediksi jumlah sampah "
        "Kota Bandung berdasarkan data historis bulanan periode 2017–2024."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Data", len(ts))
    c2.metric("Periode Awal", str(ts.index.min().date()))
    c3.metric("Periode Akhir", str(ts.index.max().date()))

    st.subheader("Tujuan Aplikasi")
    st.markdown(
        """
        - Memahami pola historis jumlah sampah
        - Melakukan eksplorasi data time series
        - Menguji stasioneritas data
        - Membandingkan model ARIMA dan SARIMA
        - Menentukan model terbaik
        - Melakukan prediksi ke periode berikutnya
        """
    )

    st.subheader("Preview Data")
    st.dataframe(df.head(10), use_container_width=True)


# =========================================================
# UNDERSTANDING DATA
# =========================================================
elif menu == "Understanding Data":
    st.title("Understanding Data")

    st.subheader("Load")
    st.dataframe(raw_df.sample(min(5, len(raw_df))), use_container_width=True)

    st.subheader("Understanding")
    st.dataframe(df[["jumlah_sampah"]].describe().T, use_container_width=True)

    info_df = pd.DataFrame({
        "Kolom": raw_df.columns,
        "Tipe Data": raw_df.dtypes.astype(str).values,
        "Non-Null Count": raw_df.notnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Missing Values:", int(raw_df.isnull().sum().sum()))
    with col2:
        st.write("Duplicates Values:", int(raw_df.duplicated().sum()))

    st.subheader("Preprocessing")
    st.code(
        "df['bulan_num'] = df['bulan'].astype(str).str.upper().map(bulan_map)\n"
        "df['tanggal'] = pd.to_datetime(df['tahun'].astype(str) + '-' + df['bulan_num'])\n"
        "df = df.set_index('tanggal').sort_values('tanggal')\n"
        "ts = df['jumlah_sampah'].astype(float).copy()",
        language="python"
    )

    st.dataframe(df.head(), use_container_width=True)

    st.info(
        "Hasil preprocessing menunjukkan bahwa dataset memiliki 96 observasi bulanan dari Januari 2017 "
        "hingga Desember 2024. Tidak ditemukan missing values maupun data duplikat. Variabel jumlah_sampah "
        "dijadikan time series utama dengan indeks tanggal bulanan sehingga siap digunakan untuk analisis."
    )


# =========================================================
# EDA
# =========================================================
elif menu == "EDA":
    st.title("EDA")

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

        st.caption(
            "2017 → awal relatif rendah (~25–30k)\n\n"
            "2017–2020 → tren meningkat\n\n"
            "2020–2022 → relatif stabil (~40k)\n\n"
            "2023–2024 → penurunan signifikan + volatilitas tinggi\n\n"
            "→ terdapat tren → data tidak stasioner"
        )

    with tab2:
        st.subheader("Moving Average")
        st.write("Digunakan untuk melihat tren jangka panjang dengan menghaluskan fluktuasi data.")

        rolling_mean = ts.rolling(window=12).mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(ts, label="Actual")
        ax.plot(rolling_mean, label="Moving Average (12 bulan)", color="red")
        ax.set_title("Moving Average Jumlah Sampah")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.caption(
            "2017–2020 → tren meningkat\n\n"
            "2020–2021 → relatif stabil\n\n"
            "2021–2022 → mulai menurun\n\n"
            "2023–2024 → penurunan cukup tajam\n\n"
            "→ tren berubah → data tidak stasioner"
        )

    with tab3:
        st.subheader("Boxplot per Tahun")
        st.write("Digunakan untuk membandingkan distribusi jumlah sampah tiap tahun, termasuk median, sebaran data, dan potensi outlier.")

        df_box = df.copy()
        df_box["tahun_plot"] = df_box.index.year
        years = sorted(df_box["tahun_plot"].unique())
        data_by_year = [df_box[df_box["tahun_plot"] == y]["jumlah_sampah"].values for y in years]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data_by_year, labels=years)
        ax.set_title("Distribusi Jumlah Sampah per Tahun")
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Ton")
        ax.grid(axis="y")
        st.pyplot(fig)

        st.caption(
            "2017–2019 → median meningkat + distribusi relatif stabil\n\n"
            "2020–2022 → median cenderung stabil\n\n"
            "2023–2024 → median turun signifikan + distribusi melebar\n\n"
            "2021 dan 2024 terlihat outlier\n\n"
            "Outlier tidak dibuang karena dapat merepresentasikan kondisi nyata."
        )

    with tab4:
        st.subheader("Seasonal Decomposition")
        st.write("Digunakan untuk memisahkan komponen time series menjadi tren, musiman, dan residual.")

        decomp = seasonal_decompose(ts, model="additive", period=12)
        fig = decomp.plot()
        fig.set_size_inches(10, 8)
        st.pyplot(fig)

        st.caption(
            "Trend → meningkat hingga ~2020 lalu menurun hingga 2024\n\n"
            "Seasonal → pola berulang ada tapi tidak terlalu kuat\n\n"
            "Residual → fluktuasi acak + ada beberapa spike"
        )


# =========================================================
# MODELING
# =========================================================
elif menu == "Pemodelan":
    st.title("Pemodelan")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Train-Test Split",
        "Uji Stasioneritas",
        "ARIMA",
        "SARIMA"
    ])

    with tab1:
        st.subheader("Train-Test Split")
        st.write("Data dibagi menjadi data latih (train) dan data uji (test) untuk mengevaluasi performa model forecasting.")

        c1, c2 = st.columns(2)
        c1.metric("Jumlah Data Train", len(train))
        c2.metric("Jumlah Data Test", len(test))

        split_df = pd.DataFrame({
            "Set": ["Train", "Test"],
            "Jumlah": [len(train), len(test)],
            "Periode Awal": [train.index.min().date(), test.index.min().date()],
            "Periode Akhir": [train.index.max().date(), test.index.max().date()]
        })
        st.dataframe(split_df, use_container_width=True)

        st.caption(
            "Train → data historis utama\n\n"
            "Test → 12 bulan terakhir\n\n"
            "→ digunakan untuk evaluasi model"
        )

    with tab2:
        st.subheader("Uji Stasioneritas (ADF Test)")
        st.write("Digunakan untuk menguji apakah data time series bersifat stasioner atau tidak.")

        adf_result = adfuller(train)

        c1, c2 = st.columns(2)
        c1.metric("ADF Statistic", f"{adf_result[0]:.6f}")
        c2.metric("p-value", f"{adf_result[1]:.6f}")

        if adf_result[1] < 0.05:
            st.success("p-value < 0.05 → data stasioner secara statistik")
        else:
            st.warning("p-value > 0.05 → data tidak stasioner")

        st.info(
            "Secara visual masih terdapat tren, sehingga differencing tetap digunakan pada pemodelan."
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

        st.caption(
            "Fluktuasi sudah di sekitar nol\n\n"
            "Tidak terlihat tren naik/turun yang jelas\n\n"
            "Masih terdapat spike pada beberapa periode\n\n"
            "→ tren hilang → data lebih stasioner"
        )

        st.subheader("ACF dan PACF")
        st.write("Digunakan untuk menentukan parameter p (AR) dan q (MA) pada model ARIMA.")

        col1, col2 = st.columns(2)

        with col1:
            fig_acf, ax_acf = plt.subplots(figsize=(6, 4))
            plot_acf(ts_diff, ax=ax_acf)
            st.pyplot(fig_acf)

        with col2:
            fig_pacf, ax_pacf = plt.subplots(figsize=(6, 4))
            plot_pacf(ts_diff, ax=ax_pacf)
            st.pyplot(fig_pacf)

        st.caption(
            "ACF → spike signifikan di lag 1, lalu cepat mendekati nol\n\n"
            "PACF → spike signifikan di lag 1, sisanya kecil\n\n"
            "→ kandidat model awal: ARIMA(1,1,1)"
        )

    with tab3:
        st.subheader("Pemilihan Parameter ARIMA")
        st.write("Dilakukan pencarian kombinasi parameter terbaik untuk model ARIMA berdasarkan nilai AIC terendah.")

        best_order, best_aic = search_best_arima(train)

        st.write(f"Best ARIMA: {best_order}")
        st.write(f"Best AIC: {best_aic:.3f}")

        st.caption(
            f"Model terbaik: ARIMA{best_order}\n\n"
            f"AIC paling kecil: {best_aic:.3f}\n\n"
            "→ model ini paling optimal untuk kandidat ARIMA"
        )

        st.subheader("Pemodelan ARIMA")
        result_arima = ARIMA(train, order=best_order).fit()
        st.text(str(result_arima.summary()))

    with tab4:
        st.subheader("Pemilihan Parameter SARIMA")
        st.write("Dilakukan pencarian kombinasi parameter terbaik untuk model SARIMA berdasarkan nilai AIC terendah.")

        best_order_s, best_seasonal_s, best_aic_s = search_best_sarima(train)

        st.write(f"Best SARIMA: {best_order_s} {best_seasonal_s}")
        st.write(f"Best AIC: {best_aic_s:.3f}")

        st.caption(
            f"Model terbaik: SARIMA{best_order_s}{best_seasonal_s}\n\n"
            f"AIC paling kecil: {best_aic_s:.3f}\n\n"
            "→ penambahan komponen musiman meningkatkan kemampuan model dalam menangkap pola data"
        )

        st.subheader("Pemodelan SARIMA")
        result_sarima = SARIMAX(
            train,
            order=best_order_s,
            seasonal_order=best_seasonal_s,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        st.text(str(result_sarima.summary()))


# =========================================================
# FORECASTING
# =========================================================
elif menu == "Forecasting":
    st.title("Forecasting")
    st.write("Halaman ini digunakan untuk melakukan prediksi menggunakan model ARIMA atau SARIMA.")

    if "forecast_mode" not in st.session_state:
        st.session_state["forecast_mode"] = "Prediksi Masa Depan"
    if "forecast_model" not in st.session_state:
        st.session_state["forecast_model"] = "ARIMA"
    if "forecast_horizon" not in st.session_state:
        st.session_state["forecast_horizon"] = 6

    st.subheader("Input Forecasting")

    with st.form("forecast_form"):
        col1, col2 = st.columns(2)

        with col1:
            forecast_mode = st.selectbox(
                "Pilih jenis prediksi",
                ["Prediksi Masa Depan", "Uji Model (Data Test)"],
                index=["Prediksi Masa Depan", "Uji Model (Data Test)"].index(st.session_state["forecast_mode"])
            )

        with col2:
            model_choice = st.selectbox(
                "Pilih model",
                ["ARIMA", "SARIMA"],
                index=["ARIMA", "SARIMA"].index(st.session_state["forecast_model"])
            )

        st.session_state["forecast_mode"] = forecast_mode
        st.session_state["forecast_model"] = model_choice

        horizon = None
        if forecast_mode == "Prediksi Masa Depan":
            horizon = st.number_input(
                "Masukkan jumlah bulan yang ingin diprediksi",
                min_value=1,
                max_value=24,
                value=st.session_state["forecast_horizon"],
                step=1
            )
            st.session_state["forecast_horizon"] = horizon
        else:
            st.info("Mode ini otomatis menggunakan 12 bulan data test terakhir.")

        c1, c2 = st.columns(2)
        with c1:
            submit_forecast = st.form_submit_button("Enter / Jalankan Forecast", use_container_width=True)
        with c2:
            reset_forecast = st.form_submit_button("Refresh / Reset Input", use_container_width=True)

    if reset_forecast:
        st.session_state["forecast_mode"] = "Prediksi Masa Depan"
        st.session_state["forecast_model"] = "ARIMA"
        st.session_state["forecast_horizon"] = 6
        st.rerun()

    if submit_forecast:
        fitted_model, best_order, best_seasonal, best_aic = get_fitted_model(model_choice, train)

        st.subheader("Ringkasan Model")
        if model_choice == "ARIMA":
            st.write(f"Model terpilih: ARIMA{best_order}")
            st.write(f"AIC: {best_aic:.3f}")
        else:
            st.write(f"Model terpilih: SARIMA{best_order}{best_seasonal}")
            st.write(f"AIC: {best_aic:.3f}")

        if forecast_mode == "Uji Model (Data Test)":
            forecast = fitted_model.forecast(steps=len(test))

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(train, label="Train")
            ax.plot(test, label="Actual")
            ax.plot(forecast, label=f"Forecast {model_choice}")
            ax.set_title(f"{model_choice} Forecast vs Actual")
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Ton")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            result_test = pd.DataFrame({
                "Periode": test.index.strftime("%Y-%m"),
                "Actual": np.round(test.values, 2),
                "Forecast": np.round(forecast.values, 2)
            })
            st.dataframe(result_test, use_container_width=True)

        else:
            future_forecast = fitted_model.forecast(steps=horizon)
            idx_future = future_index(ts.index.max(), horizon)
            future_forecast.index = idx_future

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ts, label="Data Historis")
            ax.plot(future_forecast, label=f"Prediksi {model_choice}", color="green")
            ax.set_title(f"Prediksi Jumlah Sampah {horizon} Bulan ke Depan")
            ax.set_xlabel("Tahun")
            ax.set_ylabel("Ton")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            result_future = pd.DataFrame({
                "Periode": future_forecast.index.strftime("%Y-%m"),
                "Prediksi Jumlah Sampah": np.round(future_forecast.values, 2)
            })
            st.dataframe(result_future, use_container_width=True)


# =========================================================
# EVALUASI
# =========================================================
elif menu == "Evaluasi":
    st.title("Evaluasi Model")

    # ARIMA
    best_order, _ = search_best_arima(train)
    model_arima = ARIMA(train, order=best_order).fit()
    forecast_arima = model_arima.forecast(steps=len(test))
    mae_a, rmse_a, mape_a, r2_a = calculate_metrics(test, forecast_arima)

    # SARIMA
    best_order_s, best_seasonal_s, _ = search_best_sarima(train)
    model_sarima = SARIMAX(
        train,
        order=best_order_s,
        seasonal_order=best_seasonal_s,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    forecast_sarima = model_sarima.forecast(steps=len(test))
    mae_s, rmse_s, mape_s, r2_s = calculate_metrics(test, forecast_sarima)

    st.subheader("Evaluasi ARIMA")
    arima_eval = pd.DataFrame({
        "Metrik": ["MAE", "RMSE", "MAPE", "R²"],
        "Nilai": [mae_a, rmse_a, mape_a, r2_a]
    })
    st.dataframe(arima_eval, use_container_width=True)

    st.subheader("Evaluasi SARIMA")
    sarima_eval = pd.DataFrame({
        "Metrik": ["MAE", "RMSE", "MAPE", "R²"],
        "Nilai": [mae_s, rmse_s, mape_s, r2_s]
    })
    st.dataframe(sarima_eval, use_container_width=True)

    st.subheader("Perbandingan ARIMA dan SARIMA")
    compare_df = pd.DataFrame({
        "Model": ["ARIMA(0,1,1)", f"SARIMA{best_order_s}{best_seasonal_s}"],
        "MAE": [round(mae_a, 2), round(mae_s, 2)],
        "RMSE": [round(rmse_a, 2), round(rmse_s, 2)],
        "MAPE": [round(mape_a, 2), round(mape_s, 2)],
        "R²": [round(r2_a, 3), round(r2_s, 3)]
    })
    st.dataframe(compare_df, use_container_width=True)

    if mape_a < mape_s:
        st.success("Model terbaik berdasarkan hasil evaluasi adalah ARIMA(0,1,1).")
        st.write(
            "ARIMA memberikan error yang lebih rendah dan prediksi yang lebih stabil pada data uji."
        )
    else:
        st.success("Model terbaik berdasarkan hasil evaluasi adalah SARIMA.")
        st.write(
            "SARIMA memberikan error yang lebih rendah dan lebih baik dalam menangkap pola musiman."
        )
