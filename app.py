import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose

# -----------------------------
# SAYFA AYARI + STİL
# -----------------------------
st.set_page_config(page_title="Hisse Senedi Analiz Paneli", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
h1 {letter-spacing: 0.5px;}
.small-note {opacity: 0.8; font-size: 0.9rem;}
.card {background: rgba(255,255,255,0.03); padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08);}
hr {opacity: 0.25;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Hisse Senedi Zaman Serileri Analiz Paneli")
st.caption("Kapanış • Hacim • Getiri • Volatilite • Korelasyon • Teknik Analiz • Event Study (Temettü/Split)")

# -----------------------------
# YARDIMCI: CSV OKU
# -----------------------------
@st.cache_data
def load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df

@st.cache_data
def load_event_data(path: str) -> pd.DataFrame:
    try:
        ev = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if "Date" not in ev.columns:
        return pd.DataFrame()

    ev["Date"] = pd.to_datetime(ev["Date"], errors="coerce")
    ev = ev.dropna(subset=["Date"])
    # kolonları standartlaştır
    for col in ["Ticker", "EventType", "Dividend", "SplitFactor"]:
        if col not in ev.columns:
            ev[col] = np.nan
    ev["EventType"] = ev["EventType"].astype(str)
    return ev.sort_values("Date")

# -----------------------------
# YARDIMCI: FORMAT ALGILAMA (2 FORMAT)
# -----------------------------
def build_map_ticker_style(df_):
    # AKBNK.IS_Close / AKBNK.IS_Volume
    close_cols = [c for c in df_.columns if c.endswith("_Close")]
    vol_cols   = [c for c in df_.columns if c.endswith("_Volume")]

    def t_from(c): return c.replace("_Close","").replace("_Volume","")

    close_map = {t_from(c): c for c in close_cols}
    vol_map   = {t_from(c): c for c in vol_cols}
    common = sorted(set(close_map) & set(vol_map))
    return common, close_map, vol_map, "ticker_style"

def build_map_numbered_style(df_):
    # Close, Close.1, Close.2 ... / Volume, Volume.1 ...
    tickers = ["AKBNK.IS", "ASELS.IS", "THYAO.IS", "TUPRS.IS"]

    def sort_num_suffix(col):
        if "." not in col:
            return 0
        try:
            return int(col.split(".")[-1])
        except:
            return 9999

    close_candidates = [c for c in df_.columns if c == "Close" or c.startswith("Close.")]
    vol_candidates   = [c for c in df_.columns if c == "Volume" or c.startswith("Volume.")]

    close_candidates = sorted(close_candidates, key=sort_num_suffix)
    vol_candidates   = sorted(vol_candidates, key=sort_num_suffix)

    n = min(len(tickers), len(close_candidates), len(vol_candidates))
    tickers = tickers[:n]

    close_map = {tickers[i]: close_candidates[i] for i in range(n)}
    vol_map   = {tickers[i]: vol_candidates[i]   for i in range(n)}
    common = tickers
    return common, close_map, vol_map, "numbered_style"

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def annualized_vol(returns: pd.Series) -> float:
    # Günlük -> yıllık yaklaşık 252 işlem günü
    return float(returns.std() * np.sqrt(252))

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Ayarlar")
price_path = st.sidebar.text_input("Fiyat verisi (CSV)", value="bist_multi.csv")
event_path = st.sidebar.text_input("Event verisi (CSV)", value="events_2015_2025.csv")

df = load_price_data(price_path)
ev = load_event_data(event_path)

with st.sidebar.expander("📂 Veri sütunları"):
    st.write(list(df.columns))

# mapping oluştur
common, close_map, vol_map, fmt = build_map_ticker_style(df)
if not common:
    common, close_map, vol_map, fmt = build_map_numbered_style(df)

if not common:
    st.error("❌ Close/Volume eşleşmesi bulunamadı. CSV sütunlarını kontrol et.")
    st.stop()

st.sidebar.success(f"CSV formatı algılandı: {fmt}")

st.sidebar.header("📌 Filtreler")
selected_ticker = st.sidebar.selectbox("Hisse seç:", common)

close_col = close_map[selected_ticker]
vol_col = vol_map[selected_ticker]

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Tarih aralığı seç:",
    (min_date, max_date),
    min_value=min_date, max_value=max_date
)

# filtre data
mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
data = df.loc[mask, ["Date", close_col, vol_col]].copy()
data[close_col] = safe_numeric(data[close_col])
data[vol_col] = safe_numeric(data[vol_col])
data = data.dropna(subset=[close_col]).sort_values("Date")

if data.empty:
    st.warning("Seçtiğiniz tarih aralığında veri yok.")
    st.stop()

# return
data["ret"] = data[close_col].pct_change()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏁 Özet", "📈 Fiyat & Hacim", "📊 Getiri & Volatilite",
    "🔗 Korelasyon", "🧠 Teknik Analiz", "🎯 Event Study"
])

# -----------------------------
# TAB 1: ÖZET
# -----------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    last_price = float(data[close_col].iloc[-1])
    vol_y = annualized_vol(data["ret"].dropna())
    mean_ret = float(data["ret"].dropna().mean())
    nobs = int(data.shape[0])

    c1.metric("Seçilen Hisse", selected_ticker)
    c2.metric("Son Kapanış", f"{last_price:,.2f}")
    c3.metric("Yıllıklaştırılmış Volatilite", f"{vol_y:.3f}")
    c4.metric("Gözlem Sayısı", f"{nobs}")

    st.markdown("---")

    st.write(
        f"Bu panel, **{selected_ticker}** için kapanış fiyatı ve işlem hacmi üzerinden "
        f"getiri, volatilite, korelasyon ve teknik göstergeleri üretir. "
        f"Ek olarak **temettü / split** olayları için Event Study yaklaşımıyla AAR/CAAR analizi sunar."
    )
    st.markdown('<div class="small-note">Not: Getiri = pct_change, Volatilite = std(getiri)×√252</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# TAB 2: FİYAT & HACİM
# -----------------------------
with tab2:
    colA, colB = st.columns(2)
    with colA:
        st.subheader(f"{selected_ticker} – Kapanış Fiyatı")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(data["Date"], data[close_col], linewidth=2)
        ax.set_xlabel("Tarih"); ax.set_ylabel("Fiyat")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with colB:
        st.subheader(f"{selected_ticker} – İşlem Hacmi")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(data["Date"], data[vol_col], linewidth=1)
        ax.set_xlabel("Tarih"); ax.set_ylabel("Hacim")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("📄 Filtrelenmiş Ham Veri")
    st.dataframe(data.tail(50))

# -----------------------------
# TAB 3: GETİRİ & VOLATİLİTE
# -----------------------------
with tab3:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Günlük Getiri (Zaman Serisi)")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(data["Date"], data["ret"], linewidth=1)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Tarih"); ax.set_ylabel("Getiri")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    with colB:
        st.subheader("Getiri Dağılımı (KDE)")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.kdeplot(data["ret"].dropna(), fill=True, ax=ax)
        ax.set_xlabel("Günlük Getiri"); ax.set_ylabel("Yoğunluk")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    st.subheader("Volatilite Karşılaştırması (Seçili Aralık - 4 Hisse)")
    # tüm hisseler için volatilite (varsa)
    vols = []
    for t in common:
        ccol = close_map[t]
        s = df.loc[mask, ccol].copy()
        s = safe_numeric(s)
        r = s.pct_change().dropna()
        if len(r) > 10:
            vols.append((t, annualized_vol(r)))

    if vols:
        vol_df = pd.DataFrame(vols, columns=["Ticker", "AnnualizedVol"]).sort_values("AnnualizedVol", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(vol_df["Ticker"], vol_df["AnnualizedVol"])
        ax.set_ylabel("Volatilite")
        ax.set_xlabel("Hisse")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        st.dataframe(vol_df)
    else:
        st.info("Volatilite karşılaştırması için yeterli veri bulunamadı.")

# -----------------------------
# TAB 4: KORELASYON
# -----------------------------
with tab4:
    st.subheader("Getiri Korelasyonu (Seçili Hisseler)")
    # ortak tarih aralığında return matrisi
    ret_mat = pd.DataFrame({"Date": df.loc[mask, "Date"].values})
    for t in common:
        ccol = close_map[t]
        s = safe_numeric(df.loc[mask, ccol])
        ret_mat[t] = s.pct_change().values

    corr = ret_mat.drop(columns=["Date"]).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.dataframe(corr)

# -----------------------------
# TAB 5: TEKNİK ANALİZ
# -----------------------------
with tab5:
    st.subheader("Hareketli Ortalamalar (SMA30 / SMA90)")
    data["SMA30"] = data[close_col].rolling(30).mean()
    data["SMA90"] = data[close_col].rolling(90).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["Date"], data[close_col], label="Close", linewidth=2)
    ax.plot(data["Date"], data["SMA30"], label="SMA30", linewidth=2)
    ax.plot(data["Date"], data["SMA90"], label="SMA90", linewidth=2)
    ax.set_xlabel("Tarih"); ax.set_ylabel("Fiyat")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Trend / Seasonal / Residual Ayrıştırması (Opsiyonel)")
    period = st.slider("Sezon uzunluğu (period)", min_value=5, max_value=60, value=30, step=1)

    try:
        # seasonal_decompose için index datetime olmalı
        tmp = data.set_index("Date")[close_col].dropna()
        decomp = seasonal_decompose(tmp, model="additive", period=period)

        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(411); ax1.plot(decomp.observed); ax1.set_title("Observed")
        ax2 = fig.add_subplot(412); ax2.plot(decomp.trend); ax2.set_title("Trend")
        ax3 = fig.add_subplot(413); ax3.plot(decomp.seasonal); ax3.set_title("Seasonal")
        ax4 = fig.add_subplot(414); ax4.plot(decomp.resid); ax4.set_title("Residual")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Ayrıştırma bu aralıkta çalışmadı: {e}")

# -----------------------------
# TAB 6: EVENT STUDY (AAR / CAAR)
# -----------------------------
with tab6:
    st.subheader("Event Study: Temettü & Split Etkisi (AAR / CAAR)")
    st.write("Olay günü etrafında (±10 gün) anormal getiri yaklaşımıyla kümülatif etki hesaplanır.")

    if ev.empty or ev["Date"].isna().all():
        st.warning("Event verisi bulunamadı. events_2015_2025.csv dosyasını aynı klasöre koy.")
        st.stop()

    # Seçilen tarih aralığıyla eventleri filtrele
    ev2 = ev.copy()
    ev2 = ev2[(ev2["Date"] >= pd.to_datetime(start_date)) & (ev2["Date"] <= pd.to_datetime(end_date))]

    # sadece paneldeki hisseler + seçili hisse ile uyum
    ev2 = ev2[ev2["Ticker"].isin(common)]
    if ev2.empty:
        st.info("Seçili tarih aralığında event bulunamadı.")
        st.stop()

    # parametre
    window = st.slider("Event window (± gün)", 3, 20, 10, 1)

    # getiri matrisi (tüm hisseler)
    px = pd.DataFrame({"Date": df.loc[mask, "Date"].values})
    for t in common:
        ccol = close_map[t]
        s = safe_numeric(df.loc[mask, ccol])
        px[t] = s.values
    px = px.dropna(subset=["Date"]).sort_values("Date")

    ret = px.copy()
    for t in common:
        ret[t] = px[t].pct_change()
    ret = ret.dropna()

    # basit benchmark = ortalama piyasa (ortalama getiri)
    ret["MKT"] = ret[common].mean(axis=1)

    # AAR: r_it - MKT (basit anormal getiri)
    def compute_event_paths(ev_tbl):
        rows = []
        date_to_idx = {d: i for i, d in enumerate(ret["Date"].values)}

        for _, row in ev_tbl.iterrows():
            d = row["Date"]
            t = row["Ticker"]
            et = row["EventType"]

            # tam eşleşme yoksa en yakın önceki işlem günü
            # (event tarihi hafta sonu olabilir)
            d0 = d
            while d0 not in date_to_idx and d0 > ret["Date"].min():
                d0 = d0 - pd.Timedelta(days=1)

            if d0 not in date_to_idx:
                continue

            idx0 = date_to_idx[d0]
            for rel in range(-window, window + 1):
                idx = idx0 + rel
                if idx < 0 or idx >= len(ret):
                    continue
                ar = ret.iloc[idx][t] - ret.iloc[idx]["MKT"]
                rows.append({
                    "Ticker": t,
                    "EventType": et,
                    "EventDate": d0,
                    "rel_day": rel,
                    "AR": ar
                })

        return pd.DataFrame(rows)

    ar_df = compute_event_paths(ev2)

    if ar_df.empty:
        st.info("Event penceresinde hesaplanacak veri bulunamadı.")
        st.stop()

    # AAR & CAAR hesapla
    # Ticker bazında
    tick_agg = (ar_df.groupby(["Ticker", "rel_day"])
                .agg(AAR=("AR", "mean"), N_events=("AR", "count"))
                .reset_index())
    tick_agg["CAAR"] = tick_agg.groupby("Ticker")["AAR"].cumsum()

    # EventType bazında
    type_agg = (ar_df.groupby(["EventType", "rel_day"])
                .agg(AAR=("AR", "mean"), N_events=("AR", "count"))
                .reset_index())
    type_agg["CAAR"] = type_agg.groupby("EventType")["AAR"].cumsum()

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Hisseler Bazında Ortalama CAAR")
        fig, ax = plt.subplots(figsize=(10, 4))
        for t in tick_agg["Ticker"].unique():
            tmp = tick_agg[tick_agg["Ticker"] == t]
            ax.plot(tmp["rel_day"], tmp["CAAR"], marker="o", label=t)
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Olay Gününe Göre Gün (rel_day)")
        ax.set_ylabel("Kümülatif Ortalama Anormal Getiri (CAAR)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with colB:
        st.subheader("EventType (Dividend/Split) Bazında Ortalama CAAR")
        fig, ax = plt.subplots(figsize=(10, 4))
        for etype in type_agg["EventType"].unique():
            tmp = type_agg[type_agg["EventType"] == etype]
            ax.plot(tmp["rel_day"], tmp["CAAR"], marker="o", label=etype)
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Olay Gününe Göre Gün (rel_day)")
        ax.set_ylabel("Kümülatif Ortalama Anormal Getiri (CAAR)")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    st.subheader("Event Tablosu (Seçili aralık)")
    st.dataframe(ev2.sort_values("Date").reset_index(drop=True))

    st.subheader("AAR/CAAR (Ticker bazında) – İlk Satırlar")
    st.dataframe(tick_agg.head(30))

    st.subheader("AAR/CAAR (EventType bazında) – İlk Satırlar")
    st.dataframe(type_agg.head(30))



