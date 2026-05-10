import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import math
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="TNT HOUSE PRICING",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Global */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Header bar */
    .app-header {
        background: linear-gradient(135deg, #e84141 0%, #c0392b 100%);
        padding: 18px 24px 14px;
        border-radius: 0 0 18px 18px;
        margin: -1rem -1rem 1.2rem -1rem;
        box-shadow: 0 4px 20px rgba(232,65,65,0.25);
    }
    .app-header h1 {
        color: white; font-size: 1.6rem; font-weight: 700;
        margin: 0; letter-spacing: -0.3px;
    }
    .app-header p { color: rgba(255,255,255,0.85); font-size: 0.82rem; margin: 4px 0 0; }

    /* Cards */
    .card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 14px;
        border: 1px solid #f0f0f0;
    }
    .card-title {
        font-size: 0.78rem; font-weight: 700; color: #888;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
    }
    .card-value {
        font-size: 2rem; font-weight: 800; color: #e84141; line-height: 1.1;
    }
    .card-sub { font-size: 0.82rem; color: #666; margin-top: 4px; }

    /* Price result box */
    .price-result {
        background: linear-gradient(135deg, #e84141, #c0392b);
        color: white;
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        margin: 14px 0;
        box-shadow: 0 6px 24px rgba(232,65,65,0.3);
    }
    .price-result .label { font-size: 0.85rem; opacity: 0.9; margin-bottom: 6px; }
    .price-result .amount { font-size: 2.4rem; font-weight: 900; line-height: 1; }
    .price-result .range { font-size: 0.82rem; opacity: 0.8; margin-top: 6px; }

    /* Nav tabs */
    div[data-testid="stTabs"] button {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 8px 14px !important;
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: #fff3f3;
        color: #e84141;
        border: 1px solid #ffd0d0;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-green {
        background: #f0faf0; color: #27ae60; border-color: #b8e6b8;
    }
    .badge-blue {
        background: #f0f7ff; color: #2980b9; border-color: #b8d8f8;
    }

    /* Section header */
    .section-hd {
        font-size: 1rem; font-weight: 700; color: #222;
        margin: 16px 0 10px; padding-left: 10px;
        border-left: 4px solid #e84141;
    }

    /* Metric override */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #e84141 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .app-header h1 { font-size: 1.2rem; }
        .price-result .amount { font-size: 1.8rem; }
        .card-value { font-size: 1.5rem; }
    }

    /* Input labels */
    label { font-weight: 600 !important; font-size: 0.85rem !important; color: #444 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #e84141, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 12px 0 !important;
        width: 100% !important;
        box-shadow: 0 4px 14px rgba(232,65,65,0.35) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(232,65,65,0.45) !important;
    }
    div.row-widget.stRadio > div { flex-direction: row; flex-wrap: wrap; gap: 8px; }

    /* Selectbox */
    [data-baseweb="select"] { border-radius: 10px !important; }

    /* Slider */
    [data-testid="stSlider"] { padding: 4px 0; }

    /* Footer */
    .footer {
        text-align: center; color: #aaa; font-size: 0.75rem;
        padding: 20px 0 10px; border-top: 1px solid #f0f0f0; margin-top: 24px;
    }
</style>
""", unsafe_allow_html=True)


CAMPUS_COORDS = {
    "A": (10.785420370325223, 106.69513688271378),
    "B": (10.761263992568955, 106.668357995408),
    "C": (10.773365984090377, 106.67764631075153),
}
BINARY_COLS = ["may_lanh", "tu_lanh", "gio_tu_do", "wc_rieng",
               "ban_cong", "co_bep", "cho_de_xe", "gan_sieu_thi",
               "gan_quan_an", "gan_truong_hoc"]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


@st.cache_data
def load_data():
    df = pd.read_csv("dulieu_hoan_hao_final.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={"dien_tich": "area_m2"}, inplace=True)
    df = df.dropna(subset=["price", "latitude", "longitude"])
    df = df[df["price"] > 0]
    df = df[df["price"] <= 8_000_000]
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    df["quan"] = df["quan"].str.strip()
    df["phuong"] = df["phuong"].str.strip()
    df["amenity_score"] = df[BINARY_COLS].sum(axis=1)

    # Remove outliers (IQR method on price)
    q1, q3 = df["price"].quantile(0.05), df["price"].quantile(0.95)
    df = df[(df["price"] >= q1) & (df["price"] <= q3)].copy()

    # Add distance to each UEH campus
    for key, (clat, clon) in CAMPUS_COORDS.items():
        df[f"dist_ueh_{key}"] = df.apply(
            lambda r: haversine(r["latitude"], r["longitude"], clat, clon), axis=1
        )
    df["dist_nearest_ueh"] = df[[f"dist_ueh_{k}" for k in CAMPUS_COORDS]].min(axis=1)

    return df


def _build_features(df_input, df_ref):
    """Engineer features using statistics from reference dataframe."""
    df2 = df_input.copy()

    # Target encoding: mean price per district & ward (from reference data)
    quan_mean = df_ref.groupby("quan")["price"].mean().to_dict()
    quan_median = df_ref.groupby("quan")["price"].median().to_dict()
    phuong_mean = df_ref.groupby("phuong")["price"].mean().to_dict()

    global_mean = df_ref["price"].mean()
    df2["quan_price_mean"] = df2["quan"].map(quan_mean).fillna(global_mean)
    df2["quan_price_median"] = df2["quan"].map(quan_median).fillna(global_mean)
    df2["phuong_price_mean"] = df2["phuong"].map(phuong_mean).fillna(global_mean)

    # Label encode district & ward (ordinal)
    unique_quans = sorted(df_ref["quan"].unique())
    unique_phuongs = sorted(df_ref["phuong"].unique())
    quan_map = {q: i for i, q in enumerate(unique_quans)}
    phuong_map = {p: i for i, p in enumerate(unique_phuongs)}
    df2["quan_enc"] = df2["quan"].map(quan_map).fillna(-1)
    df2["phuong_enc"] = df2["phuong"].map(phuong_map).fillna(-1)

    # Amenity interactions
    df2["premium_combo"] = (
        df2["may_lanh"].fillna(0) + df2["wc_rieng"].fillna(0) +
        df2["ban_cong"].fillna(0) + df2["co_bep"].fillna(0)
    )
    df2["utility_combo"] = (
        df2["cho_de_xe"].fillna(0) + df2["gan_sieu_thi"].fillna(0) +
        df2["gan_quan_an"].fillna(0) + df2["gan_truong_hoc"].fillna(0)
    )
    # Area feature
    df2["area_m2"] = df2["area"]

# Distance-based features
    df2["lat_lon_interaction"] = df2["latitude"] * df2["longitude"]

    for k in CAMPUS_COORDS:
        col = f"dist_ueh_{k}"
        if col in df2.columns:
          df2[f"dist_ueh_{k}_sq"] = df2[col] ** 2

# Amenity square
    df2["amenity_sq"] = df2["amenity_score"] ** 2

    feature_cols = [
    "quan_enc", "phuong_enc",
    "latitude", "longitude", "lat_lon_interaction",
    "area_m2",
    "amenity_score", "amenity_sq",
    "premium_combo", "utility_combo",
    "dist_ueh_A", "dist_ueh_B", "dist_ueh_C",
    "dist_nearest_ueh",
    "dist_ueh_A_sq", "dist_ueh_B_sq", "dist_ueh_C_sq",
] + BINARY_COLS
    

    feature_cols = [c for c in feature_cols if c in df2.columns]
    return df2, feature_cols, quan_map, phuong_map, quan_mean, quan_median, phuong_mean


@st.cache_resource
def train_models(df):
    df2, feature_cols, quan_map, phuong_map, quan_mean, quan_median, phuong_mean = \
        _build_features(df, df)

    X = df2[feature_cols].values
    y_raw = df2["price"].values
    y = np.log1p(y_raw)   # Log-transform: stabilises variance, improves R²

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Base learners
    ridge = Ridge(alpha=5)
    ridge.fit(X_train_sc, y_train)

    gb = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )
    gb.fit(X_train_sc, y_train)

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, random_state=42
    )
    rf.fit(X_train_sc, y_train)

    # Stacking meta-learner (Ridge on top of base predictions)
    meta = Ridge(alpha=1)

    def stacked_predict(X_sc):
        p_ridge = ridge.predict(X_sc)
        p_gb = gb.predict(X_sc)
        p_rf = rf.predict(X_sc)
        return np.column_stack([p_ridge, p_gb, p_rf])

    meta_X_train = stacked_predict(X_train_sc)
    meta_X_test = stacked_predict(X_test_sc)
    meta.fit(meta_X_train, y_train)

    # Evaluate on test set (convert back from log scale)
    y_pred_log = meta.predict(meta_X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Individual model scores
    r2_ridge = r2_score(y_true, np.expm1(ridge.predict(X_test_sc)))
    r2_gb = r2_score(y_true, np.expm1(gb.predict(X_test_sc)))
    r2_rf = r2_score(y_true, np.expm1(rf.predict(X_test_sc)))

    return {
        "ridge": ridge, "gb": gb, "rf": rf, "meta": meta,
        "scaler": scaler, "feature_cols": feature_cols,
        "quan_map": quan_map, "phuong_map": phuong_map,
        "quan_mean": quan_mean, "quan_median": quan_median,
        "phuong_mean": phuong_mean,
        "r2": r2, "r2_ridge": r2_ridge, "r2_gb": r2_gb, "r2_rf": r2_rf,
        "mae": mae, "rmse": rmse,
        "binary_cols": BINARY_COLS
    }


def fuzzy_membership_area(area_m2):
    if area_m2 <= 15:
        small = 1.0
        medium = 0.0
        large = 0.0
    elif area_m2 <= 25:
        small = (25 - area_m2) / 10.0
        medium = (area_m2 - 15) / 10.0
        large = 0.0
    elif area_m2 <= 40:
        small = 0.0
        medium = (40 - area_m2) / 15.0
        large = (area_m2 - 25) / 15.0
    else:
        small = 0.0
        medium = 0.0
        large = 1.0
    return small, medium, large


def fuzzy_membership_amenity(score, max_score=10):
    ratio = score / max_score
    if ratio <= 0.3:
        low = 1.0
        medium = 0.0
        high = 0.0
    elif ratio <= 0.6:
        low = (0.6 - ratio) / 0.3
        medium = (ratio - 0.3) / 0.3
        high = 0.0
    elif ratio <= 0.8:
        low = 0.0
        medium = (0.8 - ratio) / 0.2
        high = (ratio - 0.6) / 0.2
    else:
        low = 0.0
        medium = 0.0
        high = 1.0
    return low, medium, high


def fuzzy_price_adjustment(area_m2, amenity_score):
    s_small, s_med, s_large = fuzzy_membership_area(area_m2)
    a_low, a_med, a_high = fuzzy_membership_amenity(amenity_score)

    rules = {
    0.95: min(s_small, a_low),
    0.98: min(s_small, a_med),
    1.00: min(s_small, a_high),

    0.98: min(s_med, a_low),
    1.00: min(s_med, a_med),
    1.03: min(s_med, a_high),

    1.00: min(s_large, a_low),
    1.03: min(s_large, a_med),
    1.05: min(s_large, a_high),
}

    numerator = sum(k * v for k, v in rules.items())
    denominator = sum(rules.values())
    if denominator == 0:
        return 1.0
    return numerator / denominator


UEH_CAMPUSES = {
    "UEH Cơ sở A (Nguyễn Đình Chiểu, Q.3)": {
        "lat":10.783293577128273,  "lon": 106.6946681954084,
        "address": "59C Nguyễn Đình Chiểu, P.6, Q.3",
        "color": "#e84141"
    },
    "UEH Cơ sở B (Nguyễn Tri Phương, Q.10)": {
        "lat": 10.761263992568955, "lon":  106.66833653773631,
        "address": "279 Nguyễn Tri Phương, P.5, Q.10",
        "color": "#2980b9"
    },
    "UEH Cơ sở C (3 tháng 2, quận 10)": {
        "lat": 10.77327112648688, "lon": 106.67760339540814,
        "address": "Võ Văn Ngân, P.Linh Chiểu, TP.Thủ Đức",
        "color": "#27ae60"
    },
}



RADIUS_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def predict_price(models, df, quan, phuong, amenities_dict, area_m2):
    quan_map = models["quan_map"]
    phuong_map = models["phuong_map"]
    quan_mean = models["quan_mean"]
    quan_median = models["quan_median"]
    phuong_mean = models["phuong_mean"]
    scaler = models["scaler"]
    feature_cols = models["feature_cols"]
    global_mean = df["price"].mean()

    amenity_score = sum(amenities_dict.values())

    # Use mean lat/lon for the chosen district/ward
    mask_q = df["quan"] == quan
    mask_p = df["phuong"] == phuong
    if mask_p.any():
        lat = df.loc[mask_p, "latitude"].mean()
        lon = df.loc[mask_p, "longitude"].mean()
    elif mask_q.any():
        lat = df.loc[mask_q, "latitude"].mean()
        lon = df.loc[mask_q, "longitude"].mean()
    else:
        lat = df["latitude"].mean()
        lon = df["longitude"].mean()

    # Build feature row matching training schema
    row = {
        "quan": quan, "phuong": phuong,
        "latitude": lat, "longitude": lon,
        "amenity_score": amenity_score,
        "price": global_mean,  # placeholder for target encoding reference
    }
    row.update(amenities_dict)

    # Computed features
    row["quan_enc"] = quan_map.get(quan, -1)
    row["phuong_enc"] = phuong_map.get(phuong, -1)
    row["quan_price_mean"] = quan_mean.get(quan, global_mean)
    row["quan_price_median"] = quan_median.get(quan, global_mean)
    row["phuong_price_mean"] = phuong_mean.get(phuong, global_mean)
    row["lat_lon_interaction"] = lat * lon
    row["amenity_sq"] = amenity_score ** 2
    row["premium_combo"] = (
        amenities_dict.get("may_lanh", 0) + amenities_dict.get("wc_rieng", 0) +
        amenities_dict.get("ban_cong", 0) + amenities_dict.get("co_bep", 0)
    )
    row["utility_combo"] = (
        amenities_dict.get("cho_de_xe", 0) + amenities_dict.get("gan_sieu_thi", 0) +
        amenities_dict.get("gan_quan_an", 0) + amenities_dict.get("gan_truong_hoc", 0)
    )

    # Distance features
    for key, (clat, clon) in CAMPUS_COORDS.items():
        d = haversine(lat, lon, clat, clon)
        row[f"dist_ueh_{key}"] = d
        row[f"dist_ueh_{key}_sq"] = d ** 2
    row["dist_nearest_ueh"] = min(
        row[f"dist_ueh_{k}"] for k in CAMPUS_COORDS
    )

    # Build feature vector in correct order
    x = np.array([[row.get(c, 0) for c in feature_cols]], dtype=float)
    x_sc = scaler.transform(x)

    # Stacked prediction (log scale → expm1 back)
    p_ridge = models["ridge"].predict(x_sc)[0]
    p_gb = models["gb"].predict(x_sc)[0]
    p_rf = models["rf"].predict(x_sc)[0]
    meta_x = np.array([[p_ridge, p_gb, p_rf]])
    pred_log = models["meta"].predict(meta_x)[0]
    base_price = np.expm1(pred_log)

    # Fuzzy adjustment for area (m²)
    fuzzy_factor = fuzzy_price_adjustment(area_m2, amenity_score)
    final_price = base_price * fuzzy_factor
    final_price = max(800_000, min(20_000_000, final_price))

    # Confidence interval from model RMSE
    rmse = models.get("rmse", final_price * 0.12)
    low = max(800_000, final_price - 0.8 * rmse)
    high = min(20_000_000, final_price + 0.8 * rmse)

    return final_price, low, high


def format_price(p):
    if p >= 1_000_000:
        return f"{p/1_000_000:.1f} triệu"
    return f"{p/1_000:.0f}K"


def make_map_with_radius(df, campus_name, radius_km, show_suggestions=True):
    campus = UEH_CAMPUSES[campus_name]
    clat, clon = campus["lat"], campus["lon"]

    m = folium.Map(
        location=[clat, clon],
        zoom_start=14,
        tiles="CartoDB positron",
        prefer_canvas=True
    )

    for name, info in UEH_CAMPUSES.items():
        icon_color = "red" if name == campus_name else "gray"
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=folium.Popup(
                f"<b>{name}</b><br>{info['address']}",
                max_width=220
            ),
            tooltip=name,
            icon=folium.Icon(color=icon_color, icon="graduation-cap",
                             prefix="fa")
        ).add_to(m)

    colors = ["#e84141", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#8b5cf6"]
    for i, r in enumerate(RADIUS_OPTIONS):
        if r <= radius_km:
            folium.Circle(
                location=[clat, clon],
                radius=r * 1000,
                color=colors[i % len(colors)],
                fill=True,
                fill_opacity=0.04,
                weight=2,
                dash_array="6 4",
                tooltip=f"Bán kính {r} km"
            ).add_to(m)

    if show_suggestions:
        df_nearby = df.copy()
        df_nearby["dist_km"] = df_nearby.apply(
            lambda row: haversine(clat, clon, row["latitude"], row["longitude"]),
            axis=1
        )
        df_nearby = df_nearby[df_nearby["dist_km"] <= radius_km].copy()
        df_nearby = df_nearby.sort_values("price")

        cluster = MarkerCluster(name="Nhà trọ gần đây").add_to(m)

        for _, row in df_nearby.iterrows():
            price_color = (
                "#27ae60" if row["price"] < 3_000_000 else
                "#f39c12" if row["price"] < 5_000_000 else
                "#e84141"
            )
            amenities = []
            if row.get("may_lanh"): amenities.append("❄️ Máy lạnh")
            if row.get("wc_rieng"): amenities.append("🚿 WC riêng")
            if row.get("ban_cong"): amenities.append("🏗️ Ban công")
            if row.get("co_bep"): amenities.append("🍳 Bếp")
            if row.get("cho_de_xe"): amenities.append("🅿️ Để xe")

            popup_html = f"""
            <div style="font-family:sans-serif;min-width:180px">
              <div style="background:{price_color};color:white;padding:8px 12px;
                          border-radius:8px 8px 0 0;font-weight:700;font-size:1rem">
                {format_price(row['price'])}/tháng
              </div>
              <div style="padding:10px 12px;background:white;border:1px solid #eee;
                          border-radius:0 0 8px 8px">
                <b>📍 {row['phuong']}, {row['quan']}</b><br>
                <span style="color:#888;font-size:0.82rem">
                  Cách {row['dist_km']:.1f} km
                </span><br>
                <div style="margin-top:6px;font-size:0.78rem">
                  {'  '.join(amenities) if amenities else 'Không có tiện ích ghi chú'}
                </div>
              </div>
            </div>
            """
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{format_price(row['price'])}/th – {row['phuong']}",
                icon=folium.DivIcon(
                    html=f"""<div style="
                      background:{price_color};color:white;
                      border-radius:20px;padding:4px 8px;
                      font-size:0.7rem;font-weight:700;white-space:nowrap;
                      box-shadow:0 2px 6px rgba(0,0,0,0.25);">
                      {format_price(row['price'])}
                    </div>""",
                    icon_size=(80, 28),
                    icon_anchor=(40, 14)
                )
            ).add_to(cluster)

        return m, df_nearby
    return m, pd.DataFrame()


def make_heatmap(df):
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="CartoDB dark_matter"
    )

    heat_data = [
        [row["latitude"], row["longitude"], row["price"] / 1_000_000]
        for _, row in df.iterrows()
        if not np.isnan(row["latitude"])
    ]

    HeatMap(
        heat_data,
        radius=22,
        blur=18,
        min_opacity=0.3,
        max_zoom=16,
        gradient={
            0.2: "#2563eb",
            0.4: "#22d3ee",
            0.6: "#a3e635",
            0.75: "#facc15",
            1.0: "#ef4444"
        }
    ).add_to(m)

    for name, info in UEH_CAMPUSES.items():
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=name,
            tooltip=name,
            icon=folium.Icon(color="white", icon="graduation-cap", prefix="fa")
        ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:999;
                background:rgba(15,15,15,0.85);color:white;padding:14px 18px;
                border-radius:12px;font-family:sans-serif;font-size:0.78rem;
                border:1px solid rgba(255,255,255,0.1);">
      <b style="font-size:0.85rem">🌡️ Mức giá (triệu/tháng)</b><br><br>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <div style="width:12px;height:12px;border-radius:50%;background:#2563eb"></div> Dưới 2 triệu
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <div style="width:12px;height:12px;border-radius:50%;background:#22d3ee"></div> 2–3 triệu
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <div style="width:12px;height:12px;border-radius:50%;background:#a3e635"></div> 3–4 triệu
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <div style="width:12px;height:12px;border-radius:50%;background:#facc15"></div> 4–6 triệu
      </div>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="width:12px;height:12px;border-radius:50%;background:#ef4444"></div> Trên 6 triệu
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def main():
    st.markdown("""
    <div class="app-header">
      <h1>🏠 TroSV – Dự Đoán Giá Thuê Trọ</h1>
      <p>Dành cho sinh viên UEH tại TP. Hồ Chí Minh · Powered by AI & ML</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    models = train_models(df)

    districts = sorted(df["quan"].unique().tolist())
    district_wards = {}
    for q in districts:
        district_wards[q] = sorted(df[df["quan"] == q]["phuong"].unique().tolist())

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Dự đoán giá",
        "🗺️ Bản đồ vòng quét",
        "🌡️ Heatmap giá",
        "💡 Đề xuất trọ tốt"
    ])

    with tab1:
        st.markdown('<div class="section-hd">Nhập thông tin phòng trọ</div>',
                    unsafe_allow_html=True)

        col_form, col_result = st.columns([1.1, 0.9], gap="large")

        with col_form:
            col_a, col_b = st.columns(2)
            with col_a:
                selected_quan = st.selectbox(
                    "🏙️ Quận / Huyện",
                    options=districts,
                    index=districts.index("Quận Bình Thạnh") if "Quận Bình Thạnh" in districts else 0,
                    help="Chọn quận / huyện"
                )

            with col_b:
                wards = district_wards.get(selected_quan, ["Tất cả"])
                selected_phuong = st.selectbox(
                    "📍 Phường / Xã",
                    options=wards,
                    help="Phường tự động theo quận"
                )

            area_m2 = st.slider(
                "📐 Diện tích phòng (m²)",
                min_value=8, max_value=80, value=25, step=1,
                help="Diện tích thực tế của phòng"
            )

            st.markdown("**🏷️ Tiện ích phòng trọ**")
            col1, col2, col3 = st.columns(3)
            with col1:
                may_lanh = st.checkbox("❄️ Máy lạnh", value=True)
                tu_lanh = st.checkbox("🧊 Tủ lạnh")
                gio_tu_do = st.checkbox("🌙 Giờ tự do", value=True)
            with col2:
                wc_rieng = st.checkbox("🚿 WC riêng", value=True)
                ban_cong = st.checkbox("🏗️ Ban công")
                co_bep = st.checkbox("🍳 Có bếp", value=True)
            with col3:
                cho_de_xe = st.checkbox("🅿️ Để xe", value=True)
                gan_sieu_thi = st.checkbox("🛒 Gần siêu thị")
                gan_quan_an = st.checkbox("🍜 Gần quán ăn")

            gan_truong_hoc = st.checkbox("🎓 Gần trường học")

            predict_btn = st.button("🔮 Dự đoán giá thuê", use_container_width=True)

        with col_result:
            amenities = {
                "may_lanh": int(may_lanh), "tu_lanh": int(tu_lanh),
                "gio_tu_do": int(gio_tu_do), "wc_rieng": int(wc_rieng),
                "ban_cong": int(ban_cong), "co_bep": int(co_bep),
                "cho_de_xe": int(cho_de_xe), "gan_sieu_thi": int(gan_sieu_thi),
                "gan_quan_an": int(gan_quan_an), "gan_truong_hoc": int(gan_truong_hoc)
            }

            if predict_btn or True:
                price, low, high = predict_price(
                    models, df, selected_quan, selected_phuong,
                    amenities, area_m2
                )

                amenity_score = sum(amenities.values())
                fuzzy_factor = fuzzy_price_adjustment(area_m2, amenity_score)

                st.markdown(f"""
                <div class="price-result">
                  <div class="label">💰 Giá thuê dự kiến / tháng</div>
                  <div class="amount">{format_price(price)}</div>
                  <div class="range">
                    Khoảng {format_price(low)} – {format_price(high)}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Điểm tiện ích", f"{amenity_score}/10",
                              delta="Tốt" if amenity_score >= 5 else "Cơ bản")
                

                st.markdown('<div class="section-hd" style="margin-top:16px">Phân tích chi tiết</div>',
                            unsafe_allow_html=True)

                area_label = ("Nhỏ (≤15m²)" if area_m2 <= 15 else
                              "Vừa (16–40m²)" if area_m2 <= 40 else "Rộng (>40m²)")
                price_level = ("Bình dân" if price < 3_000_000 else
                               "Trung bình" if price < 5_000_000 else "Cao cấp")

                amenity_list = [k.replace("_", " ").title()
                                for k, v in amenities.items() if v]
                badge_html = ""
                for am in amenity_list:
                    badge_html += f'<span class="badge badge-green">✓ {am}</span>'
                if not amenity_list:
                    badge_html = '<span class="badge">Không có tiện ích</span>'

                st.markdown(f"""
                <div class="card">
                  <div class="card-title">Khu vực</div>
                  <div style="font-weight:700;color:#222">{selected_phuong}</div>
                  <div style="color:#888;font-size:0.82rem">{selected_quan}</div>
                </div>
                <div class="card">
                  <div class="card-title">Diện tích & Mức giá</div>
                  <div style="display:flex;gap:10px;flex-wrap:wrap;">
                    <span class="badge badge-blue">📐 {area_m2} m² · {area_label}</span>
                    <span class="badge">💎 {price_level}</span>
                  </div>
                </div>
                <div class="card">
                  <div class="card-title">Tiện ích có</div>
                  <div>{badge_html}</div>
                </div>
                """, unsafe_allow_html=True)

                similar = df[
                    (df["quan"] == selected_quan) &
                    (abs(df["price"] - price) <= 1_500_000)
                ].copy()

                if len(similar) > 0:
                    st.markdown("**📊 Trọ tương tự trong khu vực**")
                    similar_show = similar[
                        ["phuong", "price", "amenity_score"]
                    ].rename(columns={
                        "phuong": "Phường",
                        "price": "Giá (VNĐ)",
                        "amenity_score": "Tiện ích"
                    }).head(5)
                    similar_show["Giá (VNĐ)"] = similar_show["Giá (VNĐ)"].apply(
                        lambda x: f"{x:,.0f}")
                    st.dataframe(similar_show, use_container_width=True, hide_index=True)


    with tab2:
        st.markdown('<div class="section-hd">Bản đồ vòng quét nhà trọ xung quanh UEH</div>',
                    unsafe_allow_html=True)

        col_ctrl1, col_ctrl2 = st.columns([1.5, 1])
        with col_ctrl1:
            campus_choice = st.selectbox(
                "🎓 Chọn cơ sở UEH",
                options=list(UEH_CAMPUSES.keys())
            )
        with col_ctrl2:
            radius_choice = st.selectbox(
                "📡 Bán kính tìm kiếm (km)",
                options=RADIUS_OPTIONS,
                index=2,
                format_func=lambda x: f"{x} km"
            )

        map_obj, nearby_df = make_map_with_radius(
            df, campus_choice, radius_choice, show_suggestions=True
        )

        st_folium(map_obj, width=None, height=500, returned_objects=[])

        if len(nearby_df) > 0:
            st.markdown(f"""
            <div class="section-hd">
              📋 {len(nearby_df)} nhà trọ trong bán kính {radius_choice} km
            </div>
            """, unsafe_allow_html=True)

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Giá trung bình",
                          format_price(nearby_df["price"].mean()))
            with col_s2:
                st.metric("Giá thấp nhất",
                          format_price(nearby_df["price"].min()))
            with col_s3:
                st.metric("Số lượng trọ",
                          f"{len(nearby_df)} phòng")

            st.markdown("**🏆 Top 10 trọ giá tốt gần nhất**")
            top10 = nearby_df.nsmallest(10, "price")[
                ["phuong", "quan", "price", "dist_km", "amenity_score",
                 "may_lanh", "wc_rieng", "ban_cong", "cho_de_xe"]
            ].rename(columns={
                "phuong": "Phường", "quan": "Quận",
                "price": "Giá (VNĐ)", "dist_km": "Cách (km)",
                "amenity_score": "Tiện ích",
                "may_lanh": "Máy lạnh", "wc_rieng": "WC riêng",
                "ban_cong": "Ban công", "cho_de_xe": "Để xe"
            })
            top10["Giá (VNĐ)"] = top10["Giá (VNĐ)"].apply(lambda x: f"{x:,.0f}")
            top10["Cách (km)"] = top10["Cách (km)"].apply(lambda x: f"{x:.2f}")
            for c in ["Máy lạnh", "WC riêng", "Ban công", "Để xe"]:
                top10[c] = top10[c].apply(lambda x: "✅" if x else "❌")
            st.dataframe(top10, use_container_width=True, hide_index=True)
        else:
            st.info(f"⚠️ Không tìm thấy trọ trong bán kính {radius_choice} km. Hãy thử tăng bán kính.")

    with tab3:
        st.markdown('<div class="section-hd">Bản đồ nhiệt giá thuê trọ TP.HCM</div>',
                    unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_quan = st.multiselect(
                "🏙️ Lọc theo quận",
                options=["Tất cả"] + districts,
                default=["Tất cả"]
            )
        with col_f2:
            price_range = st.slider(
                "💰 Lọc theo giá (triệu VNĐ/tháng)",
                min_value=0.5, max_value=15.0,
                value=(1.0, 10.0), step=0.5
            )

        df_filtered = df.copy()
        if "Tất cả" not in filter_quan and len(filter_quan) > 0:
            df_filtered = df_filtered[df_filtered["quan"].isin(filter_quan)]
        df_filtered = df_filtered[
            (df_filtered["price"] >= price_range[0] * 1_000_000) &
            (df_filtered["price"] <= price_range[1] * 1_000_000)
        ]

        heatmap_obj = make_heatmap(df_filtered)
        st_folium(heatmap_obj, width=None, height=520, returned_objects=[])

        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;">
          <div class="card" style="flex:1;min-width:140px;text-align:center">
            <div class="card-title">Số điểm dữ liệu</div>
            <div class="card-value">{len(df_filtered)}</div>
            <div class="card-sub">trên {len(df)} bản ghi</div>
          </div>
          <div class="card" style="flex:1;min-width:140px;text-align:center">
            <div class="card-title">Giá trung bình</div>
            <div class="card-value">{format_price(df_filtered['price'].mean())}</div>
            <div class="card-sub">/ tháng</div>
          </div>
          <div class="card" style="flex:1;min-width:140px;text-align:center">
            <div class="card-title">Số quận</div>
            <div class="card-value">{df_filtered['quan'].nunique()}</div>
            <div class="card-sub">quận/huyện</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-hd">Đề xuất trọ giá tốt theo cơ sở UEH</div>',
                    unsafe_allow_html=True)

        campus_tab_choice = st.selectbox(
            "🎓 Chọn cơ sở cần tìm trọ",
            options=list(UEH_CAMPUSES.keys()),
            key="tab4_campus"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            max_price_suggest = st.slider(
                "💰 Ngân sách tối đa (triệu/tháng)",
                min_value=1.0, max_value=10.0, value=4.0, step=0.5
            )
        with col_opt2:
            max_radius_suggest = st.selectbox(
                "📡 Bán kính tối đa (km)",
                options=[1.0, 1.5, 2.0, 3.0, 5.0],
                index=2,
                key="tab4_radius",
                format_func=lambda x: f"{x} km"
            )

        st.markdown("**🏷️ Yêu cầu tiện ích**")
        col_req1, col_req2, col_req3 = st.columns(3)
        with col_req1:
            req_may_lanh = st.checkbox("❄️ Máy lạnh", key="r1")
            req_wc_rieng = st.checkbox("🚿 WC riêng", key="r2", value=True)
        with col_req2:
            req_ban_cong = st.checkbox("🏗️ Ban công", key="r3")
            req_cho_de_xe = st.checkbox("🅿️ Để xe", key="r4", value=True)
        with col_req3:
            req_co_bep = st.checkbox("🍳 Có bếp", key="r5")
            req_gio_tu_do = st.checkbox("🌙 Giờ tự do", key="r6")

        campus_info = UEH_CAMPUSES[campus_tab_choice]
        clat, clon = campus_info["lat"], campus_info["lon"]

        df_suggest = df.copy()
        df_suggest["dist_km"] = df_suggest.apply(
            lambda r: haversine(clat, clon, r["latitude"], r["longitude"]),
            axis=1
        )
        df_suggest = df_suggest[
            (df_suggest["dist_km"] <= max_radius_suggest) &
            (df_suggest["price"] <= max_price_suggest * 1_000_000)
        ]

        if req_may_lanh:
            df_suggest = df_suggest[df_suggest["may_lanh"] == 1]
        if req_wc_rieng:
            df_suggest = df_suggest[df_suggest["wc_rieng"] == 1]
        if req_ban_cong:
            df_suggest = df_suggest[df_suggest["ban_cong"] == 1]
        if req_cho_de_xe:
            df_suggest = df_suggest[df_suggest["cho_de_xe"] == 1]
        if req_co_bep:
            df_suggest = df_suggest[df_suggest["co_bep"] == 1]
        if req_gio_tu_do:
            df_suggest = df_suggest[df_suggest["gio_tu_do"] == 1]

        df_suggest["score"] = (
            (1 - df_suggest["dist_km"] / max_radius_suggest) * 40 +
            (1 - df_suggest["price"] / (max_price_suggest * 1_000_000)) * 35 +
            (df_suggest["amenity_score"] / 10) * 25
        )
        df_suggest = df_suggest.sort_values("score", ascending=False)

        if len(df_suggest) > 0:
            st.success(
                f"✅ Tìm thấy **{len(df_suggest)}** phòng phù hợp trong bán kính "
                f"{max_radius_suggest} km với ngân sách ≤ {max_price_suggest:.0f} triệu/tháng"
            )

            for i, (_, row) in enumerate(df_suggest.head(8).iterrows()):
                score_pct = min(100, int(row["score"]))
                price_diff_pct = (row["price"] / (max_price_suggest * 1_000_000) - 1) * 100

                amenities_row = []
                if row.get("may_lanh"): amenities_row.append("❄️")
                if row.get("tu_lanh"): amenities_row.append("🧊")
                if row.get("wc_rieng"): amenities_row.append("🚿")
                if row.get("ban_cong"): amenities_row.append("🏗️")
                if row.get("co_bep"): amenities_row.append("🍳")
                if row.get("cho_de_xe"): amenities_row.append("🅿️")
                if row.get("gio_tu_do"): amenities_row.append("🌙")
                if row.get("gan_sieu_thi"): amenities_row.append("🛒")

                star = "⭐" if i == 0 else f"#{i+1}"
                bar_width = score_pct

                st.markdown(f"""
                <div class="card" style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;align-items:start">
                    <div>
                      <span style="font-size:0.78rem;font-weight:700;color:#e84141">{star} GỢI Ý PHÙ HỢP</span>
                      <div style="font-size:1.05rem;font-weight:800;color:#222;margin-top:2px">
                        {format_price(row['price'])}<span style="font-size:0.78rem;font-weight:400;color:#666">/tháng</span>
                      </div>
                      <div style="color:#666;font-size:0.82rem;margin-top:2px">
                        📍 {row['phuong']}, {row['quan']}
                      </div>
                    </div>
                    <div style="text-align:right">
                      <div style="font-size:0.72rem;color:#888">Phù hợp</div>
                      <div style="font-size:1.2rem;font-weight:800;color:#27ae60">{score_pct}%</div>
                    </div>
                  </div>
                  <div style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0">
                    <span class="badge badge-blue">📏 {row['dist_km']:.1f} km</span>
                    <span class="badge">{'  '.join(amenities_row) if amenities_row else 'Cơ bản'}</span>
                    <span class="badge badge-green">⭐ {row['amenity_score']}/10 tiện ích</span>
                  </div>
                  <div style="background:#f5f5f5;border-radius:6px;height:6px;margin-top:6px">
                    <div style="background:linear-gradient(90deg,#e84141,#f97316);
                                width:{bar_width}%;height:6px;border-radius:6px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**🗺️ Bản đồ các gợi ý**")
            m_suggest = folium.Map(
                location=[clat, clon],
                zoom_start=14,
                tiles="CartoDB positron"
            )
            folium.Marker(
                [clat, clon],
                popup=campus_tab_choice,
                icon=folium.Icon(color="red", icon="graduation-cap", prefix="fa")
            ).add_to(m_suggest)

            for _, row in df_suggest.head(20).iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=8,
                    color="#27ae60",
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"{format_price(row['price'])}/tháng – {row['phuong']}",
                    tooltip=f"{format_price(row['price'])}"
                ).add_to(m_suggest)

            folium.Circle(
                [clat, clon],
                radius=max_radius_suggest * 1000,
                color="#e84141",
                fill=True,
                fill_opacity=0.05,
                weight=2,
                dash_array="6 4"
            ).add_to(m_suggest)

            st_folium(m_suggest, width=None, height=380, returned_objects=[])
        else:
            st.warning(
                "⚠️ Không tìm thấy phòng phù hợp. Hãy thử:\n"
                "• Tăng ngân sách\n• Mở rộng bán kính tìm kiếm\n"
                "• Bỏ bớt yêu cầu tiện ích"
            )

    st.markdown("""
    <div class="footer">
      TroSV © 2025 · Dự đoán giá thuê trọ sinh viên UEH · TP. Hồ Chí Minh<br>
      Dữ liệu thực tế · Mô hình hồi quy máy (GBR + Ridge) + Fuzzy Logic · Folium Maps
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
