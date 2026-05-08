import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Triple Tea - Dự đoán giá trọ", layout="wide")

st.title("🏠 Ứng dụng Dự đoán Giá thuê Trọ - Nhóm Triple Tea")
st.markdown("---")

# --- 2. ĐỌC VÀ LÀM SẠCH DỮ LIỆU ---
file_path = 'Dự đoán giá thuê trọ (Câu trả lời).xlsx'

@st.cache_data
def load_data():
    try:
        df = pd.read_excel(file_path)
    except:
        df = pd.DataFrame({
            'Khoảng cách': [1.2, 2.5, 4.0, 1.0, 5.5, 3.2, 2.1, 1.8, 4.7, 3.9] * 20,
            'Diện tích': [18, 25, 35, 15, 45, 22, 30, 20, 40, 28] * 20,
            'Phường': ['P1', 'P2', 'P3', 'P1', 'P4', 'P2', 'P3', 'P1', 'P4', 'P2'] * 20,
            'Xung quanh trọ có các tiện ích nào?': ['Máy lạnh, Chợ', 'Siêu thị, Trạm xe buýt', 'Bếp riêng', 'WC riêng'] * 50,
            'Tổng giá thuê...': ['3.5 triệu', '4.5 triệu', '6.0 triệu', '3.2', '7.0', '4.2', '5.5', '3.8', '6.5', '4.8'] * 20
        })
    return df.drop_duplicates().copy()

df_raw = load_data()

# Hàm tìm cột thông minh
def find_col(df, keywords, default_idx):
    for i, col in enumerate(df.columns):
        if any(key.lower() in col.lower() for key in keywords): 
            return col
    return df.columns[default_idx]

km_col_raw = find_col(df_raw, ['khoảng cách', 'km'], 0)
size_col_raw = find_col(df_raw, ['diện tích', 'm2'], 1)
ward_col_raw = find_col(df_raw, ['phường', 'quận'], 2)
util_col_raw = find_col(df_raw, ['tiện ích'], 3)
price_col_raw = find_col(df_raw, ['giá thuê', 'tổng giá'], -1)

# Làm sạch dữ liệu
df_clean = df_raw.copy()
def clean_price(text):
    if pd.isnull(text): return np.nan
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(text).replace(',', '.'))
    if not nums: return np.nan
    val = float(nums[0])
    return val / 1000000 if val > 1000 else val

df_clean[price_col_raw] = df_clean[price_col_raw].apply(clean_price)
df_clean[size_col_raw] = pd.to_numeric(df_clean[size_col_raw], errors='coerce')
df_clean = df_clean.dropna(subset=[price_col_raw, size_col_raw])

# Feature Engineering
utils_list = ['Máy lạnh', 'WC riêng', 'Bếp riêng', 'Giờ giấc tự do', 'Chỗ để xe', 'Chợ', 'Trạm xe buýt', 'Siêu thị', 'Cửa hàng tiện lợi']
for u in utils_list:
    df_clean[u] = df_clean[util_col_raw].apply(lambda x: 1 if pd.notnull(x) and u.lower() in str(x).lower() else 0)
df_clean['Tổng tiện ích'] = df_clean[utils_list].sum(axis=1)

# --- 3. HUẤN LUYỆN MÔ HÌNH ---
df_encoded = pd.get_dummies(df_clean, columns=[ward_col_raw], drop_first=True)
X_all = df_encoded.drop(columns=[price_col_raw, util_col_raw, 'Dấu thời gian'], errors='ignore')
X_all = X_all.select_dtypes(include=[np.number, 'bool'])
y_all = df_clean[price_col_raw]
MODEL_FEATURES = X_all.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

scaler = StandardScaler()
cont_cols = [km_col_raw, size_col_raw, 'Tổng tiện ích']
X_train_scaled = X_train.copy()
X_train_scaled[cont_cols] = scaler.fit_transform(X_train[cont_cols])

rf_model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# --- 4. GIAO DIỆN STREAMLIT (SIDEBAR NHẬP LIỆU) ---
with st.sidebar:
    st.header("📍 Thông tin phòng")
    in_ward = st.selectbox("Chọn khu vực:", sorted(df_clean[ward_col_raw].unique()))
    in_dist = st.slider("Khoảng cách (km):", 0.1, 15.0, 1.0)
    in_size = st.number_input("Diện tích (m2):", 5, 100, 20)
    
    st.header("✨ Tiện ích")
    user_utils = {}
    for u in utils_list:
        user_utils[u] = st.checkbox(u)

# --- 5. HIỂN THỊ KẾT QUẢ VÀ TRỰC QUAN HÓA ---
tab1, tab2, tab3 = st.tabs(["🚀 Dự đoán giá", "📊 Phân tích dữ liệu", "🎯 Hiệu năng mô hình"])

with tab1:
    col_res1, col_res2 = st.columns([1, 1])
    
    # Chuẩn bị dữ liệu dự đoán
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
    input_df[km_col_raw] = in_dist
    input_df[size_col_raw] = in_size
    count_util = 0
    for u, val in user_utils.items():
        if u in input_df.columns:
            input_df[u] = 1 if val else 0
            count_util += 1 if val else 0
    input_df['Tổng tiện ích'] = count_util
    
    w_col = f"{ward_col_raw}_{in_ward}"
    if w_col in input_df.columns:
        input_df[w_col] = 1
        
    input_scaled = input_df.copy()
    input_scaled[cont_cols] = scaler.transform(input_df[cont_cols])
    
    prediction = rf_model.predict(input_scaled)[0]

    with col_res1:
        st.success(f"### Dự báo giá thuê tại {in_ward}")
        st.metric("Giá trọ ước tính", f"{prediction * 1000000:,.0f} VNĐ/tháng")
        st.info("Giá trên đã bao gồm các tiện ích bạn đã chọn.")

    with col_res2:
        importances = pd.Series(rf_model.feature_importances_, index=MODEL_FEATURES).sort_values().tail(10)
        fig_feat, ax_feat = plt.subplots()
        importances.plot(kind='barh', ax=ax_feat, color='forestgreen')
        ax_feat.set_title("Các yếu tố ảnh hưởng nhất")
        st.pyplot(fig_feat)

with tab2:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.write("### Ma trận tương quan")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df_clean[[price_col_raw, size_col_raw, km_col_raw, 'Tổng tiện ích']].corr(), annot=True, cmap='RdYlGn', ax=ax_corr)
        st.pyplot(fig_corr)
        
    with col_v2:
        st.write("### Phân bổ giá theo diện tích")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(data=df_clean, x=size_col_raw, y=price_col_raw, hue=km_col_raw, ax=ax_scatter)
        st.pyplot(fig_scatter)

with tab3:
    X_test_scaled = X_test.copy()
    X_test_scaled[cont_cols] = scaler.transform(X_test[cont_cols])
    y_pred = rf_model.predict(X_test_scaled)
    
    st.write(f"**R2 Score:** {r2_score(y_test, y_pred):.3f}")
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")
    
    fig_perf, ax_perf = plt.subplots()
    ax_perf.scatter(y_test, y_pred, alpha=0.5)
    ax_perf.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'r--')
    ax_perf.set_xlabel("Thực tế (Triệu)")
    ax_perf.set_ylabel("Dự đoán (Triệu)")
    st.pyplot(fig_perf)

if st.checkbox("Hiển thị dữ liệu mẫu"):
    st.write(df_clean.head(20))
