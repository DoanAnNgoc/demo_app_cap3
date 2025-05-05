import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.interpolate import make_interp_spline
import requests
import io
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tải dữ liệu từ Google Drive
@st.cache_data
def load_data():
    FILE_URL = "https://drive.google.com/uc?export=download&id=1BEgh4x_dS0W-31ITcrt5iTT8Rv_aqviZ"
    logger.info("Bắt đầu tải dữ liệu từ Google Drive...")
    
    try:
        response = requests.get(FILE_URL, stream=True, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        content_length = response.headers.get('content-length', 'Unknown')
        logger.info(f"Content-Type: {content_type}, Content-Length: {content_length} bytes")
        
        if 'text/csv' not in content_type and 'application/octet-stream' not in content_type:
            st.error(f"URL không trả về file CSV. Content-Type: {content_type}")
            logger.error(f"Invalid Content-Type: {content_type}")
            return None

        content = response.content
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        for encoding in encodings:
            try:
                logger.info(f"Thử đọc CSV với encoding: {encoding}")
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    on_bad_lines='skip',
                    quoting=3,
                    low_memory=False
                )
                logger.info("Đọc file CSV thành công")
                break
            except (pd.errors.ParserError, UnicodeDecodeError) as e:
                logger.warning(f"Lỗi với encoding {encoding}: {str(e)}")
                if encoding == encodings[-1]:
                    st.error("Không thể đọc file CSV với bất kỳ encoding nào.")
                    logger.error(f"Thất bại với tất cả encoding: {str(e)}")
                    return None
                continue

        logger.info(f"Dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")
        st.write(f"**Thông tin dữ liệu gốc**: {df.shape[0]} dòng, {df.shape[1]} cột")
        st.write(f"**Cột**: {list(df.columns)}")

        # Xử lý dữ liệu
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['year'] = df['Order Date'].dt.year
        numeric_cols = ['Order Total', 'Product Cost', 'Shipping Fee', 'Profit', 'Quantity', 'Marketplace Fee']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    st.warning(f"Cột '{col}' có {nan_count} giá trị NaN sau khi chuyển sang số.")

        # Tính Profit theo công thức mới
        required_cols = ['Order Total', 'Product Cost', 'Quantity', 'Shipping Fee']
        if all(col in df.columns for col in required_cols):
            if 'Marketplace Fee' in df.columns:
                df['Profit'] = df['Order Total'] - (df['Product Cost'] * df['Quantity']) - df['Shipping Fee'] - df['Marketplace Fee']
                logger.info("Tính Profit với Marketplace Fee")
            else:
                df['Profit'] = df['Order Total'] - (df['Product Cost'] * df['Quantity']) - df['Shipping Fee']
                logger.info("Tính Profit không có Marketplace Fee")
                st.warning("Cột 'Marketplace Fee' không có, Profit được tính mà không trừ phí sàn.")
        else:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.warning(f"Thiếu các cột cần thiết để tính Profit: {missing_cols}")
            logger.warning(f"Thiếu cột: {missing_cols}")

        # Làm sạch dữ liệu
        df_clean = df.dropna(subset=['Order Date', 'Order Total'])
        logger.info(f"Dữ liệu sau khi làm sạch: {df_clean.shape[0]} dòng")
        st.write(f"**Dữ liệu sau khi làm sạch**: {df_clean.shape[0]} dòng (mất {df.shape[0] - df_clean.shape[0]} dòng do thiếu Order Date hoặc Order Total)")

        return df_clean
    
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi tải file từ Google Drive: {str(e)}")
        logger.error(f"Lỗi tải file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Lỗi không xác định: {str(e)}")
        logger.error(f"Lỗi không xác định: {str(e)}")
        return None

# Tải dữ liệu
with st.spinner("Đang tải dữ liệu từ Google Drive..."):
    df = load_data()
    if df is None:
        st.error("Không thể tải dữ liệu. Vui lòng kiểm tra file CSV hoặc URL.")
        st.stop()

# Tiêu đề và mô tả
st.title("Đề Án Tốt Nghiệp - Phân Tích Tình hình kinh doanh - Dự đoán doanh thu và Phân Cụm Khách Hàng")
st.markdown("""
Ứng dụng này hiển thị phân cụm khách hàng, dự đoán doanh thu, 
và các biểu đồ phân tích dựa trên dữ liệu từ file CSV trên Google Drive.
Bạn vui lòng chọn tab để xem các phân tích chi tiết.
""")

# Tạo các tab
tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan Tình Hình Kinh Doanh", "💵 Dự Đoán Doanh Thu", "📀 Phân Cụm Khách Hàng"])

# Tab 1: Tổng Quan Doanh Thu
with tab1:
    st.header("📊 Tổng Quan Doanh Thu Theo Năm")
    revenue_by_year = df.groupby('year')['Order Total'].sum().reset_index()
    fig = px.bar(revenue_by_year, x='year', y='Order Total', title='Tổng Doanh Thu Theo Năm',
                 labels={'year': 'Năm', 'Order Total': 'Tổng Doanh Thu'}, color_discrete_sequence=['red'])
    fig.update_layout(xaxis_tickangle=0, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig)

    st.subheader("Tổng Order Total theo Sub-Category Theo Năm")
    df['Year'] = df['Order Date'].dt.year
    pivot_data = df.groupby(['Year', 'Sub Category'])['Order Total'].sum().unstack()
    years = sorted(pivot_data.index)
    year_colors = {
        years[-3] if len(years) >= 3 else years[0]: '#1f77b4',
        years[-2] if len(years) >= 2 else years[0]: '#2ca02c',
        years[-1] if len(years) >= 1 else years[0]: '#ff7f0e',
    }
    selected_year = st.selectbox("Chọn năm:", years, index=len(years)-1)
    if years:
        data = pivot_data.loc[selected_year].sort_values()
        top5 = data.nlargest(5).index
        colors = [year_colors.get(selected_year, 'lightgray') if subcat in top5 else 'lightgray' for subcat in data.index]
        plt.figure(figsize=(10, 6))
        bars = plt.barh(data.index, data.values, color=colors)
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:,.2f}', va='center', fontsize=9)
        plt.title(f'Tổng Doanh thu theo Sub-Category - Năm {selected_year}')
        plt.xlabel('Tổng Order Total')
        plt.ylabel('Sub-Category')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.warning("Không có dữ liệu để hiển thị biểu đồ theo Sub-Category.")

    if 'Marketplace' in df.columns:
        st.subheader("💳 Tổng Quan Theo Marketplace")
        summary = df.groupby('Marketplace').agg({
            'Order Total': 'sum', 'Product Cost': 'sum', 'Shipping Fee': 'sum', 'Profit': 'sum'
        }).reset_index()
        summary.columns = ['Marketplace', 'Revenue', 'Cost', 'ShippingFee', 'Profit']
        fig2 = go.Figure()
        for i, row in summary.iterrows():
            fig2.add_trace(go.Indicator(mode="number+delta", value=row['Revenue'], delta={'reference': 0, 'valueformat':'.2f'},
                                        title={"text": f"<b>{row['Marketplace']}</b><br>Revenue"}, domain={'row': i, 'column': 0}))
            fig2.add_trace(go.Indicator(mode="number+delta", value=row['Cost'], delta={'reference': 0, 'valueformat':'.2f'},
                                        title={"text": f"<b>{row['Marketplace']}</b><br>Cost"}, domain={'row': i, 'column': 1}))
            fig2.add_trace(go.Indicator(mode="number+delta", value=row['Profit'], delta={'reference': 0, 'valueformat':'.2f'},
                                        title={"text": f"<b>{row['Marketplace']}</b><br>Profit"}, domain={'row': i, 'column': 2}))
        fig2.update_layout(grid={'rows': len(summary), 'columns': 3, 'pattern': "independent"}, height=250 * len(summary), title="💳 Tổng Quan Theo Marketplace")
        st.plotly_chart(fig2)
    else:
        st.warning("Cột 'Marketplace' không có trong dữ liệu.")

    if 'Marketplace' in df.columns and 'Order Id' in df.columns:
        st.subheader("Số Lượng Đơn Hàng Theo Sàn")
        grouped = df.groupby('Marketplace').agg({'Order Total': 'sum', 'Order Id': 'count'}).reset_index().rename(columns={'Order Id': 'OrderCount'})
        colors = ['red', 'green'] * (len(grouped) // 2 + 1)
        grouped['Color'] = colors[:len(grouped)]
        fig3 = px.bar(grouped, x='Marketplace', y='OrderCount', title='Số lượng đơn hàng theo sàn', text_auto=True,
                      color='Color', color_discrete_map={'red': 'red', 'green': 'green'})
        fig3.update_layout(xaxis_title='Sàn', yaxis_title='Số lượng đơn', showlegend=False)
        st.plotly_chart(fig3)
    else:
        st.warning("Cột 'Marketplace' hoặc 'Order Id' không có trong dữ liệu.")

    if 'Sub Category' in df.columns and 'Quantity' in df.columns:
        st.subheader("Top 5 Sản Phẩm Bán Chạy Nhất")
        top_products = df.groupby('Sub Category')['Quantity'].sum().sort_values(ascending=False).head(5).reset_index()
        fig4 = px.bar(top_products, x='Sub Category', y='Quantity', title='Top 5 sản phẩm bán chạy nhất', text_auto=True,
                      color_discrete_sequence=['red'])
        fig4.update_layout(xaxis_title='Tên sản phẩm', yaxis_title='Số lượng bán')
        st.plotly_chart(fig4)
    else:
        st.warning("Cột 'Sub Category' hoặc 'Quantity' không có trong dữ liệu.")

    if 'City' in df.columns and 'Country' in df.columns:
        st.subheader("Doanh Thu Theo Thành Phố")
        city_group = df.groupby(['City', 'Country']).agg({'Order Total': 'sum'}).reset_index()
        fig5 = px.scatter_geo(city_group, locations="Country", locationmode="country names", color="Order Total",
                              size="Order Total", hover_name="City", scope='world', title='Doanh thu theo thành phố', size_max=20)
        st.plotly_chart(fig5)
        st.warning("Biểu đồ bản đồ hiện chỉ hiển thị theo quốc gia do thiếu tọa độ lat/lon. Cần thêm tọa độ hoặc dùng geopy.")
    else:
        st.warning("Cột 'City' hoặc 'Country' không có trong dữ liệu.")

# Tab 2: Dự Đoán Doanh Thu
with tab2:
    st.header("💵 Dự Đoán Doanh Thu với Prophet")
    prophet_df = df[['Order Date', 'Order Total']].rename(columns={'Order Date': 'ds', 'Order Total': 'y'})
    prophet_df = prophet_df.groupby('ds').sum().reset_index()
    prophet_df['y_smooth'] = prophet_df['y'].rolling(window=5, center=True, min_periods=1).mean()
    prophet_df['ds_numeric'] = prophet_df['ds'].apply(lambda x: x.timestamp())
    prophet_df = prophet_df.sort_values('ds_numeric')
    x = prophet_df['ds_numeric']
    y = prophet_df['y_smooth']
    x_smooth = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    ds_smooth = pd.to_datetime(x_smooth, unit='s')
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.01)
    model.fit(prophet_df)
    past_future = prophet_df[['ds']].copy()
    past_forecast = model.predict(past_future)
    past_forecast['yhat_smooth'] = past_forecast['yhat'].rolling(window=5, center=True, min_periods=1).mean()
    past_forecast['yhat_lower_smooth'] = past_forecast['yhat_lower'].rolling(window=5, center=True, min_periods=1).mean()
    past_forecast['yhat_upper_smooth'] = past_forecast['yhat_upper'].rolling(window=5, center=True, min_periods=1).mean()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ds_smooth, y=y_smooth, mode='lines', name='Thực tế', line=dict(color='blue', width=1)))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_smooth'], mode='lines', name='Dự đoán', line=dict(color='orange', width=1)))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_upper_smooth'], mode='lines', name='Khoảng tin cậy (trên)', line=dict(color='yellow', width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_lower_smooth'], mode='lines', name='Khoảng tin cậy (dưới)', line=dict(color='yellow', width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)'))
    fig1.update_layout(title='Giá trị bán hàng hàng tháng - Prophet (Tập gốc)', xaxis_title='Ngày', yaxis_title='Giá trị bán hàng', xaxis_tickformat='%Y-%m', xaxis_tickangle=45, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig1)
    future = model.make_future_dataframe(periods=365, freq='D')
    future_forecast = model.predict(future)
    future_forecast['yhat_smooth'] = future_forecast['yhat'].rolling(window=5, center=True, min_periods=1).mean()
    future_forecast['yhat_lower_smooth'] = future_forecast['yhat_lower'].rolling(window=5, center=True, min_periods=1).mean()
    future_forecast['yhat_upper_smooth'] = future_forecast['yhat_upper'].rolling(window=5, center=True, min_periods=1).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ds_smooth, y=y_smooth, mode='lines', name='Thực tế', line=dict(color='blue', width=1)))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()], y=future_forecast['yhat_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()], mode='lines', name='Dự đoán', line=dict(color='orange', width=1)))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()], y=future_forecast['yhat_smooth'][future_forecast['ds'] > prophet_df['ds'].max()], mode='lines', name='Dự đoán tương lai', line=dict(color='red', width=1)))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()], y=future_forecast['yhat_upper_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()], mode='lines', name='Khoảng tin cậy', line=dict(color='yellow', width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()], y=future_forecast['yhat_lower_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()], mode='lines', name='Khoảng tin cậy', line=dict(color='yellow', width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)'))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()], y=future_forecast['yhat_upper_smooth'][future_forecast['ds'] > prophet_df['ds'].max()], mode='lines', name='Khoảng tin cậy tương lai', line=dict(color='pink', width=0), showlegend=False))
    fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()], y=future_forecast['yhat_lower_smooth'][future_forecast['ds'] > prophet_df['ds'].max()], mode='lines', name='Khoảng tin cậy tương lai', line=dict(color='pink', width=0), fill='tonexty', fillcolor='rgba(255, 192, 203, 0.2)'))
    fig2.update_layout(title='Dự đoán giá trị bán hàng 12 tháng tiếp theo - Prophet', xaxis_title='Ngày', yaxis_title='Giá trị bán hàng', xaxis_tickformat='%Y-%m', xaxis_tickangle=45, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig2)
    eval_df = pd.merge(prophet_df[['ds', 'y']], past_forecast[['ds', 'yhat']], on='ds')
    mae = mean_absolute_error(eval_df['y'], eval_df['yhat'])
    rmse = np.sqrt(mean_squared_error(eval_df['y'], eval_df['yhat']))
    mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
    r2 = r2_score(eval_df['y'], eval_df['yhat'])
    st.subheader("Đánh Giá Mô Hình Dự Đoán")
    st.write(f"📊 MAE: {mae:.2f}")
    st.write(f"📊 RMSE: {rmse:.2f}")
    st.write(f"📊 MAPE: {mape:.2f}%")
    st.write(f"📊 R² Score: {r2:.2f}")
    st.subheader("Dự Đoán Doanh Thu Cho Ngày Cụ Thể")
    today = datetime.today().date()
    max_date = today + timedelta(days=365)
    selected_date = st.date_input("Chọn ngày trong tương lai để dự đoán doanh thu:", min_value=today, max_value=max_date, value=today + timedelta(days=30))
    selected_date_df = pd.DataFrame({'ds': [pd.to_datetime(selected_date)]})
    selected_forecast = model.predict(selected_date_df)
    st.markdown(f"**Dự đoán doanh thu cho ngày {selected_date}:**")
    st.write(f"📈 Giá trị dự đoán: **${selected_forecast['yhat'].iloc[0]:,.2f}**")
    st.write(f"📉 Khoảng tin cậy thấp: **${selected_forecast['yhat_lower'].iloc[0]:,.2f}**")
    st.write(f"📊 Khoảng tin cậy cao: **${selected_forecast['yhat_upper'].iloc[0]:,.2f}**")

# Tab 3: Phân Cụm Khách Hàng
with tab3:
    st.header("📀 Phân Cụm Khách Hàng với GMM")
    df_sample = df.sample(n=35000, random_state=42)
    df_cluster = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit']]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    gmm = GaussianMixture(n_components=7, random_state=42)
    clusters = gmm.fit_predict(df_pca)
    df_sample['Cluster'] = clusters
    fig3 = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1], color=clusters.astype(str), title='Phân Cụm Khách Hàng với GMM',
                      labels={'x': 'PCA 1', 'y': 'PCA 2', 'color': 'Cluster'}, color_discrete_sequence=px.colors.qualitative.T10)
    fig3.update_layout(showlegend=True)
    st.plotly_chart(fig3)
    st.subheader("Kết Quả Phân Cụm (Mẫu)")
    st.dataframe(df_sample[['Order Id', 'City', 'Country', 'Cluster']].head())
    df_analysis = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit', 'Cluster']].copy()
    cluster_summary = df_analysis.groupby('Cluster').mean().round(2)
    cluster_counts = df_analysis['Cluster'].value_counts().sort_index()
    cluster_summary['Count'] = cluster_counts
    st.subheader("Đặc Trưng Trung Bình của Từng Cụm")
    st.dataframe(cluster_summary)
    st.subheader("Đánh Giá Mô Hình Phân Cụm")
    df_valid = df_sample.dropna(subset=['Cluster'])
    X_valid = df_pca
    labels = df_valid['Cluster']
    sil_score = silhouette_score(X_valid, labels)
    db_index = davies_bouldin_score(X_valid, labels)
    ch_index = calinski_harabasz_score(X_valid, labels)
    st.write(f"📊 Silhouette Score: {sil_score:.3f}")
    st.write(f"📊 Davies-Bouldin Index: {db_index:.3f}")
    st.write(f"📊 Calinski-Harabasz Index: {ch_index:.3f}")
    st.write(f"📊 Số cụm: {len(set(labels))}")

# Footer
st.markdown("---")
st.markdown("Web App Demo Đề Án Tốt Nghiệp được xây dựng với Streamlit bởi Ấn Ngọc. Liên hệ hỗ trợ: anngocmukbang@gmail.com")
