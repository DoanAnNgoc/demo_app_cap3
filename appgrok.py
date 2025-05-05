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

# Thiết lập tiêu đề
st.title("Đề Án Tốt Nghiệp - Phân Tích Doanh Thu và Phân Cụm Khách Hàng")
st.markdown("""
Ứng dụng này hiển thị phân cụm khách hàng, dự đoán doanh thu,
và các biểu đồ phân tích dựa trên dữ liệu từ tệp Excel trên Google Drive.
Bạn vui lòng chọn tab để xem các phân tích chi tiết.
""")

# Tải dữ liệu từ Google Drive
@st.cache_data
def load_data():
    # Liên kết Google Drive (thay bằng liên kết Excel công khai)
    shareable_link = "https://docs.google.com/spreadsheets/d/1u2aXzp7gXuKF7qOEx-maeBNDMw7-pbQA"  
        # Lấy ID tệp từ liên kết
        file_id = shareable_link.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        df = pd.read_excel(download_url, engine='openpyxl')
        
        # Đảm bảo cột 'Order Date' là datetime
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['year'] = df['Order Date'].dt.year
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp Excel từ Google Drive: {e}")
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu lỗi

# Tải dữ liệu
with st.spinner("Đang tải dữ liệu từ Google Drive..."):
    df = load_data()

# Kiểm tra số cột trong df
if not df.empty:
    expected_cols = 22
    print(f"Số cột trong tiêu đề DataFrame: {len(df.columns)}")
    print(f"Các cột: {list(df.columns)}")

    # Đếm số dòng hợp lệ và không hợp lệ
    clean_lines = []
    error_lines = []
    col_count_distribution = {}

    for i, row in df.iterrows():
        num_cols = len(row.dropna())
        col_count_distribution[num_cols] = col_count_distribution.get(num_cols, 0) + 1
        if num_cols == expected_cols:
            clean_lines.append(row.tolist())
        else:
            if len(error_lines) < 5:
                error_lines.append((i + 2, num_cols, row.tolist()))

    # Hiển thị kết quả trong Streamlit
    st.write(f"**Số dòng hợp lệ (có đúng {expected_cols} cột):** {len(clean_lines)}")
    st.write("**Phân bố số cột trong DataFrame:**")
    for num_cols, count in sorted(col_count_distribution.items()):
        st.write(f"Số dòng có {num_cols} cột: {count}")

    if error_lines:
        st.write("**Các dòng không đạt số cột chuẩn (tối đa 5 dòng):**")
        for i, col_count, content in error_lines:
            st.write(f"Dòng {i} có {col_count} cột: {content}")
    else:
        st.write("Tất cả các dòng đều đúng số cột.")

# Tiếp tục các tab phân tích (giữ nguyên mã của bạn)
tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan Doanh Thu", "💵 Dự Đoán Doanh Thu", "📀 Phân Cụm Khách Hàng"])

# Tab 1: Tổng Quan Doanh Thu
with tab1:
    st.header("📊 Tổng Quan Doanh Thu Theo Năm")
    if not df.empty:
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
                plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                         f'{width:,.2f}', va='center', fontsize=9)
            plt.title(f'Tổng Doanh thu theo Sub-Category - Năm {selected_year}')
            plt.xlabel('Tổng Order Total')
            plt.ylabel('Sub-Category')
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.warning("Không có dữ liệu để hiển thị biểu đồ theo Sub-Category.")
    else:
        st.error("Không thể hiển thị biểu đồ do lỗi tải dữ liệu.")

# Tab 2: Dự Đoán Doanh Thu
with tab2:
    st.header("💵 Dự Đoán Doanh Thu với Prophet")
    if not df.empty:
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
        fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()],
                                  y=future_forecast['yhat_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()],
                                  mode='lines', name='Dự đoán', line=dict(color='orange', width=1)))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()],
                                  y=future_forecast['yhat_smooth'][future_forecast['ds'] > prophet_df['ds'].max()],
                                  mode='lines', name='Dự đoán tương lai', line=dict(color='red', width=1)))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()],
                                  y=future_forecast['yhat_upper_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()],
                                  mode='lines', name='Khoảng tin cậy', line=dict(color='yellow', width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] <= prophet_df['ds'].max()],
                                  y=future_forecast['yhat_lower_smooth'][future_forecast['ds'] <= prophet_df['ds'].max()],
                                  mode='lines', name='Khoảng tin cậy', line=dict(color='yellow', width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)'))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()],
                                  y=future_forecast['yhat_upper_smooth'][future_forecast['ds'] > prophet_df['ds'].max()],
                                  mode='lines', name='Khoảng tin cậy tương lai', line=dict(color='pink', width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=future_forecast['ds'][future_forecast['ds'] > prophet_df['ds'].max()],
                                  y=future_forecast['yhat_lower_smooth'][future_forecast['ds'] > prophet_df['ds'].max()],
                                  mode='lines', name='Khoảng tin cậy tương lai', line=dict(color='pink', width=0), fill='tonexty', fillcolor='rgba(255, 192, 203, 0.2)'))
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
        selected_date = st.date_input("Chọn ngày trong tương lai để dự đoán doanh thu:",
                                      min_value=today,
                                      max_value=max_date,
                                      value=today + timedelta(days=30))
        selected_date_df = pd.DataFrame({'ds': [pd.to_datetime(selected_date)]})
        selected_forecast = model.predict(selected_date_df)
        st.markdown(f"**Dự đoán doanh thu cho ngày {selected_date}:**")
        st.write(f"📈 Giá trị dự đoán: **${selected_forecast['yhat'].iloc[0]:,.2f}**")
        st.write(f"📉 Khoảng tin cậy thấp: **${selected_forecast['yhat_lower'].iloc[0]:,.2f}**")
        st.write(f"📊 Khoảng tin cậy cao: **${selected_forecast['yhat_upper'].iloc[0]:,.2f}**")
    else:
        st.error("Không thể thực hiện dự đoán do lỗi tải dữ liệu.")

# Tab 3: Phân Cụm Khách Hàng
with tab3:
    st.header("📀 Phân Cụm Khách Hàng với GMM")
    if not df.empty:
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
    else:
        st.error("Không thể thực hiện phân cụm do lỗi tải dữ liệu.")

# Footer
st.markdown("---")
st.markdown("Web App Demo Đề Án Tốt Nghiệp được xây dựng với Streamlit bởi Ấn Ngọc. Liên hệ hỗ trợ: anngocmukbang@gmail.com")
