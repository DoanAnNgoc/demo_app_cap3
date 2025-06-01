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

# Tải dữ liệu từ Google Drive
@st.cache_data
def load_data():
    # Liên kết Google Drive (thay bằng liên kết Excel công khai)
    shareable_link = "https://docs.google.com/spreadsheets/d/1u2aXzp7gXuKF7qOEx-maeBNDMw7-pbQA"  # Ví dụ: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    try:
        file_id = shareable_link.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        df = pd.read_excel(download_url, engine='openpyxl')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['year'] = df['Order Date'].dt.year
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc tệp Excel từ Google Drive: {e}")
        return pd.DataFrame()

# Tải dữ liệu
with st.spinner("Đang tải dữ liệu từ Google Drive...Vui lòng chờ trong giây lát"):
    df = load_data()

# Tiêu đề và mô tả
st.title("Đề Án Tốt Nghiệp - Phân Tích Tình hình kinh doanh - Dự đoán doanh thu và Phân Cụm Khách Hàng")
st.markdown("""
Ứng dụng này hiển thị phân cụm khách hàng, dự đoán doanh thu, 
và các biểu đồ phân tích dựa trên dữ liệu từ Google Drive.
Bạn vui lòng chọn Tab để xem các phân tích chi tiết.
""")

# Tạo các tab
tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan Tình Hình Kinh Doanh", "💵 Dự Đoán Doanh Thu", "📀 Phân Cụm Khách Hàng"])

# Tab 1: Tổng Quan Doanh Thu
with tab1:
    st.header("📊 Tổng Quan Doanh Thu Theo Năm")
    revenue_by_year = df.groupby('year')['Order Total'].sum().reset_index()

    # Vẽ biểu đồ doanh thu theo năm với Plotly
    fig = px.bar(revenue_by_year, x='year', y='Order Total', title='Tổng Doanh Thu Theo Năm',
                 labels={'year': 'Năm', 'Order Total': 'Tổng Doanh Thu'}, color_discrete_sequence=['red'])
    fig.update_layout(xaxis_tickangle=0, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig)

    # Thêm biểu đồ Tổng Order Total theo Sub-Category (động)
    st.subheader("Tổng Order Total theo Sub-Category Theo Năm")
    
    # Tính tổng Order Total theo Sub-Category và Year
    df['Year'] = df['Order Date'].dt.year
    pivot_data = df.groupby(['Year', 'Sub Category'])['Order Total'].sum().unstack()

    # Lấy danh sách năm
    years = sorted(pivot_data.index)

    # Định nghĩa màu theo năm (3 năm gần nhất)
    year_colors = {
        years[-3] if len(years) >= 3 else years[0]: '#1f77b4',  # blue
        years[-2] if len(years) >= 2 else years[0]: '#2ca02c',  # green
        years[-1] if len(years) >= 1 else years[0]: '#ff7f0e',  # orange
    }

    # Dropdown để chọn năm
    selected_year = st.selectbox("Chọn năm:", years, index=len(years)-1)

    # Vẽ biểu đồ cho năm được chọn
    if years:
        data = pivot_data.loc[selected_year].sort_values()
        top5 = data.nlargest(5).index

        # Đặt màu: top 5 màu khác (đậm), còn lại là màu nhạt
        colors = [year_colors.get(selected_year, 'lightgray') if subcat in top5 else 'lightgray' for subcat in data.index]

        # Tạo figure
        plt.figure(figsize=(10, 6))
        bars = plt.barh(data.index, data.values, color=colors)
        
        # Ghi nhãn giá trị trên cột
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{width:,.2f}', va='center', fontsize=9)

        plt.title(f'Tổng Doanh thu theo Sub-Category - Năm {selected_year}')
        plt.xlabel('Tổng Order Total')
        plt.ylabel('Sub-Category')
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(plt.gcf())
        plt.close()  # Đóng figure để tránh xung đột
    else:
        st.warning("Không có dữ liệu để hiển thị biểu đồ theo Sub-Category.")

    # Bổ sung: Chỉ số thống kê theo Marketplace
    if 'Marketplace' in df.columns:
        st.subheader("💳 Tổng Quan Theo Marketplace")
        summary = df.groupby('Marketplace').agg({
            'Order Total': 'sum',
            'Product Cost': 'sum',
            'Shipping Fee': 'sum',
            'Profit': 'sum'
        }).reset_index()
        summary.columns = ['Marketplace', 'Revenue', 'Cost', 'ShippingFee', 'Profit']
        
        # Hiển thị bảng tổng hợp
        st.write("**Dữ liệu tổng hợp Marketplace**:")
        st.dataframe(summary)

        # Vẽ biểu đồ indicator
        fig2 = go.Figure()
        for i, row in summary.iterrows():
            fig2.add_trace(go.Indicator(
                mode="number+delta",
                value=row['Revenue'],
                delta={'reference': 0, 'valueformat':'.2f'},
                title={"text": f"<b>{row['Marketplace']}</b><br>Revenue"},
                domain={'row': i, 'column': 0}
            ))
            fig2.add_trace(go.Indicator(
                mode="number+delta",
                value=row['Cost'],
                delta={'reference': 0, 'valueformat':'.2f'},
                title={"text": f"<b>{row['Marketplace']}</b><br>Cost"},
                domain={'row': i, 'column': 1}
            ))
            fig2.add_trace(go.Indicator(
                mode="number+delta",
                value=row['Profit'],
                delta={'reference': 0, 'valueformat':'.2f'},
                title={"text": f"<b>{row['Marketplace']}</b><br>Profit"},
                domain={'row': i, 'column': 2}
            ))
        fig2.update_layout(
            grid={'rows': len(summary), 'columns': 3, 'pattern': "independent"},
            height=250 * len(summary),
            title="💳 Tổng Quan Theo Marketplace"
        )
        st.plotly_chart(fig2)
    else:
        st.error("Cột 'Marketplace' không có trong dữ liệu.")

    # Bổ sung: Số lượng đơn hàng theo Marketplace
    if 'Marketplace' in df.columns and 'Order Id' in df.columns:
        st.subheader("Số Lượng Đơn Hàng Theo Sàn")
        grouped = df.groupby('Marketplace').agg({
            'Order Total': 'sum',
            'Order Id': 'count'
        }).reset_index().rename(columns={'Order Id': 'OrderCount'})

        # Gán màu xen kẽ đỏ và xanh lá cây
        colors = ['red', 'green'] * (len(grouped) // 2 + 1)
        grouped['Color'] = colors[:len(grouped)]

        fig3 = px.bar(
            grouped,
            x='Marketplace',
            y='OrderCount',
            title='Số lượng đơn hàng theo sàn',
            text_auto=True,
            color='Color',
            color_discrete_map={'red': 'red', 'green': 'green'}
        )
        fig3.update_layout(
            xaxis_title='Sàn',
            yaxis_title='Số lượng đơn',
            showlegend=False
        )
        st.plotly_chart(fig3)
    else:
        st.warning("Cột 'Marketplace' hoặc 'Order Id' không có trong dữ liệu.")

    # Bổ sung: Top 5 sản phẩm bán chạy nhất (theo Quantity)
    if 'Sub Category' in df.columns and 'Quantity' in df.columns:
        st.subheader("Top 5 Sản Phẩm Bán Chạy Nhất")
        top_products = df.groupby('Sub Category')['Quantity'].sum().sort_values(ascending=False).head(5).reset_index()

        fig4 = px.bar(
            top_products,
            x='Sub Category',
            y='Quantity',
            title='Top 5 sản phẩm bán chạy nhất',
            text_auto=True,
            color_discrete_sequence=['red']
        )
        fig4.update_layout(
            xaxis_title='Tên sản phẩm',
            yaxis_title='Số lượng bán'
        )
        st.plotly_chart(fig4)
    else:
        st.warning("Cột 'Sub Category' hoặc 'Quantity' không có trong dữ liệu.")

    # Bổ sung: Bản đồ doanh thu theo thành phố
    if 'City' in df.columns and 'Country' in df.columns:
        st.subheader("Doanh Thu Theo Thành Phố")
        city_group = df.groupby(['City', 'Country']).agg({'Order Total': 'sum'}).reset_index()

        # Hiển thị theo Country do thiếu lat/lon
        fig5 = px.scatter_geo(
            city_group,
            locations="Country",
            locationmode="country names",
            color="Order Total",
            size="Order Total",
            hover_name="City",
            scope='world',
            title='Doanh thu theo thành phố',
            size_max=20
        )
        st.plotly_chart(fig5)

# Tab 2: Dự Đoán Doanh Thu
with tab2:
    st.header("💵 Dự Đoán Doanh Thu với Prophet")

    # Chuẩn bị dữ liệu cho Prophet
    prophet_df = df[['Order Date', 'Order Total']].rename(columns={'Order Date': 'ds', 'Order Total': 'y'})
    prophet_df = prophet_df.groupby('ds').sum().reset_index()

    # Làm mượt dữ liệu thực tế
    prophet_df['y_smooth'] = prophet_df['y'].rolling(window=5, center=True, min_periods=1).mean()
    prophet_df['ds_numeric'] = prophet_df['ds'].apply(lambda x: x.timestamp())
    prophet_df = prophet_df.sort_values('ds_numeric')

    # Nội suy để làm mượt
    x = prophet_df['ds_numeric']
    y = prophet_df['y_smooth']
    x_smooth = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    ds_smooth = pd.to_datetime(x_smooth, unit='s')

    # Huấn luyện mô hình Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.01)
    model.fit(prophet_df)

    # Biểu đồ 1: Thực tế và dự đoán trong phạm vi dữ liệu gốc
    past_future = prophet_df[['ds']].copy()
    past_forecast = model.predict(past_future)
    past_forecast['yhat_smooth'] = past_forecast['yhat'].rolling(window=5, center=True, min_periods=1).mean()
    past_forecast['yhat_lower_smooth'] = past_forecast['yhat_lower'].rolling(window=5, center=True, min_periods=1).mean()
    past_forecast['yhat_upper_smooth'] = past_forecast['yhat_upper'].rolling(window=5, center=True, min_periods=1).mean()

    # Vẽ biểu đồ với Plotly
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ds_smooth, y=y_smooth, mode='lines', name='Thực tế', line=dict(color='blue', width=1)))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_smooth'], mode='lines', name='Dự đoán', line=dict(color='orange', width=1)))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_upper_smooth'], mode='lines', name='Khoảng tin cậy (trên)', line=dict(color='yellow', width=0), showlegend=False))
    fig1.add_trace(go.Scatter(x=past_forecast['ds'], y=past_forecast['yhat_lower_smooth'], mode='lines', name='Khoảng tin cậy (dưới)', line=dict(color='yellow', width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)'))
    fig1.update_layout(title='Giá trị bán hàng hàng tháng - Prophet (Tập gốc)', xaxis_title='Ngày', yaxis_title='Giá trị bán hàng', xaxis_tickformat='%Y-%m', xaxis_tickangle=45, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig1)

    # Biểu đồ 2: Dự đoán 12 tháng tiếp theo
    future = model.make_future_dataframe(periods=365, freq='D')
    future_forecast = model.predict(future)
    future_forecast['yhat_smooth'] = future_forecast['yhat'].rolling(window=5, center=True, min_periods=1).mean()
    future_forecast['yhat_lower_smooth'] = future_forecast['yhat_lower'].rolling(window=5, center=True, min_periods=1).mean()
    future_forecast['yhat_upper_smooth'] = future_forecast['yhat_upper'].rolling(window=5, center=True, min_periods=1).mean()

    # Vẽ biểu đồ dự đoán tương lai
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

    # Hiển thị các chỉ số đánh giá
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

    # Thêm mục chọn ngày để dự đoán doanh thu
    st.subheader("Dự Đoán Doanh Thu Cho Ngày Cụ Thể")
    today = datetime.today().date()
    max_date = today + timedelta(days=365)  # Giới hạn 1 năm từ hôm nay
    selected_date = st.date_input("Chọn ngày trong tương lai để dự đoán doanh thu:", 
                                  min_value=today, 
                                  max_value=max_date, 
                                  value=today + timedelta(days=30))

    # Dự đoán cho ngày được chọn
    selected_date_df = pd.DataFrame({'ds': [pd.to_datetime(selected_date)]})
    selected_forecast = model.predict(selected_date_df)

    # Hiển thị kết quả dự đoán
    st.markdown(f"**Dự đoán doanh thu cho ngày {selected_date}:**")
    st.write(f"📈 Giá trị dự đoán: **${selected_forecast['yhat'].iloc[0]:,.2f}**")
    st.write(f"📉 Khoảng tin cậy thấp: **${selected_forecast['yhat_lower'].iloc[0]:,.2f}**")
    st.write(f"📊 Khoảng tin cậy cao: **${selected_forecast['yhat_upper'].iloc[0]:,.2f}**")

# Tab 3: Phân Cụm Khách Hàng
# Tab 3: Phân Cụm Khách Hàng
with tab3:
    st.header("📀 Phân Cụm Khách Hàng với GMM")

    # Lấy mẫu dữ liệu
    df_sample = df.sample(n=40000, random_state=42)
    df_cluster = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit']]

    # Chuẩn hóa và PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    # Bước 1: Xác định số cụm tối ưu bằng AIC và BIC
    k_range = range(1, 11)  # Thử từ 1 đến 10 cụm
    aic_scores = []
    bic_scores = []
    models = []

    with st.spinner("Đang tính toán số cụm tối ưu..."):
        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
            gmm.fit(df_pca)
            aic_scores.append(gmm.aic(df_pca))
            bic_scores.append(gmm.bic(df_pca))
            models.append(gmm)

    # Vẽ biểu đồ AIC và BIC với Plotly
    fig_aic_bic = go.Figure()
    fig_aic_bic.add_trace(go.Scatter(x=list(k_range), y=aic_scores, mode='lines+markers', name='AIC', line=dict(color='#1f77b4')))
    fig_aic_bic.add_trace(go.Scatter(x=list(k_range), y=bic_scores, mode='lines+markers', name='BIC', line=dict(color='#ff7f0e', dash='dash')))
    fig_aic_bic.update_layout(
        title='AIC và BIC theo số cụm',
        xaxis_title='Số cụm (k)',
        yaxis_title='Score',
        yaxis=dict(griddash='dash', gridcolor='gray'),
        showlegend=True
    )
    st.plotly_chart(fig_aic_bic)

    # Tìm số cụm tối ưu
    optimal_k_aic = k_range[np.argmin(aic_scores)]
    optimal_k_bic = k_range[np.argmin(bic_scores)]
    st.write(f"**Số cụm tối ưu theo AIC:** {optimal_k_aic}")
    st.write(f"**Số cụm tối ưu theo BIC:** {optimal_k_bic}")

    # Cho phép người dùng chọn số cụm
    selected_k = st.slider("Chọn số cụm để phân cụm:", min_value=1, max_value=10, value=optimal_k_aic)

    # Bước 2: Phân cụm với số cụm được chọn
    with st.spinner("Đang thực hiện phân cụm..."):
        gmm_optimal = GaussianMixture(n_components=selected_k, random_state=42, n_init=10)
        clusters = gmm_optimal.fit_predict(df_pca)
        df_sample['Cluster'] = clusters

    # Bước 3: Vẽ biểu đồ phân cụm với Plotly
    fig_cluster = px.scatter(
        x=df_pca[:, 0], 
        y=df_pca[:, 1], 
        color=clusters.astype(str), 
        title=f'Phân Cụm Khách Hàng với GMM (k={selected_k})',
        labels={'x': 'PCA 1', 'y': 'PCA 2', 'color': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.T10
    )
    fig_cluster.update_traces(marker=dict(size=5))
    fig_cluster.update_layout(showlegend=True, yaxis=dict(griddash='dash', gridcolor='gray'))
    st.plotly_chart(fig_cluster)

    # Bước 4: Hiển thị kết quả phân cụm
    st.subheader("Kết Quả Phân Cụm (Mẫu)")
    st.dataframe(df_sample[['Order Id', 'City', 'Country', 'Cluster']].head())

    # Phân tích đặc trưng từng cụm
    df_analysis = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit', 'Cluster']].copy()
    cluster_summary = df_analysis.groupby('Cluster').mean().round(2)
    cluster_counts = df_analysis['Cluster'].value_counts().sort_index()
    cluster_summary['Count'] = cluster_counts

    st.subheader("Đặc Trưng Trung Bình của Từng Cụm")
    st.dataframe(cluster_summary)

    # Đánh giá mô hình phân cụm
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

st.markdown("---")
st.markdown("Web App Demo Đề Án Tốt Nghiệp được xây dựng với Streamlit bởi Ấn Ngọc. Liên hệ hỗ trợ: anngocmukbang@gmail.com")
