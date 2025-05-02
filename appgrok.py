

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import json
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from scipy.interpolate import make_interp_spline
# Thiết lập xác thực Google BigQuery
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:/CÁ NHÂN/terminal/terminal/etl-cap3-27b899b6d343.json"
#client = bigquery.Client(project='etl-cap3')
#st.image("D:/CÁ NHÂN\ĐOÀN ẤN NGỌC.jpg", caption="", width=400)
# Thiết lập tiêu đề và mô tả
# Thiết lập xác thực Google BigQuery
# Thiết lập xác thực Google BigQuery với key nhúng trực tiếp
# Thiết lập xác thực Google BigQuery với key nhúng trực tiếp
try:
    # Service account key nhúng trực tiếp
    service_account_key = {
        "type": "service_account",
        "project_id": "etl-cap3",
        "private_key_id": "245da5598637b821c070805759e56b78de5ea1b9",
        "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDQucMxbrHEXP8I\\nIp9n4VJMVzddafuktm4evdM94BwqVU3/YP6+ssokyNTqu/21s71n3mO5EeyXQOsP\\nW4V/uQzroA6GQMaXhum+5XSNi0kLfaE5SS2A7a6cID40CHpT+XvuTumztOeZsPjX\\nkDe1KiDKAzFVztn7lrVZgcS0aFbRF4/yD7wtOuw53PlqFHWcBlPbVS01y/ga55k/\\nkoSKMbVu18HpEytiK4JOOYngupooCP6nGUChZloqkUqAJ55Aa7asGELCe8oc2zuO\\nk0PB+dqSPMz3d3IjttSdlAR/lMDdQ0zTdfnrQcVWATDOwUaj3016ctk+NyHltSAN\\nDCOpIKQ1AgMBAAECggEAJJyI6/lOLjLfuyYTju2beW+M8xI9hSFIxJSoYoCWgChx\\nY3u4g2uqAHTHwBrUgFQ0Q21jUZmexdMpyApWKtajOQtcsNjY3MhZ5NWrsY4+dEkH\\niBxZYnWOfK09K/RWvLcUsXLfFFoFi9Raebk+xETM9pJDy5jabIJ9ceNMxRZOMCOg\\nMos6MitSxAk5u/5hNUcORziw75RziGa+NoW89CDstfo7N0ttcbFm8QmVoSdcgRk1\\nh4M6yhoDwIzB8sD9vX3gbMpx/Ka25DDR93uJZZnZQsGc6ZZHhs5MgVvmnk94IR1A\\nmEr1i/j7JvUmhNqrwDo/RsWmuo2my0NpRf32sE+IMQKBgQDz9laPC582rdlskf8K\\nPfZJb80LvpXB1RV3mKh8BpKJbY9E3a2WW5RWMTMZg5T0K/6jyXMIAb0iSnHm9a3c\\nK/Gu9SJ9HuSK9bzhRAaIIinwSvACzsiSBGA1HJcK+fy4Erb1Um4MyLiSjUsEJIOU\\nN5MvQz+aZQeEzC7dIOCTYVDM/QKBgQDbBlNJoKUHyqiM9jCKDSma3ZMx+PfPY/qU\\n5CU/pT2LrWUZvm4PwDBLg8x3rJJdbl4kMz9wuXnPQGTQdhEf5vGkVTudSM3q5eAH\\nG4nfuzh2eW59rkNPlGyySOSoJOLiPZ4nH1Jr8H7czXLD1LGYv5uez5exQ9OZYfDJ\\ngsvd/sj1mQKBgQCB/IQcTZ30fWb2wUd5xo/+ivJeyppyn72DbrR++OWFEfofQFoo\\nXf1Ie4h1lLV98B8KV+DRqFOE/NN5sveC82sfxH8u2ZH+NHcCM8OjL0jXPqZpaJJX\\nsrssMWKNN6uezGHtjpESgLRPabdJqsbs5zumGBfAhQofiMeDjOv7uwNZZQKBgHjt\\nTR7hqzxaECWD59jQ25+9pn94zpmawcdjD43MVebAnYlRMsNPgVqwMjwA8+7R+PhT\\nszsxrKdZDlBqX5oVkGUDCh8VC1wd/0rtUXpNjCcddHu/szpE4J6u0FBlzT7Mdn1z\\nNbaS2fhfFnY5ALUtVWahh0dAA5w0DxWzOYwgGCRBAoGANZaaE3xHU0oexWGCXDYs\\nokzjlKZYP9lgoO467jAXNFFRvWBYTtiIdGYzurLQ6LwpGsy8I5ZD/zTMoTCYFNqu\\nJul/yjH3PucYFgjEzdObPR88Fab/GQ6qFRSgH7YoAY7aqWwZgaaJ/PPHIrNQNTYy\nsJ+RMrnoB1iysXyl6y/lcDk=\\n-----END PRIVATE KEY-----\\n",
        "client_email": "etl-cap3@etl-cap3.iam.gserviceaccount.com",
        "client_id": "118374989078209513090",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/etl-cap3%40etl-cap3.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
        }
    credentials = service_account.Credentials.from_service_account_info(service_account_key)
    client = bigquery.Client(project='etl-cap3', credentials=credentials)
except Exception as e:
    st.error(f"BigQuery authentication failed: {e}")
    st.stop()


#
st.title("Đề Án Tốt Nghiệp - Phân Tích Doanh Thu và Phân Cụm Khách Hàng")
st.markdown("""
Ứng dụng này hiển thị phân cụm khách hàng, dự đoán doanh thu, 
và các biểu đồ phân tích dựa trên dữ liệu được lưu ở Google BigQuery sau quá trình ETL Pipeline trước đó.
Bạn vui lòng chọn tab để xem các phân tích chi tiết.
""")

# Tải dữ liệu từ BigQuery
@st.cache_data
def load_data():
    query = """
        SELECT * FROM etl-cap3.Sale_AMZ_ETSY.FinalData
        LIMIT 500000000
    """
    df = client.query(query).to_dataframe()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['year'] = df['Order Date'].dt.year
    return df

# Tải dữ liệu
with st.spinner("Đang tải dữ liệu từ BigQuery..."):
    df = load_data()

# Tạo các tab
tab1, tab2, tab3 = st.tabs(["📊 Tổng Quan Doanh Thu", "💵 Dự Đoán Doanh Thu", "📀 Phân Cụm Khách Hàng"])

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
with tab3:
    st.header("📀 Phân Cụm Khách Hàng với GMM")

    # Lấy mẫu dữ liệu
    df_sample = df.sample(n=35000, random_state=42)
    df_cluster = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit']]

    # Chuẩn hóa và PCA
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster)
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    # Phân cụm với GMM
    gmm = GaussianMixture(n_components=7, random_state=42)
    clusters = gmm.fit_predict(df_pca)
    df_sample['Cluster'] = clusters

    # Vẽ biểu đồ phân cụm
    fig3 = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1], color=clusters.astype(str), title='Phân Cụm Khách Hàng với GMM',
                      labels={'x': 'PCA 1', 'y': 'PCA 2', 'color': 'Cluster'}, color_discrete_sequence=px.colors.qualitative.T10)
    fig3.update_layout(showlegend=True)
    st.plotly_chart(fig3)

    # Hiển thị kết quả phân cụm
    st.subheader("Kết Quả Phân Cụm (Mẫu)")
    st.dataframe(df_sample[['Order Id', 'City', 'Country', 'Cluster']].head())

    # Phân tích đặc trưng từng cụm
    df_analysis = df_sample[['Product Cost', 'Shipping Fee', 'Order Total', 'Profit', 'Cluster']].copy()
    cluster_summary = df_analysis.groupby('Cluster').mean().round(2)
    cluster_counts = df_analysis['Cluster'].value_counts().sort_index()
    cluster_summary['Count'] = cluster_counts

    st.subheader("Đặc Trưng Trung Bình của Từng Cụm")
    st.dataframe(cluster_summary)

    # Đánh giá mô hình phân cụm (từ Tab 4)
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
st.markdown("Web App Demo Đề Án Tốt Nghiệp được xây dựng với Streamlit bởi Ấn Ngọc . Liên hệ hỗ trợ: anngocmukbang@gmail.com")
