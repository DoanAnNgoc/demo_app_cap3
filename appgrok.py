import streamlit as st
import pandas as pd
import logging
import csv

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tải dữ liệu từ file .gz trong repository
@st.cache_data
def load_data():
    try:
        file_path = "Data_Streamlit.csv.gz"  # Đường dẫn tới file .gz
        logger.info(f"Bắt đầu đọc file {file_path}...")
        
        # Thử đọc với nhiều encoding
        encodings = ['utf-8', 'latin1', 'iso-8859-1']
        df = None
        for encoding in encodings:
            try:
                logger.info(f"Thử đọc với encoding: {encoding}")
                df = pd.read_csv(
                    file_path,
                    compression='gzip',
                    encoding=encoding,
                    on_bad_lines='skip',  # Tạm thời bỏ qua dòng lỗi
                    quoting=csv.QUOTE_NONNUMERIC,  # Bao quanh giá trị văn bản bằng dấu ngoặc kép
                    low_memory=False
                )
                logger.info("Đọc file thành công")
                break
            except Exception as e:
                logger.warning(f"Lỗi với encoding {encoding}: {str(e)}")
                if encoding == encodings[-1]:
                    st.error("Không thể đọc file với bất kỳ encoding nào.")
                    logger.error(f"Thất bại với tất cả encoding: {str(e)}")
                    return None
        
        # Ghi log và hiển thị thông tin dữ liệu gốc
        logger.info(f"Dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột")
        st.write(f"**Thông tin dữ liệu gốc**: {df.shape[0]} dòng, {df.shape[1]} cột")
        st.write(f"**Cột**: {list(df.columns)}")
        
        # Kiểm tra số dòng
        if df.shape[0] < 270000:
            st.warning(f"Dữ liệu chỉ có {df.shape[0]} dòng, ít hơn 270,000 dòng mong đợi! Có thể do dòng lỗi bị bỏ qua.")
        elif df.shape[0] > 270000:
            st.warning(f"Dữ liệu có {df.shape[0]} dòng, nhiều hơn 270,000 dòng mong đợi!")
        
        # Kiểm tra số cột
        expected_cols = 22
        if df.shape[1] != expected_cols:
            st.warning(f"File có {df.shape[1]} cột, mong đợi {expected_cols} cột!")
        
        # Kiểm tra giá trị NaN
        nan_counts = df[['Order Date', 'Order Total']].isna().sum()
        st.write(f"**Giá trị NaN**:")
        st.write(f"- Order Date: {nan_counts['Order Date']} dòng")
        st.write(f"- Order Total: {nan_counts['Order Total']} dòng")
        
        # Xử lý dữ liệu
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['year'] = df['Order Date'].dt.year
        
        # Đảm bảo các cột số
        numeric_cols = ['Order Total', 'Product Cost', 'Shipping Fee', 'Profit', 'Quantity', 'Marketplace Fee']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    st.warning(f"Cột '{col}' có {nan_count} giá trị NaN.")
        
        # Tính Profit nếu cần
        required_cols = ['Order Total', 'Product Cost', 'Shipping Fee', 'Quantity']
        if all(col in df.columns for col in required_cols):
            if df['Profit'].isna().any():
                if 'Marketplace Fee' in df.columns:
                    df['Profit'] = df['Order Total'] - (df['Product Cost'] * df['Quantity']) - df['Shipping Fee'] - df['Marketplace Fee'].fillna(0)
                    logger.info("Tính Profit với Marketplace Fee")
                else:
                    df['Profit'] = df['Order Total'] - (df['Product Cost'] * df['Quantity']) - df['Shipping Fee']
                    logger.info("Tính Profit không có Marketplace Fee")
                    st.warning("Cột 'Marketplace Fee' không có, tính Profit mà không trừ phí sàn.")
        else:
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"Thiếu cột cần thiết để tính Profit: {missing_cols}")
        
        # Kiểm tra Marketplace
        if 'Marketplace' in df.columns:
            st.write(f"**Số lượng Marketplace**: {df['Marketplace'].nunique()}")
            st.write(f"**Danh sách Marketplace**: {df['Marketplace'].unique().tolist()}")
        else:
            st.error("Cột 'Marketplace' không có trong dữ liệu.")
        
        # Loại bỏ dòng thiếu Order Date hoặc Order Total
        df_cleaned = df.dropna(subset=['Order Date', 'Order Total'])
        st.write(f"**Dữ liệu sau khi loại NaN (Order Date, Order Total)**: {df_cleaned.shape[0]} dòng")
        
        return df_cleaned
    
    except Exception as e:
        st.error(f"Lỗi khi đọc file .gz: {str(e)}")
        logger.error(f"Lỗi: {str(e)}")
        return None
