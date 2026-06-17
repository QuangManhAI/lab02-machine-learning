# Requirements - Dự đoán doanh số bán hàng (Lab 02)

Tài liệu này tóm tắt yêu cầu, dữ liệu đầu vào, đầu ra và các bước triển khai của bài toán dự đoán doanh số trong bài thực hành Lab 02.

---

## 1. Purpose - Mục tiêu
Xây dựng hệ thống dự báo doanh số bán hàng qua hai bài toán thực tế:
1. **Dự đoán doanh thu sản phẩm**: Dự đoán số tiền doanh thu Revenue phát sinh từ các đơn hàng đơn lẻ dựa trên thông tin sản phẩm và khách hàng.
2. **Dự báo nhu cầu chuỗi thời gian**: Dự đoán số lượng hàng bán ra sales hàng ngày cho từng cửa hàng và nhóm sản phẩm nhằm tối ưu hóa tồn kho.

Yêu cầu đặc biệt là tự viết các thuật toán học máy từ đầu như Ridge Regression, MLP Regressor, và HistGradientBoosting Regressor để đối chiếu với thư viện Scikit-Learn.

---

## 2. Input - Dữ liệu đầu vào
* **Bài toán Product Sales**: 
  - File dữ liệu: data/product_sales_dataset_final.csv
  - Các đặc trưng chính: Ngày đặt hàng, nhóm hàng Category, nhóm phụ Sub-Category, tên sản phẩm, khu vực Region, bang, thành phố, số lượng Quantity, lợi nhuận Profit.
* **Bài toán Store Sales**:
  - Thư mục dữ liệu: data/new_data/
  - Các file bao gồm: train.csv - lịch sử bán hàng, test.csv - danh sách cần dự báo, stores.csv - thông tin cửa hàng, oil.csv - giá dầu hàng ngày, holidays_events.csv - sự kiện ngày lễ, và transactions.csv - lượng giao dịch.

---

## 3. Output - Kết quả đầu ra
1. **Thư viện mã nguồn dùng chung trong src/lab02/**:
   - eda.py: Chứa các hàm vẽ biểu đồ phân phối và phân tích doanh số nhóm.
   - process.py: Pipeline làm sạch dữ liệu, tạo đặc trưng chuỗi thời gian lag, rolling mean và gộp bảng.
   - model.py: Chứa các class mô hình tự viết từ đầu như Ridge, MLP, HistGradientBoosting tương thích với chuẩn fit và predict của sklearn.
2. **Các Notebook chạy hoàn chỉnh**:
   - notebooks/product_sales_prediction_complete.ipynb: Chạy thông suốt với mô hình tự viết, đánh giá qua R2, RMSE, MAE.
   - notebooks/store_sales_forecasting_new_data.ipynb: Chạy thông suốt, tích hợp các mô hình lai để vượt qua các baseline chuỗi thời gian mạnh trên chỉ số RMSLE, MAE và RMSE.

---

## 4. How to do - Các bước thực hiện

### Bước 1: EDA & Khám phá dữ liệu
* Phân tích phân phối của biến mục tiêu target.
* Tìm và xử lý giá trị khuyết thiếu như giá dầu thiếu ngày nghỉ bằng phương pháp điền trước hoặc điền sau ffill, bfill.
* Xác định các dòng trùng lặp và loại bỏ.

### Bước 2: Tạo đặc trưng - Feature Engineering
* **Thời gian**: Trích xuất ngày, tháng, năm, ngày trong tuần, ngày trong tháng, cuối tuần.
* **Chuỗi thời gian**: Tạo các đặc trưng trễ lag_16, lag_28 và trung bình trượt rolling_mean_28, rolling_mean_56 với độ dịch tối thiểu là 16 ngày bằng độ dài tập test để tránh rò rỉ thông tin data leakage.
* **Tương tác**: Nhân chéo các cột quan trọng ví dụ onpromotion * lag_28, is_weekend * lag_28 giúp mô hình học nhanh hơn.

### Bước 3: Thiết lập Pipeline & Huấn luyện
* Chia tập train và validation theo trục thời gian ví dụ lấy 16 ngày cuối làm validation.
* Sử dụng ColumnTransformer để xử lý đặc trưng: mã hóa cột phân loại, chuẩn hóa bằng StandardScaler cho cột số áp dụng cho Ridge và MLP.
* Huấn luyện mô hình tự viết trên cả hai dạng target: target gốc và target đã biến đổi logarit log1p.

### Bước 4: Tối ưu & Vượt Baseline
* **Huấn luyện mô hình target gốc**: Để mô hình tập trung tối ưu hóa các chỉ số tuyệt đối MAE, RMSE.
* **Mô hình Lai - Ensemble**: Tổ hợp tuyến tính ví dụ 0.6 * Pred_HGB + 0.4 * Pred_Lag28 để trung hòa giữa sai số tỷ lệ phần trăm RMSLE và sai số tuyệt đối MAE, RMSE.

### Bước 5: Đóng gói thư viện
* Trích xuất các hàm và class dùng chung từ Notebook ra các file python trong src/lab02/.
* Cài đặt gói ở chế độ editable bằng pip install -e . và gọi lại từ các Notebook bằng lệnh import tiêu chuẩn.
