goal: refactor /Users/quangmanh/Project/lab02/notebooks/store_sales_forecasting_new_data.ipynb

struct:
- Bước 0: Xác định bài toán (RMSLE, validation 16 ngày cuối)
- Bước 1: Đọc dữ liệu (train, test, stores, oil, holidays, transactions)
- Bước 2: EDA tổng quan (target, zero sales, family, thời gian, promotion, store, oil, holidays)
- Bước: xử lí missing data, duplicate data. cân nhắc có nên chuẩn hoá giá trị không? dựa vào biểu đồ ma trận tương quan
xem xét chọn các đặc trưng sao cho hợp lý.
- Bước 3: Tạo đặc trưng (calendar, store, holiday, oil, lag/rolling sales)
- Bước 4: Tạo tập train/validation theo thời gian (train 365 ngày, validation 16 ngày cuối)
- Bước 5: Metric (RMSLE) và tiền xử lý (OneHot cho Ridge/MLP, Ordinal cho HistGradientBoosting)
- Bước 6: Baseline time-series (Lag_16, Lag_28, Rolling_Mean_28/56) giải thích baseline là cái gì ?
- Bước 7: Huấn luyện và so sánh mô hình ML (Ridge, MLP, HistGradientBoosting -  dùng hoàn toàn bản tự code scratch từ model.py từ đầu đến cuối luôn) - chỉ dùng 1 cell để so sánh hiệu năng và check lại với sklearn(gọi cái lớp trong model.py)
- Bước 8: Chẩn đoán mô hình (thực tế vs dự báo, phân phối residuals, RMSLE theo cửa hàng/nhóm sản phẩm)
- Bước 9: Train final model (HistGradientBoosting 160 iterations) & tạo submission
- Bước 10: Tổng kết & Hướng cải thiện
 
 *NOTICE: Đây là kiến trúc notebook bạn phải reafactor theo nhằm tối ưu code và cell md. không gây ảnh hưởng tới kết quả của notebook và các yếu tố khác.* 