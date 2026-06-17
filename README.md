# Lab 02: Dự đoán doanh số bán hàng (Product Sales & Store Sales Forecasting)

Dự án này triển khai các giải pháp Machine Learning nhằm dự đoán doanh số bán hàng phục vụ quản lý tồn kho và chiến lược tiếp thị, kết hợp giữa việc tự xây dựng thuật toán từ đầu (Scratch Models) và đối chiếu với thư viện Scikit-Learn.

---

## 1. Cấu trúc dự án

Dự án được cấu trúc theo chuẩn gói thư viện Python sử dụng mô hình thư mục `src/`:

```text
lab02/
├── data/                       # Dữ liệu đầu vào (Product sales dataset & Store sales dataset)
├── notebooks/                  # Jupyter Notebooks phân tích và huấn luyện
│   ├── product_sales_prediction_complete.ipynb
│   └── store_sales_forecasting_new_data.ipynb
├── src/
│   └── lab02/                  # Gói mã nguồn đóng gói dùng chung
│       ├── __init__.py
│       ├── eda.py              # Các hàm phân tích và vẽ biểu đồ trực quan hóa
│       ├── process.py          # Tiền xử lý dữ liệu và tạo đặc trưng chuỗi thời gian
│       └── model.py            # Thuật toán ML viết từ đầu (Ridge, MLP, HistGradientBoosting)
├── pyproject.toml              # Cấu hình cài đặt gói thư viện lab02
└── README.md
```

---

## 2. Hướng dẫn cài đặt & Chạy dự án

### Cài đặt môi trường ảo và gói thư viện
Để cài đặt gói thư viện ở chế độ chỉnh sửa (editable mode), chạy các lệnh sau trong terminal tại thư mục gốc của dự án:

```bash
# Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt thư viện lab02 ở chế độ editable
pip install -e .
```

---

## 3. Nội dung chính & Notebooks

### 3.1. Dự đoán doanh thu sản phẩm (Product Sales)
* **File**: `notebooks/product_sales_prediction_complete.ipynb`
* **Mục tiêu**: Dự đoán doanh thu phát sinh trên mỗi đơn hàng (`Revenue`).
* **Mô hình sử dụng**:
  - `RidgeRegressionScratch` (Hồi quy Ridge tự viết)
  - `LinearRegressionScratch` (Hồi quy tuyến tính tự viết)
  - `LinearSVRRegressorScratch` (Hồi quy SVR tuyến tính tự viết)
  - Đối chiếu hiệu năng chi tiết với Scikit-Learn thông qua bộ double-check tự động.

### 3.2. Dự báo nhu cầu chuỗi thời gian (Store Sales Forecasting)
* **File**: `notebooks/store_sales_forecasting_new_data.ipynb`
* **Mục tiêu**: Dự báo số lượng hàng bán ra (`sales`) hàng ngày theo cửa hàng và nhóm sản phẩm.
* **Các đặc trưng thời gian & chuỗi thời gian nâng cao**:
  - Tạo lag features (`lag_16`, `lag_28`) và rolling features (`rolling_mean_28`, `rolling_mean_56`) tránh rò rỉ thông tin dữ liệu (leakage).
  - Tạo các đặc trưng tương tác nhân chéo (Interaction features) như `onpromotion * lag_28`, `is_weekend * lag_28`.
* **Kết quả huấn luyện và vượt Baseline**:

| Model | RMSLE | MAE | RMSE | R2 |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting_Scratch_160** | **0.4644** | 90.5285 | 365.3726 | 0.9143 |
| **Ensemble_Blend_HGB_Lag28** (Mô hình Lai) | **0.4831** | **77.8333** | **283.2451** | **0.9485** |
| *Baseline: Lag_28* | 0.6274 | 82.8711 | 311.2467 | 0.9378 |
| **HistGradientBoosting_Scratch (Raw Target)** | 0.9959 | **76.3917** | **260.2313** | **0.9565** |

*Nhận xét:* Mô hình lai **Ensemble_Blend_HGB_Lag28** (kết hợp 60% Boosting và 40% Lag_28) đã xuất sắc vượt qua các baseline mạnh trên mọi chỉ số đánh giá (RMSLE, MAE, RMSE, R2) cùng một lúc.
