# Complete Plan: Predicting Product Sales

## 1. Muc tieu bai lam

Xay dung mot pipeline machine learning de du doan doanh so san pham cua mot cong ty e-commerce. Bien muc tieu la `product sales`, dang so lien tuc, nen day la bai toan hoi quy.

Ket qua cuoi cung can co:

- Bo du lieu duoc lua chon, mo ta ro nguon, y nghia cot va chat luong du lieu.
- EDA day du de hieu data truoc khi modeling.
- Pipeline xu ly du lieu gom clean data, encode categorical features, xu ly missing values va outliers.
- Feature engineering phu hop voi bai toan sales.
- So sanh nhieu mo hinh regression.
- Bao cao metric MSE, MAE, RMSE, R2 tren test set.
- Ket luan mo hinh tot nhat va de xuat cach trien khai / cai tien.

## 2. Pham vi va gia dinh

Yeu cau goc mo ta du lieu ly tuong gom:

- Product information: category, price, description.
- Customer demographics: age, gender, location.
- Purchase history: previous purchases, purchase frequency.
- Marketing campaign data: campaign, ad spend.
- Seasonal and holiday information.
- Target: product sales.

Trong thuc te, dataset Kaggle co the khong co du tat ca nhom cot tren. Vi vay can:

- Uu tien dataset co transaction/order-level data.
- Neu thieu campaign/holiday, co the tao feature theo ngay thang: month, quarter, weekday, weekend, holiday flag neu co ngay mua hang.
- Neu thieu customer demographics, van co the lam tot voi product, order history va time features.
- Neu target goc khong co san, target co the duoc tao bang `quantity * unit_price` hoac tong sales theo product/time period.

## 3. Ke hoach Data

### 3.1. Tieu chi chon dataset

Uu tien chon dataset theo thu tu:

1. Co du lieu e-commerce / online retail thuc te.
2. Co cot san pham, gia, so luong, ngay giao dich va ma khach hang.
3. Co the tao target sales ro rang.
4. Co du so dong de train/test co y nghia.
5. License cho phep dung hoc tap.
6. It loi nghiem trong nhu qua nhieu missing target, duplicate bat thuong, gia am khong giai thich duoc.

Dataset de xuat:

- Online Retail Dataset: phu hop vi co invoice, stock code, description, quantity, invoice date, unit price, customer id, country.
- E-commerce Dataset tren Kaggle neu co them campaign/customer/product category thi tot hon.

### 3.2. Data dictionary can lap

Sau khi chon dataset, tao bang mo ta cot:

| Cot | Kieu du lieu | Y nghia | Vai tro |
| --- | --- | --- | --- |
| invoice/order id | categorical | Ma don hang | group key / duplicate check |
| product id/stock code | categorical | Ma san pham | feature |
| description/name | text | Ten/mo ta san pham | feature phu, co the clean text |
| quantity | numeric | So luong mua | tao target / feature |
| unit price/price | numeric | Gia san pham | feature |
| invoice date/order date | datetime | Ngay mua hang | tao seasonal features |
| customer id | categorical | Ma khach hang | tao history features |
| country/location | categorical | Khu vuc | feature |
| sales | numeric | Doanh so | target |

Neu dataset khong co `sales`, tao:

```text
sales = quantity * unit_price
```

Can xac dinh don vi du doan:

- Option A: du doan sales tren tung dong giao dich.
- Option B: gom nhom theo `product_id + date/week/month` de du doan doanh so san pham theo thoi gian.

Khuyen nghi cho bai nay: dung Option B neu co ngay thang, vi gan hon voi bai toan toi uu ton kho va marketing.

### 3.3. Quy trinh nap va kiem tra data

Cac buoc bat buoc:

1. Doc file bang `pandas`.
2. Kiem tra shape: so dong, so cot.
3. Kiem tra ten cot, chuan hoa thanh snake_case.
4. Kiem tra data type tung cot.
5. Parse cot ngay thang ve `datetime`.
6. Kiem tra missing values theo so luong va ty le.
7. Kiem tra duplicate rows va duplicate invoice/order.
8. Kiem tra gia tri am/0 bat thuong:
   - `quantity <= 0`
   - `price <= 0`
   - `sales < 0`
9. Kiem tra outliers bang quantile/IQR.
10. Tach train/test dung theo thoi gian neu data co time component.

### 3.4. Data cleaning

Quy tac xu ly de xuat:

- Xoa dong khong co target hoac khong tao duoc target.
- Xoa/cach ly don hang huy neu quantity am va dataset giai thich do la return/cancel.
- Xoa price <= 0 neu khong co ly do nghiep vu hop le.
- Impute missing category/location bang `Unknown`.
- Impute numeric bang median trong pipeline, khong impute truoc khi split neu co nguy co leakage.
- Chuan hoa text description: strip spaces, lowercase neu dung lam feature.
- Giu log so dong bi loai o moi buoc.

### 3.5. Data splitting

Neu co cot ngay:

- Sort theo date.
- Train: cac moc thoi gian cu.
- Test: giai doan gan nhat, vi mo phong du doan tuong lai.
- Co the dung validation set ngay truoc test de tuning.

Neu khong co cot ngay:

- Dung random train/test split.
- Fix `random_state` de ket qua lap lai duoc.

Khuyen nghi:

```text
Train: 70%
Validation: 15%
Test: 15%
```

Hoac:

```text
Train: 80%
Test: 20%
```

## 4. Ke hoach EDA

EDA la phan can lam ky nhat vi no quyet dinh cach clean, feature engineering va chon mo hinh.

### 4.1. EDA tong quan

Can tra loi:

- Dataset co bao nhieu records va features?
- Moi cot thuoc kieu nao?
- Target sales phan phoi nhu the nao?
- Co missing values nghiem trong khong?
- Co duplicate hoac transaction bat thuong khong?
- Sales co bi skew nang khong?
- Co outliers lon anh huong den model khong?

Bang/plot can co:

- `df.head()`, `df.info()`, `df.describe()`.
- Missing value table.
- Duplicate count.
- Histogram/KDE cua target sales.
- Boxplot cua sales, quantity, price.
- Top 10 gia tri lon nhat/nho nhat cua sales.

### 4.2. EDA target `sales`

Can phan tich:

- Min, max, mean, median, std.
- Percentile: p1, p5, p25, p50, p75, p95, p99.
- Do lech skewness.
- Ty le sales = 0, sales < 0 neu co.
- So sanh target truoc va sau khi xu ly outliers.

Neu target skew manh:

- Thu log transform: `log1p(sales)`.
- Danh gia mo hinh tren scale goc bang inverse transform.

Plot can co:

- Histogram sales.
- Histogram `log1p(sales)`.
- Boxplot sales theo category/country/time.

### 4.3. EDA product

Can tra loi:

- San pham nao ban chay nhat theo quantity?
- San pham nao tao doanh thu cao nhat?
- Category nao co doanh thu cao/thap?
- Gia san pham phan phoi the nao?
- Gia co lien quan den sales khong?

Neu co category:

- Bar chart top categories by revenue.
- Boxplot sales by category.
- Average sales/category.

Neu khong co category:

- Dung product description/product id de thong ke top products.
- Co the tao feature don gian tu description: do dai mo ta, keyword count, hoac nhom theo product id.

### 4.4. EDA customer

Can tra loi:

- Co bao nhieu customer?
- Sales moi customer phan phoi the nao?
- Tan suat mua hang cua customer?
- Customer repeat vs one-time buyer khac nhau ra sao?
- Location/country nao co doanh thu cao?

Feature co the tao sau EDA:

- `customer_total_spend`
- `customer_order_count`
- `customer_avg_order_value`
- `customer_purchase_frequency`
- `customer_recency_days`

Chu y leakage: cac feature history phai tinh chi tu du lieu qua khu, khong duoc dung thong tin tu tuong lai so voi dong can du doan.

### 4.5. EDA time/seasonality

Neu co cot ngay, day la phan quan trong:

- Sales theo ngay/tuan/thang.
- So sanh weekday vs weekend.
- Month/quarter nao ban tot?
- Co ngay le/mua cao diem nao khong?
- Sales co trend tang/giam theo thoi gian khong?

Plot can co:

- Line chart monthly sales.
- Line chart weekly sales.
- Bar chart sales by month.
- Bar chart sales by weekday.
- Heatmap month x weekday neu phu hop.

Feature nen tao:

- `year`
- `month`
- `quarter`
- `weekofyear`
- `dayofweek`
- `is_weekend`
- `is_month_start`
- `is_month_end`
- `is_holiday` neu co holiday calendar.

### 4.6. EDA marketing/campaign

Neu dataset co campaign/ad spend:

- Sales theo campaign.
- Correlation giua ad spend va sales.
- ROI gan dung: `sales / ad_spend`.
- Campaign nao co sales cao nhung chi phi thap?

Neu dataset khong co campaign:

- Ghi ro han che.
- Co the dung time spikes/holiday flags nhu proxy cho campaign/seasonal effects, nhung khong khang dinh la campaign effect.

### 4.7. EDA relationship voi target

Can lam:

- Correlation matrix cho numeric features.
- Scatter price vs sales.
- Scatter quantity vs sales neu quantity khong phai thanh phan tao target.
- Boxplot sales theo categorical features.
- Groupby categorical features: mean/median/count sales.
- Mutual information hoac feature importance sau baseline model.

Can canh bao:

- Neu `sales = quantity * price`, khong nen dua truc tiep `quantity` vao model neu muc tieu la du doan sales truoc khi biet so luong ban. Neu dua vao, model se qua de va khong phan anh bai toan that.
- Nen xac dinh thoi diem du doan: truoc chien dich, truoc thang ban hang, hay tai transaction.

## 5. Feature Engineering

Nhom feature de xuat:

### 5.1. Product features

- Product/category encoded.
- Price.
- Product historical sales lag/rolling mean.
- Product popularity: order count, customer count, average rating/review neu co.

### 5.2. Customer features

- Customer lifetime value.
- Number of previous purchases.
- Purchase frequency.
- Recency.
- Average basket value.

### 5.3. Time features

- Month, quarter, week, day of week.
- Weekend flag.
- Holiday flag.
- Lag sales: sales previous week/month.
- Rolling mean sales: 7-day/30-day neu du lieu ngay.

### 5.4. Marketing features

- Campaign indicator.
- Ad spend.
- Discount/promotion flag neu co.
- Campaign type encoded.

### 5.5. Encoding va scaling

- Numeric: median imputation, optional standard scaling cho linear models.
- Categorical low-cardinality: OneHotEncoder.
- Categorical high-cardinality: frequency encoding hoac target encoding neu lam can than trong CV.
- Text description: co the dung TF-IDF neu can, nhung chi nen them sau baseline.

## 6. Modeling Plan

### 6.1. Baseline

Bat dau bang cac baseline:

- Predict mean sales.
- Predict median sales.
- Linear Regression / Ridge Regression.

Muc dich: co moc so sanh don gian.

### 6.2. Candidate models

Thu cac mo hinh:

- Linear Regression.
- Ridge/Lasso.
- Decision Tree Regressor.
- Random Forest Regressor.
- Gradient Boosting Regressor.
- HistGradientBoostingRegressor neu dataset lon.

Neu du lieu co time series ro:

- Baseline theo lag/rolling features.
- Xem xet ARIMA/Prophet neu bai yeu cau forecasting theo chuoi thoi gian tong hop.

### 6.3. Hyperparameter tuning

Dung:

- `RandomizedSearchCV` cho Random Forest/Gradient Boosting.
- `TimeSeriesSplit` neu split theo thoi gian.
- `KFold` hoac `train_test_split` neu data khong co time.

Metric toi uu chinh:

- RMSE neu muon phat nang sai so lon.
- MAE neu muon metric de giai thich voi business.

## 7. Evaluation Plan

Metric can bao cao:

- MAE: sai so trung binh theo don vi sales.
- MSE: sai so binh phuong trung binh.
- RMSE: sai so trung binh da dua ve don vi sales.
- R2: ty le bien thien duoc giai thich.

Phan tich sau evaluation:

- So sanh metric train vs test de phat hien overfitting.
- Plot predicted vs actual.
- Plot residuals.
- Residual theo time/category/price range de xem model sai o dau.
- Feature importance/permutation importance cho model tot nhat.

Bang ket qua can co:

| Model | MAE | MSE | RMSE | R2 | Ghi chu |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline mean | | | | | moc so sanh |
| Linear/Ridge | | | | | de giai thich |
| Decision Tree | | | | | de bat nonlinear |
| Random Forest | | | | | manh nhung nang |
| Gradient Boosting | | | | | ung vien chinh |

## 8. Deployment Plan

Muc tieu deployment trong pham vi lab:

- Luu trained model bang `joblib`.
- Luu pipeline preprocessing + model chung mot object.
- Tao script/notebook demo predict cho input moi.
- Viet huong dan input schema.

File dau ra de xuat:

```text
artifacts/
  best_model.joblib
  metrics.json
  feature_columns.json
reports/
  eda_summary.md
  model_report.md
notebooks/
  01_data_eda.ipynb
  02_modeling.ipynb
src/
  data_preparation.py
  features.py
  train.py
  evaluate.py
  predict.py
```

## 9. Checklist thuc hien

### Data

- [ ] Chon dataset va ghi nguon.
- [ ] Tai data vao thu muc `data/raw`.
- [ ] Tao data dictionary.
- [ ] Chuan hoa ten cot.
- [ ] Xac dinh target `sales`.
- [ ] Kiem tra missing values.
- [ ] Kiem tra duplicates.
- [ ] Kiem tra outliers.
- [ ] Xu ly gia tri am/0 bat thuong.
- [ ] Luu data da clean vao `data/processed`.

### EDA

- [ ] Tong quan shape, info, describe.
- [ ] Missing value report.
- [ ] Target distribution.
- [ ] Outlier analysis.
- [ ] Product/category sales analysis.
- [ ] Customer/location sales analysis.
- [ ] Time/seasonality analysis neu co date.
- [ ] Campaign/ad spend analysis neu co cot lien quan.
- [ ] Correlation/relationship voi target.
- [ ] Tong hop insight thanh `reports/eda_summary.md`.

### Modeling

- [ ] Tao train/validation/test split.
- [ ] Xay preprocessing pipeline.
- [ ] Train baseline.
- [ ] Train Linear/Ridge.
- [ ] Train Decision Tree.
- [ ] Train Random Forest.
- [ ] Train Gradient Boosting.
- [ ] Tune hyperparameters cho model tot.
- [ ] Chon best model theo validation/test metrics.

### Evaluation va bao cao

- [ ] Tinh MAE, MSE, RMSE, R2.
- [ ] Ve predicted vs actual.
- [ ] Ve residual plot.
- [ ] Phan tich feature importance.
- [ ] Viet ket luan business.
- [ ] Luu model va metrics.

## 10. Rui ro va cach xu ly

| Rui ro | Anh huong | Cach xu ly |
| --- | --- | --- |
| Dataset thieu campaign/ad spend | Khong phan tich duoc marketing truc tiep | Ghi ro limitation, dung seasonal/time features thay the |
| Target tao tu quantity * price | Leakage neu dung quantity lam feature | Loai quantity khoi feature neu du doan sales truoc ban hang |
| Sales skew manh | Model bi anh huong boi outliers | Thu log1p target, robust metrics, cap outliers neu hop ly |
| Time leakage | Test metric ao | Split theo thoi gian, tinh historical features chi tu qua khu |
| High-cardinality product/customer id | One-hot qua lon | Dung frequency encoding, aggregate history features |
| Missing customer id | Khong tao duoc customer features | Impute Unknown hoac tap trung vao product/time features |

## 11. Timeline de xuat

1. Ngay 1: chon dataset, nap data, tao data dictionary, clean so bo.
2. Ngay 2: lam EDA chi tiet va viet insight.
3. Ngay 3: feature engineering va baseline models.
4. Ngay 4: train/tune cac model nang hon, so sanh metrics.
5. Ngay 5: final report, luu artifacts, chuan bi demo predict.

## 12. Tieu chi hoan thanh

Bai lam duoc xem la hoan thanh khi:

- Co giai thich ro dataset va target.
- EDA khong chi co plot, ma co insight va quyet dinh xu ly data dua tren insight.
- Data cleaning co log/rationale ro rang.
- Train/test split hop ly voi ban chat du lieu.
- Co it nhat 3 mo hinh regression duoc so sanh.
- Co metric MAE, MSE, RMSE, R2.
- Co nhan xet model tot nhat sai o dau va vi sao.
- Co de xuat trien khai hoac cai tien tiep theo.
