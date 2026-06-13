# Model Report: Product Sales Prediction

## Split Strategy

- Split type: time-based split.
- Train rows: 154,048
- Test rows: 45,952
- Train dates: 2023-01-01 to 2024-09-30
- Test dates: 2024-10-01 to 2024-12-31

## Model Variants

- Model A includes `Quantity`. This demonstrates leakage because `Revenue = Quantity * Unit_Price`.
- Model B excludes `Quantity`. This is the business-realistic model for predicting revenue before sold quantity is known.
- Excluded from all models: `Order_ID`, `Customer_Name`, `Country`, `Profit`, `Revenue`.
- Ridge and Lasso are compared as regularized linear models for the non-leaking Model B setup.
- Model B includes train-only historical average quantity proxies and expected-revenue interaction features, but still excludes the current order's `Quantity`.

## Metrics

| variant | model | MAE | MSE | RMSE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model_A_With_Quantity | HistGradientBoosting | 5.90 | 214.15 | 14.63 | 1.00 |
| Model_A_With_Quantity | DecisionTree | 3.88 | 1,196.19 | 34.59 | 1.00 |
| Model_A_With_Quantity | RandomForest | 4.71 | 1,804.42 | 42.48 | 1.00 |
| Model_A_With_Quantity | Lasso | 196.16 | 94,419.46 | 307.28 | 0.83 |
| Model_A_With_Quantity | Ridge | 196.51 | 94,477.11 | 307.37 | 0.83 |
| Model_A_With_Quantity | Baseline_Mean | 509.80 | 552,643.42 | 743.40 | -0.00 |
| Model_B_No_Quantity | Lasso | 329.91 | 271,078.00 | 520.65 | 0.51 |
| Model_B_No_Quantity | Ridge | 330.14 | 271,194.64 | 520.76 | 0.51 |
| Model_B_No_Quantity | HistGradientBoosting | 329.93 | 271,391.89 | 520.95 | 0.51 |
| Model_B_No_Quantity | RandomForest | 331.05 | 273,078.00 | 522.57 | 0.51 |
| Model_B_No_Quantity | DecisionTree | 340.79 | 290,659.65 | 539.13 | 0.47 |
| Model_B_No_Quantity | Baseline_Mean | 509.80 | 552,643.42 | 743.40 | -0.00 |

## Selected Model

The selected deployment model is **Lasso** from **Model_B_No_Quantity**.

- MAE: 329.91
- RMSE: 520.65
- R2: 0.5095

Model B is selected even if Model A has stronger metrics, because Model A uses `Quantity` and therefore knows a major component of `Revenue` in advance.

## Limitations

- The dataset does not contain true campaign/ad-spend fields; time features are used as seasonal proxies.
- The dataset does not contain age/gender demographics; geography fields are used as customer-context proxies.
- Aggregate features are computed from training data only to reduce time leakage.

## Plot Files

- `/Users/quangmanh/Project/lab02/reports/plots/predicted_vs_actual_model_a_with_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/residuals_model_a_with_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/residual_by_category_model_a_with_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/residual_by_month_model_a_with_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/predicted_vs_actual_model_b_no_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/residuals_model_b_no_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/residual_by_category_model_b_no_quantity.png`
- `/Users/quangmanh/Project/lab02/reports/plots/residual_by_month_model_b_no_quantity.png`
