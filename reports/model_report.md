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

## Metrics

| variant | model | MAE | MSE | RMSE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model_A_With_Quantity | HistGradientBoosting | 5.12 | 190.19 | 13.79 | 1.00 |
| Model_A_With_Quantity | DecisionTree | 3.37 | 985.26 | 31.39 | 1.00 |
| Model_A_With_Quantity | RandomForest | 5.02 | 1,996.12 | 44.68 | 1.00 |
| Model_A_With_Quantity | Ridge | 199.38 | 95,778.04 | 309.48 | 0.83 |
| Model_A_With_Quantity | Baseline_Mean | 509.80 | 552,643.42 | 743.40 | -0.00 |
| Model_B_No_Quantity | Ridge | 331.20 | 272,432.28 | 521.95 | 0.51 |
| Model_B_No_Quantity | HistGradientBoosting | 330.45 | 272,520.53 | 522.03 | 0.51 |
| Model_B_No_Quantity | RandomForest | 331.06 | 272,940.58 | 522.44 | 0.51 |
| Model_B_No_Quantity | DecisionTree | 340.23 | 287,986.69 | 536.64 | 0.48 |
| Model_B_No_Quantity | Baseline_Mean | 509.80 | 552,643.42 | 743.40 | -0.00 |

## Selected Model

The selected deployment model is **Ridge** from **Model_B_No_Quantity**.

- MAE: 331.20
- RMSE: 521.95
- R2: 0.5070

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
