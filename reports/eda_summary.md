# EDA Summary: Product Sales Dataset

## Data Quality

- Rows: 200,000
- Columns: 14
- Missing values: 0
- Duplicate rows: 0
- Date range: 2023-01-01 to 2024-12-31
- Date parse failures: 0
- Revenue formula mismatches: 0
- Columns stripped for whitespace: [' Unit_Price ', ' Revenue ', ' Profit ']

## Target: Revenue

- Mean revenue: 712.04
- Median revenue: 464.88
- Std revenue: 742.47
- Min revenue: 17.03
- Max revenue: 9,014.25
- Skewness: 2.5357

Revenue is right-skewed, so MAE and RMSE should both be reported. RMSE will penalize high-revenue errors more strongly.

## Time Insights

- Best revenue month: 2024-11-30 with 13,899,671.91
- Best weekday by revenue: Friday with 23,257,334.62
- The dataset has daily records across 2023 and 2024, so a time-based train/test split is more realistic than a random split.

## Product Insights

### Top Categories by Revenue

| Category | orders | quantity | revenue | avg_revenue | median_revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Electronics | 51230 | 97115 | 57,485,698.06 | 1,122.11 | 837.06 |
| Home & Furniture | 50564 | 90403 | 47,674,426.96 | 942.85 | 709.31 |
| Clothing & Apparel | 62298 | 115523 | 27,134,365.30 | 435.56 | 331.27 |
| Accessories | 35908 | 67759 | 10,113,254.61 | 281.64 | 213.30 |

### Top Sub-Categories by Revenue

| Sub_Category | orders | quantity | revenue | avg_revenue | median_revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bedding | 10171 | 18575 | 13,042,783.30 | 1,282.35 | 975.00 |
| Laptops | 8656 | 17630 | 12,358,319.81 | 1,427.72 | 1,037.54 |
| Smartphones | 8442 | 16985 | 10,904,335.31 | 1,291.68 | 951.83 |
| Furniture | 10096 | 19002 | 9,697,778.92 | 960.56 | 722.46 |
| Wearables | 8563 | 16755 | 9,216,507.48 | 1,076.32 | 810.12 |
| Kitchenware | 10091 | 19541 | 9,117,341.17 | 903.51 | 676.86 |
| Home Appliances | 8360 | 16080 | 8,638,544.88 | 1,033.32 | 789.34 |
| Tablets | 8557 | 15187 | 8,373,830.76 | 978.59 | 760.46 |
| Home Decor | 10165 | 17024 | 8,070,216.82 | 793.92 | 625.06 |
| TVs & Audio | 8652 | 14478 | 7,994,159.82 | 923.97 | 726.58 |

### Top Products by Revenue

| Product_Name | orders | quantity | revenue | avg_revenue | median_revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tempur-Pedic Mattress | 5523 | 10333 | 9,061,755.86 | 1,640.73 | 1,221.98 |
| Instant Pot | 8657 | 18200 | 8,903,475.26 | 1,028.47 | 768.11 |
| MacBook Air | 3926 | 8763 | 7,362,516.81 | 1,875.32 | 1,464.60 |
| Apple Watch | 5931 | 12368 | 6,834,472.35 | 1,152.33 | 860.82 |
| Apple iPhone 14 | 3636 | 7642 | 5,740,819.18 | 1,578.88 | 1,167.65 |
| iPad Pro | 5421 | 10075 | 5,574,458.89 | 1,028.31 | 781.94 |
| KitchenAid Mixer | 5376 | 9831 | 4,989,740.69 | 928.15 | 703.52 |
| Storage Rack | 5707 | 9394 | 4,463,941.27 | 782.19 | 614.98 |
| Brooklinen Sheets | 4648 | 8242 | 3,981,027.44 | 856.50 | 660.93 |
| Samsung Galaxy S23 | 3365 | 6974 | 3,851,240.02 | 1,144.50 | 854.23 |

## Geography Insights

### Top Regions by Revenue

| Region | orders | quantity | revenue | avg_revenue | median_revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| East | 57034 | 105965 | 44,980,048.22 | 788.65 | 547.19 |
| West | 55428 | 102931 | 36,242,841.73 | 653.87 | 416.57 |
| Centre | 49603 | 91464 | 36,081,894.34 | 727.41 | 482.19 |
| South | 37935 | 70440 | 25,102,960.64 | 661.74 | 428.53 |

### Top States by Revenue

| State | orders | quantity | revenue | avg_revenue | median_revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| California | 10272 | 19061 | 6,766,728.65 | 658.75 | 413.42 |
| Arizona | 10262 | 19032 | 6,668,187.20 | 649.79 | 411.23 |
| New Jersey | 5780 | 10712 | 4,552,067.75 | 787.55 | 548.01 |
| Vermont | 5677 | 10582 | 4,550,459.19 | 801.56 | 552.64 |
| New York | 5766 | 10730 | 4,541,307.27 | 787.60 | 551.28 |
| Connecticut | 5726 | 10666 | 4,515,748.44 | 788.64 | 545.68 |
| Delaware | 5673 | 10565 | 4,515,199.69 | 795.91 | 554.96 |
| Pennsylvania | 5765 | 10756 | 4,495,637.28 | 779.82 | 552.50 |
| New Hampshire | 5682 | 10532 | 4,489,012.87 | 790.04 | 539.04 |
| Massachusetts | 5627 | 10484 | 4,471,454.84 | 794.64 | 549.22 |

### Top Cities by Revenue

| City | orders | quantity | revenue | avg_revenue | median_revenue |
| --- | ---: | ---: | ---: | ---: | ---: |
| Burlington | 2920 | 5487 | 2,367,578.63 | 810.81 | 564.22 |
| Manchester | 2912 | 5349 | 2,316,261.28 | 795.42 | 548.09 |
| Rochester | 2905 | 5416 | 2,312,165.68 | 795.93 | 556.44 |
| Wilmington | 2874 | 5406 | 2,311,588.97 | 804.31 | 560.91 |
| Providence | 2901 | 5355 | 2,298,465.51 | 792.30 | 537.81 |
| Jersey City | 2905 | 5343 | 2,290,243.36 | 788.38 | 556.06 |
| Boston | 2878 | 5353 | 2,274,264.86 | 790.22 | 531.81 |
| Philadelphia | 2833 | 5279 | 2,262,453.14 | 798.61 | 571.82 |
| Newark | 2875 | 5369 | 2,261,824.39 | 786.72 | 541.26 |
| New Haven | 2882 | 5399 | 2,261,371.05 | 784.65 | 540.70 |

## Modeling Notes

- `Revenue` is the target.
- `Profit` is excluded because it is a post-sale outcome.
- `Order_ID` is excluded because it is only a row identifier.
- `Customer_Name` is excluded from the main model because it has very high cardinality and privacy/noise risk.
- `Country` is excluded because it has one value only.
- Model A includes `Quantity` and is useful to demonstrate leakage because `Revenue = Quantity * Unit_Price`.
- Model B excludes `Quantity` and is the business-realistic model for predicting revenue before the sold quantity is known.
- Regularized linear models such as Ridge and Lasso are included to reduce variance and keep Model B usable without the leaked quantity signal.
- Model B uses train-only historical average quantity features and `Unit_Price * historical_avg_quantity` proxies, never the current row's `Quantity`.

## Plot Files

- `/Users/quangmanh/Project/lab02/reports/plots/revenue_distribution.png`
- `/Users/quangmanh/Project/lab02/reports/plots/revenue_boxplot.png`
- `/Users/quangmanh/Project/lab02/reports/plots/monthly_revenue.png`
- `/Users/quangmanh/Project/lab02/reports/plots/revenue_by_category.png`
- `/Users/quangmanh/Project/lab02/reports/plots/revenue_by_region.png`
- `/Users/quangmanh/Project/lab02/reports/plots/top_products_by_revenue.png`
- `/Users/quangmanh/Project/lab02/reports/plots/top_states_by_revenue.png`
- `/Users/quangmanh/Project/lab02/reports/plots/unit_price_vs_revenue.png`
- `/Users/quangmanh/Project/lab02/reports/plots/quantity_vs_revenue.png`
