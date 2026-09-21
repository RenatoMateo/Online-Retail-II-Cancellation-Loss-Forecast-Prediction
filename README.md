# Online Retail II — Advanced Analytics (Part 2)

This project extends the cancellation-loss investigation (Part 1 in the project's footnote) into predictive analysis. It applies K-Means segmentation across RFMR dimensions, uses Prophet for gross sales and cluster-level cancellation forecasting, and compares XGBoost with Random Forest for cancellation prediction. These methods translate behavioral clusters into targeted account priorities, inform monthly operational planning, and provide a forward-looking risk ranking for the UK customer base.

---

## Stage

| Stage | Tool | What It Does |
|---|---|---|
| 1. Source | Part 1 cleaned dataset | Shared cleaning pipeline from DataScience_Merged.ipynb |
| 2. Segment | Scikit-learn / K-Means | Cluster UK and Germany customers by RFMR behavioral dimensions |
| 3. Forecast | Prophet / pmdarima | Monthly gross sales and cluster-level cancellation forecasting |
| 4. Predict | XGBoost / Random Forest | Customer-level cancellation loss prediction using Year 1 → Year 2 feature engineering |
| 5. Validate | SciPy / Silhouette Analysis | Elbow method and silhouette scoring for optimal K selection |
| 6. Visualize | Matplotlib | Cluster heatmaps, forecast charts, feature importance plots, risk ranking |

---

## Dataset

Same source as Part 1 — Online Retail II (UCI Machine Learning Repository).

| Field | Detail |
|---|---|
| Source | Online Retail II (UCI Machine Learning Repository) |
| URL | archive.ics.uci.edu/dataset/502/online+retail+ii |
| Volume | 1,048,576 transaction rows |
| Period | December 2009 to December 2011 |
| Markets analyzed | United Kingdom (85.2% of losses) and Germany |

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Forecasting | Prophet, pmdarima (auto-ARIMA) |
| Statistical Testing | SciPy |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

## Analytical Framework

| Lens | Method | Question |
|---|---|---|
| Segmentation | K-Means (K=3) on scaled RFMR features | Which behavioral clusters define cancellation risk, and how should each be managed? |
| Forecasting | Prophet (primary) + ARIMA (comparison) | What cancellation losses and gross sales should we expect in the next six months, and when are the peaks? |
| Prediction | XGBoost vs Random Forest, Year 1 → Year 2 | Which individual customers are most likely to cancel in the next period, and by how much? |

---

## Key Design Decisions

**RobustScaler was chosen over StandardScaler for K-Means clustering.** Using the median and IQR instead of the mean and standard deviation to prevent extreme outliers, such as Customer 12346's £77k bulk cancellation, from distorting the feature space and skewing cluster assignments.

**Customer 12346 was isolated before clustering.** This customer's single 74,215-unit bulk cancellation in January 2011 was a unique B2B event, not a recurring behavioral pattern. Including this account would have created a segment of one and distorted the analysis of the remaining 2,291 accounts. The account was labeled "The Event" and included in the summary after clustering.

**K=3 was selected over K=2 for both markets.** Although silhouette scoring peaked at K=2 for the UK and Germany, K=3 produced more distinct behavioral segments. The third cluster in each market identified a structurally different group: Question Mark in the UK and One-and-Gone in Germany. Each carries separate business implications.

**Prophet was chosen over ARIMA for forecasting.** While ARIMA achieved better test-set accuracy (MAE £62,590 vs £368,160), its six-month forecast was flat at £695k per month, offering no actionable insights. Prophet's monthly variation, ranging from £319k in May to £835k in March, enables effective resource allocation decisions.

**A log transformation was applied to the machine learning target.** The cancellation loss distribution is highly right-skewed, with most customers canceling small amounts and a few canceling very large amounts. Applying log(y + 1) before training and exponentiating predictions improved model stability and reduced the impact of extreme values on the loss function.

**Year 1 features were used to predict Year 2 targets.** The machine learning model was trained on behavioral features from Year 1 (Dec 2009 to Nov 2010) to predict cancellation loss in Year 2 (Dec 2010 to Nov 2011). This out-of-sample approach avoids data leakage and reflects real-world deployment.

---

## Key Findings

**Segmentation shows that cancellation risk is behavioral, not random.** Three distinct customer profiles emerged in both markets. In the UK, 24 Stars accounts generated £2.01M in gross sales with a 7.4% return rate but represent the highest relationship risk. Eight Question Mark accounts had a 44.9% return rate before disengaging entirely. In Germany, five Recents Risky accounts are actively buying and canceling at the same time, making them the most urgent intervention target.

**Forecasting identifies June as the critical month.** The projected cancellation rate reaches 10% in June 2012, nearly double the historical average of 16.9% at the invoice level. Question Mark accounts are responsible for 57% of June's expected losses. May and January are secondary peaks. February and March provide a low-risk window for operational drawdown and preparation.

**Prediction results show that cluster behavior and spend drive cancellation risk.** XGBoost ranked cluster membership as the top predictive feature (46% importance), with gross purchases second (15%). This confirms that behavioral characteristics, not just spending levels, are the strongest indicators of future cancellation. Four Stars accounts (Customers 16013, 15311, 17450, 12748) have the highest combined risk scores entering 2012.

**Anonymous transactions are a structural blind spot.** £383k in cancellations across the two-year period carry no Customer ID. We can't cluster them, forecast them at the account level, or manage them proactively. In January 2012 alone, anonymous transactions are expected to account for 59% of projected losses.

---

## Key Recommendations

### Segmentation

**Stars (UK) and Recents Risky (Germany):** assign one-on-one account management. These are the highest-value, most active accounts in each market. Their cancellations reflect friction in ongoing relationships rather than disengagement. Assign a single point of contact to each account and provide real-time inventory visibility before orders are placed to reduce B2B cancellation triggers.

**Normal Bypass (UK) and Regulars (Germany):** apply a single blanket order policy. These customers canceled at some point but continued purchasing, so individual management is not necessary. Require order confirmation within 48 hours for orders above £500 or 50 units, and mandate confirmation at checkout for customers with fewer than three prior invoices.

**Question Mark (UK):** conduct individual account reviews. Eight accounts with £441k in gross sales and a 44.9% return rate stopped all activity at the same time. Review order history and communication records for each account, determine whether the cancellation caused the exit or followed it, and decide on recovery actions individually.

### Forecasting

**Engage accounts pre-emptively in May for June.** June has the highest projected cancellation rate (10%) and volume (£57k). Reach out to Question Mark and Stars accounts one month in advance through direct communication, commercial negotiation, or delivery guarantees to prevent cancellations.

**Maintain returns-processing capacity in January and May, despite different risk triggers.** January's risk is driven by Anonymous transactions (59% of projected losses), which is an operational issue rather than an account issue. May's risk is spread across clusters, with no single group dominating. Both months require increased staffing rather than account management.

**Use February and March as preparation periods.** February has the lowest projected volume (£14k, 3.7% rate), while March has the lowest cancellation rate (3.4%) despite the highest sales (£835k). Reduce returns resources in February and shift capacity to fulfillment in March.

**Mandate Customer ID capture at the point of sale.** Anonymous cancellations cannot be managed at the account level. This requires a process change, not a data change. Require login or staff-assisted identification for every transaction.

### Prediction

**Monitor cluster migration as a leading indicator of future losses.** Cluster membership is the strongest predictor of cancellation risk. Any customer shifting toward Question Mark or Stars behavior, indicated by rising frequency and increasing cancellation loss, should be flagged before losses occur. Prioritize outreach to Customers 16013, 15311, 17450, and 12748 before January 2012.

---

## Future Work

| Priority | Description |
|---|---|
| Investigate churn | 33% of Year 1 customers did not appear in Year 2. Whether cancellation behavior drove that exit is unknown — and the answer changes what intervention looks like. |
| Identify Anonymous segment | £383k in cancellations with no Customer ID. Resolving this requires a point-of-sale process fix, not a data enrichment exercise. |
| Expand the dataset | Two years and 1,193 modeled customers produced an R² of 0.21. Five or more years of data, plus product category and relationship history features, would materially improve both segmentation stability and predictive accuracy. |

---

## Repository Structure

```
online-retail-advanced-analytics/
├── data/
│   └── raw/                              # Online Retail II source file (not redistributed — see References)
│
├── notebooks/
│   └── Analytics_Forecast_Online_Retail.ipynb   # Full Part 2 analysis
│
├── charts/
│   └── *.png                             # All exported chart images
│
└── README.md
```

---

## How to Run

**Prerequisites**

```
pip install pandas numpy scipy matplotlib jupyter scikit-learn xgboost prophet pmdarima
```

1. Complete Part 1 first — this notebook uses the same cleaned dataset and shared cleaning pipeline.
2. Download `online_retail_II.csv` from the UCI Machine Learning Repository and place it in `data/raw/`.
3. Open `Analytics_Forecast_Online_Retail.ipynb` and run all cells in order.

---

## References

Chen, D. (2019). Online Retail II [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/502/online+retail+ii

---

## Author

Renato Silva — Data Reporting Analyst

[LinkedIn](https://www.linkedin.com/in/renato-silva-portilla/) | [GitHub](https://github.com/RenatoMateo) | [Retail Part 1](https://github.com/RenatoMateo/online-retail-cancellation-analysis-EDA)
