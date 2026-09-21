# Online Retail II — Advanced Analytics (Part 2)

Extension of the cancellation-loss investigation into predictive territory: K-Means customer segmentation across RFMR dimensions, Prophet-based gross sales and cluster-level cancellation forecasting, and XGBoost vs Random Forest cancellation prediction — translating behavioral clusters into targeted account priorities, monthly operational planning, and a forward-looking risk ranking of the UK customer base.

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

**RobustScaler over StandardScaler.** K-Means clustering used RobustScaler (median and IQR) instead of StandardScaler (mean and std) to prevent extreme outliers — particularly Customer 12346's £77k bulk cancellation — from distorting the feature space and pulling clusters toward the outlier.

**Customer 12346 isolated before clustering.** This customer's single 74,215-unit bulk cancellation in January 2011 represents a one-off B2B event, not a behavioral pattern. Including it in clustering would create a segment of one and distort the remaining 2,291 accounts. It was isolated, labeled "The Event," and added back to the summary after clustering.

**K=3 over K=2 for both markets.** Silhouette scoring peaked at K=2 for both UK and Germany, but K=3 was selected because it produced meaningfully distinct behavioral segments rather than a binary split. The third cluster in each market identified a structurally different group — Question Mark in the UK, One-and-Gone in Germany — that carries separate business implications.

**Prophet over ARIMA for forecasting.** ARIMA outperformed Prophet on test-set accuracy (MAE £62,590 vs £368,160), but its six-month forecast was completely flat at £695k every month. A flat forecast provides no basis for planning. Prophet's monthly variation — ranging from £319k in May to £835k in March — is what makes resource allocation decisions possible.

**Log transform on ML target.** The cancellation loss distribution is highly right-skewed, with most customers canceling small amounts and a few canceling very large ones. Applying log(y + 1) before training and exponentiating predictions back improved model stability and reduced the influence of extreme values on the loss function.

**Year 1 features → Year 2 target.** The ML model is trained on behavioral features from Year 1 (Dec 2009 – Nov 2010) to predict Year 2 (Dec 2010 – Nov 2011) cancellation loss — a genuine out-of-sample setup that avoids data leakage and reflects how the model would be deployed in practice.

---

## Key Findings

**Segmentation — Cancellation risk is behavioral, not random.** Three distinct customer profiles emerged in both markets. In the UK, 24 Stars accounts generate £2.01M in gross sales with only a 7.4% return rate but represent the highest relationship risk. Eight Question Mark accounts carried a 44.9% return rate before disengaging entirely. In Germany, five Recents Risky accounts are actively buying and canceling simultaneously — the most urgent intervention target.

**Forecasting — June is the critical month.** The projected cancellation rate reaches 10% in June 2012, nearly double the historical two-year average. Question Mark accounts are responsible for 57% of June's expected losses. May and January are secondary peaks. February and March offer a low-risk window for operational drawdown and preparation.

**Prediction — Cluster behavior and spend drive cancellation risk.** XGBoost ranked cluster membership as the top predictive feature (46% importance), with gross purchases second (15%). Together, they confirm that who a customer is behaviorally — not just how much they spend — is the strongest signal for future cancellation. Four Stars accounts (Customers 16013, 15311, 17450, 12748) carry the highest combined risk scores entering 2012.

**Anonymous transactions are a structural blind spot.** £383k in cancellations across the two-year period carry no Customer ID. These cannot be clustered, forecasted at the account level, or managed proactively. In January 2012 alone, anonymous transactions are expected to account for 59% of projected losses.

---

## Key Recommendations

### Segmentation

**Stars (UK) and Recents Risky (Germany) — assign one-on-one account management.** These are the highest-value, most active accounts in each market. Their cancellations reflect friction in an ongoing relationship, not disengagement. Dedicate a single point of contact to each account and provide real-time inventory visibility before orders are placed to remove a common B2B cancellation trigger.

**Normal Bypass (UK) and Regulars (Germany) — apply a single blanket order policy.** These customers canceled at some point but continued purchasing. Individual management is not warranted. Require order confirmation within 48 hours for orders above £500 or 50+ units, and mandatory confirmation at checkout for customers with fewer than three prior invoices.

**Question Mark (UK) — conduct individual account review.** Eight accounts with £441k in gross sales and a 44.9% return rate stopped all activity simultaneously. Pull order history and communication records for each, assess whether the cancellation caused the exit or followed it, and make a recovery decision per account.

### Forecasting

**Pre-emptive account engagement in May for June.** June carries the highest projected cancellation rate (10%) and volume (£57k). Engage Question Mark and Stars accounts one month in advance through direct outreach, commercial negotiation, or delivery guarantees before orders convert to cancellations.

**Maintain returns-processing capacity in January and May despite different triggers.** January's risk is driven by Anonymous transactions (59% of projected losses) — an operational problem, not an account one. May's risk is distributional, with no single cluster dominating. Both require staffing, not account management.

**Use February and March as the preparation window.** February has the lowest projected volume (£14k, 3.7% rate). March has the lowest rate (3.4%) despite the highest sales (£835k). Reduce returns resources in February and redirect to fulfillment capacity in March.

**Mandate Customer ID capture at the point of sale.** Anonymous cancellations cannot be managed at the account level. This is a process fix, not a data fix — require login or staff-assisted identification for every transaction.

### Prediction

**Monitor cluster migration as the leading indicator of future losses.** Cluster membership is the strongest predictor of cancellation risk. Any customer shifting toward Question Mark or Stars behavior — rising frequency paired with rising cancellation loss — should be flagged before losses materialize, not after. Prioritize outreach to Customers 16013, 15311, 17450, and 12748 before January 2012.

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

[LinkedIn](#) | [GitHub](#)
