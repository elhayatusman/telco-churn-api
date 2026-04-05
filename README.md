# 📡 Telco Customer Churn Prediction API

A production-ready REST API that predicts the probability of customer churn for a telecommunications company, built with **FastAPI** and a trained **Random Forest** classifier.
---

## 🧠 Model Background

The model was trained on the [IBM Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) (7,043 customers). Key decisions:

| Decision | Detail |
|----------|--------|
| Algorithm | Random Forest (200 trees, balanced class weights) |
| Imbalance handling | SMOTE applied to training set |
| Evaluation split | Stratified 80/20 train/test |
| Primary metric | F1-score on the churner class |
| Classification threshold | 0.4376 (F1-optimised, not default 0.5) |

**Test set performance:**

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.818 |
| Recall (churners) | 0.655 |
| Precision (churners) | 0.542 |
| F1-score (churners) | 0.593 |

---

## 📦 Project Structure

```
CUSTOMER_CHURN/
├── main.py                          ← FastAPI application
├── model_artifacts/
│   ├── churn_model.pkl              ← trained Random Forest
│   ├── scaler.pkl                   ← StandardScaler (for LR fallback)
│   ├── feature_columns.pkl          ← ordered list of 25 feature names
│   └── optimal_threshold.pkl        ← F1-optimal threshold (0.4376)
├── requirements.txt
├── Procfile                         ← Render startup command
└── README.md
```

---

## 🛠️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/telco-churn-api.git
cd telco-churn-api

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload --port 8000

# 5. Open interactive docs
# → http://localhost:8000/docs
```

---

## 📮 API Endpoints

### `GET /`
Health check. Returns API status and model metadata.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "online",
  "model": "RandomForestClassifier",
  "features": 25,
  "threshold": 0.4376
}
```

---

### `POST /predict`
Predict churn for a **single customer**.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 171.00,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_label": "Critical",
  "threshold_used": 0.4376,
  "model_name": "RandomForestClassifier",
  "processing_time_ms": 18.4
}
```

**Risk labels:**

| Label | Probability Range | Recommended Action |
|-------|-------------------|-------------------|
| Low | < 0.30 | No action needed |
| Medium | 0.30 – 0.50 | Monitor; include in next outreach cycle |
| High | 0.50 – 0.70 | Proactive outreach within 2 weeks |
| Critical | ≥ 0.70 | Immediate personal outreach + retention offer |

---

### `POST /predict/batch`
Predict churn for up to **500 customers** in a single request.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      { ...customer_1_fields... },
      { ...customer_2_fields... }
    ]
  }'
```

**Response includes per-customer predictions plus aggregate stats:**
```json
{
  "predictions": [...],
  "total_records": 2,
  "predicted_churners": 1,
  "churn_rate_pct": 50.0,
  "processing_time_ms": 22.1
}
```

---

### `GET /model/info`
Returns model metadata, feature names, and risk tier definitions.

---

## 👤 Author

**Abdulrahman Hayatu Usman**  
BSc Computer Science, Ahmadu Bello University, Zaria
Machine Learning Engineer | DataCamp Associate Data Scientist

---

## 📄 License
MIT
