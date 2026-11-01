# 📊 Customer Churn Prediction System

An end-to-end Machine Learning application that predicts customer churn using **Python, Pandas, Scikit-learn, and Streamlit**.

The project uses a Random Forest Classification model to analyze customer information and estimate the probability of a customer discontinuing a service. It also provides an interactive dashboard for exploring customer data and generating individual predictions.

## 🚀 Features

- **Customer Churn Prediction:** Predict whether a customer is likely to leave a subscription service.
- **Machine Learning Pipeline:** Automated preprocessing and classification using Scikit-learn.
- **Data Processing:** Dataset manipulation and analysis using Pandas and NumPy.
- **Interactive Dashboard:** User-friendly web interface built with Streamlit.
- **Model Evaluation:** Performance analysis using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Model Persistence:** Save and load trained models using Joblib.
- **Dataset Exploration:** View customer records and churn distribution.

## 🛠️ Tech Stack

| Technology | Application |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Random Forest | Classification algorithm |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |

## 🧠 Machine Learning Workflow

The project follows a standard supervised machine learning workflow:

1. **Data Generation:** Create a reproducible synthetic customer dataset.
2. **Data Preparation:** Separate input features and target variable.
3. **Preprocessing:** Apply One-Hot Encoding to categorical features using ColumnTransformer.
4. **Model Training:** Train a Random Forest Classifier.
5. **Model Evaluation:** Evaluate predictions using classification metrics.
6. **Model Serialization:** Save the trained pipeline using Joblib.
7. **Prediction:** Load the trained model and generate predictions through Streamlit.

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py                  # Streamlit application
├── train.py                # Model training and evaluation
├── generate_data.py        # Synthetic dataset generator
│
├── data/
│   └── customer_data.csv   # Customer dataset
│
├── models/
│   └── churn_model.joblib  # Generated trained model
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## 📊 Dataset Information

The project uses a synthetic dataset containing 1,200 customer records.

| Feature | Description |
|---|---|
| tenure | Duration of customer subscription in months |
| monthly_charges | Monthly subscription charges |
| support_calls | Number of customer support calls |
| contract_type | Monthly, Yearly, or Two-Year |
| internet_service | DSL, Fiber, or No |
| churn | Target variable (0 = Stayed, 1 = Churned) |

**Target Variable:** `churn`

- `0` – Customer retained.
- `1` – Customer churned.

*Note: The dataset is synthetically generated for educational purposes and does not represent actual customer records.*

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git

cd customer-churn-prediction
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Generate the dataset

```bash
uv run python generate_data.py
```

### 4. Train the machine learning model

```bash
uv run python train.py
```

The trained model will be saved inside the `models/` directory.

### 5. Launch the Streamlit application

```bash
uv run streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 📈 Model Evaluation

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Evaluation results are printed in the terminal after training.

## 🔮 Future Improvements

- Implement Logistic Regression and Decision Tree for model comparison.
- Add feature importance visualization.
- Introduce interactive confusion matrix and ROC curve.
- Implement CSV upload for batch customer predictions.
- Perform hyperparameter tuning using GridSearchCV.
- Deploy the application using Streamlit Community Cloud.

## 🎯 Learning Outcomes

Through this project, I explored:

- Supervised Machine Learning and classification.
- Data preprocessing and categorical encoding.
- Scikit-learn Pipelines and model evaluation.
- Model serialization and deployment.
- Building interactive ML applications with Streamlit.

## ⚠️ Disclaimer

This project is developed for educational and portfolio purposes. It uses synthetic data and should not be considered a production-ready customer analytics solution.

## 👨‍💻 Author

**Shrayan Biswas**

BSc Information Technology | Information Security

[GitHub](https://github.com/Shrayan-biswas)

---

⭐ If you find this project useful, consider giving it a star!