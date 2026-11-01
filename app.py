from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
DATA_PATH = BASE_DIR / "data" / "customer_data.csv"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("📊 Customer Churn Prediction")
st.caption(
    "A demo machine-learning application that estimates whether a customer "
    "may leave a subscription service."
)

if model is None:
    st.warning("The trained model hasn't been created yet.")
    st.code("uv run python train.py")
    st.stop()

tab_predict, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "📁 Explore dataset", "ℹ️ About"]
)

with tab_predict:
    st.subheader("Enter customer details")
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 18)
        monthly_charges = st.number_input(
            "Monthly charges", min_value=20.0, max_value=120.0,
            value=65.0, step=1.0
        )
        support_calls = st.slider("Support calls in recent period", 0, 9, 2)

    with col2:
        contract_type = st.selectbox(
            "Contract type", ["Monthly", "Yearly", "Two-Year"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber", "No"]
        )

    if st.button("Predict churn risk", type="primary", use_container_width=True):
        customer = pd.DataFrame([{
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "contract_type": contract_type,
            "internet_service": internet_service,
        }])

        churn_probability = float(model.predict_proba(customer)[0][1])
        prediction = int(model.predict(customer)[0])

        st.divider()
        st.subheader("Prediction")
        metric_col, result_col = st.columns([1, 2])

        with metric_col:
            st.metric("Estimated churn probability", f"{churn_probability:.1%}")
            st.progress(churn_probability)

        with result_col:
            if prediction == 1:
                st.error(
                    "Higher churn risk: the model predicts this customer may leave."
                )
                st.write(
                    "Consider reviewing service issues, support experience, "
                    "and contract options."
                )
            else:
                st.success(
                    "Lower churn risk: the model predicts this customer is likely to stay."
                )
                st.write(
                    "Continue monitoring customer satisfaction and service usage."
                )

        st.caption(
            "This is a demonstration model trained on synthetic data. "
            "Its probability is not a guarantee or a real-world business assessment."
        )

with tab_data:
    st.subheader("Training dataset")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        a, b, c = st.columns(3)
        a.metric("Customer records", f"{len(data):,}")
        b.metric("Churned customers", f"{int(data['churn'].sum()):,}")
        c.metric("Churn rate", f"{data['churn'].mean():.1%}")

        st.dataframe(data.head(100), use_container_width=True)
        st.subheader("Churn distribution")
        counts = data["churn"].value_counts().rename(
            index={0: "Stayed", 1: "Churned"}
        )
        st.bar_chart(counts)
    else:
        st.info("Dataset file not found.")

with tab_about:
    st.subheader("Project overview")
    st.markdown(
        """
        **Goal:** demonstrate a complete, beginner-friendly classification workflow.

        - **Pandas:** load and inspect tabular data.
        - **Scikit-learn:** preprocess categorical columns and train a
          Random Forest classifier.
        - **Pipeline:** keep preprocessing and prediction together.
        - **Streamlit:** provide an interactive prediction interface.
        - **Joblib:** save and reload the trained model.

        The included dataset is synthetic and generated for learning. Do not
        use its predictions to make real customer decisions.
        """
    )
