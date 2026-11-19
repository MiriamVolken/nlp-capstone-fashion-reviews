
import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Customer Voice",
    page_icon="👗",
    layout="wide"
)

# Load dataset and trained NLP model
@st.cache_data
def load_data():
    return pd.read_csv(
        Path(__file__).parent / "fashion_reviews_cleaned.csv"
    )

@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).parent / "fashion_sentiment_model.joblib"
    )

df = load_data()
model = load_model()

# Same text preprocessing used in Google Colab
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("👗 Customer Voice")
st.write("An NLP-powered Fashion Review Analyzer")

tab1, tab2, tab3 = st.tabs([
    "📊 Customer Dashboard",
    "🤖 Review Predictor",
    "🔍 Complaint Analysis"
])

with tab1:
    st.header("Customer Review Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reviews", f"{len(df):,}")
    col2.metric("Average Rating", f"{df['Rating'].mean():.2f}/5")
    col3.metric(
        "Recommendation Rate",
        f"{df['Recommended IND'].mean() * 100:.1f}%"
    )

    st.subheader("Ratings Distribution")
    ratings = df["Rating"].value_counts().sort_index()
    st.bar_chart(ratings)

with tab2:
    st.header("Predict Customer Recommendation")

    review = st.text_area(
        "Enter a clothing customer review:",
        placeholder="The dress fits perfectly and I love the quality!"
    )

    if st.button("Analyze Review"):
        cleaned = clean_text(review)

        if not cleaned:
            st.warning("Please enter a valid review.")
        else:
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]

            probability = probabilities[
                list(model.classes_).index(prediction)
            ]

            if prediction == 1:
                st.success("Recommended ✅")
            else:
                st.error("Not Recommended ❌")

            st.write(f"Model probability: {probability:.1%}")
            st.caption(
                "Predictions are estimates, not guaranteed outcomes."
            )

with tab3:
    st.header("Complaint Analysis")

    st.write(
        "Explore feedback from customers who did not "
        "recommend their clothing purchases."
    )

    negative_df = df[df["Recommended IND"] == 0]

    st.metric(
        "Non-recommended Reviews",
        f"{len(negative_df):,}"
    )

    st.subheader("Explore Negative Feedback")
    departments = sorted(
        negative_df["Department Name"].dropna().unique()
    )

    selected_department = st.selectbox(
        "Select a clothing department",
        ["All Departments"] + departments
    )

    if selected_department != "All Departments":
        negative_df = negative_df[
            negative_df["Department Name"] == selected_department
        ]

    st.dataframe(
        negative_df[[
            "Review Text",
            "Rating",
            "Department Name"
        ]],
        use_container_width=True,
        hide_index=True
    )

