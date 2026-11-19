
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
    
with tab2:
    st.header("Test a Customer Review in Real Time")

    # Example reviews
    examples = {
        "positive": "I absolutely love this dress! The fabric is beautiful and the fit is perfect.",
        "negative": "Wanted to love this top, but it runs way too small and the material feels cheap.",
        "mixed": "The color is beautiful, but the fabric feels uncomfortable and the fit is disappointing.",
        "edge": "😍😍😍"
    }

    # Store text between Streamlit reruns
    if "review_input" not in st.session_state:
        st.session_state.review_input = examples["negative"]

    # Four preset buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("✅ Positive Review"):
            st.session_state.review_input = examples["positive"]

    with col2:
        if st.button("❌ Sizing & Fabric Issue"):
            st.session_state.review_input = examples["negative"]

    with col3:
        if st.button("⚠️ Mixed Sentiment"):
            st.session_state.review_input = examples["mixed"]

    with col4:
        if st.button("🧪 Edge Case (Emojis/Blank)"):
            st.session_state.review_input = examples["edge"]

    review = st.text_area(
        "Enter or paste a customer review:",
        key="review_input",
        height=110
    )

    # Adjustable threshold
    threshold = st.slider(
        "Dissatisfaction Alert Threshold "
        "(flag if risk is at least this value):",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        format="%d%%"
    )

    if st.button("Analyze Review", type="primary"):

        cleaned = clean_text(review)

        if not cleaned:
            st.warning(
                "Please enter a review containing valid text. "
                "Emojis or blank input alone cannot be analyzed."
            )

        else:
            # Get probabilities from trained model
            probabilities = model.predict_proba([cleaned])[0]
            classes = list(model.classes_)

            p_recommend = float(
                probabilities[classes.index(1)]
            )

            p_dissatisfaction = float(
                probabilities[classes.index(0)]
            )

            # Apply adjustable business threshold
            flagged = p_dissatisfaction >= threshold / 100

            st.divider()

            result_col, prob_col, risk_col = st.columns(3)

            with result_col:
                if flagged:
                    st.error(
                        "❌ NOT RECOMMENDED "
                        "(Dissatisfaction Flagged)"
                    )
                else:
                    st.success(
                        "✅ RECOMMENDED "
                        "(No Dissatisfaction Alert)"
                    )

            with prob_col:
                st.metric(
                    "Recommendation Probability",
                    f"{p_recommend:.1%}"
                )

            with risk_col:
                st.metric(
                    "Dissatisfaction Risk",
                    f"{p_dissatisfaction:.1%}"
                )

            # Explain prediction using TF-IDF contributions
            st.subheader("What Influenced the Prediction?")

            # Works with our TF-IDF + Logistic Regression model
            vectorizer = model.named_steps["tfidf"]
            classifier = model.named_steps["classifier"]

            vector = vectorizer.transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()

            # Contributions toward recommendation (class 1)
            weights = classifier.coef_[0]
            contributions = vector.multiply(weights)

            feature_indices = vector.indices
            feature_contributions = contributions.toarray()[0]

            positive_signals = []
            negative_signals = []

            for idx in feature_indices:
                word = feature_names[idx]
                score = float(feature_contributions[idx])

                if score > 0:
                    positive_signals.append((word, score))
                elif score < 0:
                    negative_signals.append((word, score))

            positive_signals.sort(
                key=lambda x: x[1], reverse=True
            )

            negative_signals.sort(
                key=lambda x: x[1]
            )

            left, right = st.columns(2)

            with left:
                st.markdown("### 🟢 Top Positive Phrase Signals")

                if positive_signals:
                    for word, score in positive_signals[:5]:
                        st.write(f"• `{word}` (+{score:.2f})")
                else:
                    st.write("No positive signals identified.")

            with right:
                st.markdown("### 🔴 Top Negative / Complaint Signals")

                if negative_signals:
                    for word, score in negative_signals[:5]:
                        st.write(f"• `{word}` ({score:.2f})")
                else:
                    st.write("No negative signals identified.")

            st.caption(
                "Phrase scores show contributions to the "
                "Logistic Regression decision. They are not "
                "separate probabilities or proof of causality. "
                "The threshold changes the alert decision, "
                "not the underlying model probabilities."
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

