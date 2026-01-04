import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download once
# nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    model = pickle.load(open("Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open("Models/scaler.pkl", "rb"))
    cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))
    return model, scaler, cv

model, scaler, cv = load_models()


# ---------- TEXT CLEANING ----------
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    return " ".join(text)


# ---------- UI ----------
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("💬 Sentiment Analysis Web App")
st.write("Analyze sentiment of text or CSV file using ML model")

# ---------- SINGLE TEXT ----------
st.subheader("🔹 Single Text Prediction")
user_text = st.text_area("Enter text")

if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter text")
    else:
        corpus = [clean_text(user_text)]
        X = cv.transform(corpus).toarray()
        X = scaler.transform(X)

        pred = model.predict(X)[0]
        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")

# ---------- BULK CSV ----------
st.subheader("🔹 Bulk Prediction (CSV)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "Sentence" not in df.columns:
        st.error("CSV must contain a 'Sentence' column")
    else:
        corpus = df["Sentence"].apply(clean_text).tolist()
        X = cv.transform(corpus).toarray()
        X = scaler.transform(X)

        preds = model.predict(X)
        df["Predicted Sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]

        st.dataframe(df.head())

        # ---------- PIE CHART ----------
        st.subheader("📊 Sentiment Distribution")
        fig, ax = plt.subplots()
        df["Predicted Sentiment"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            colors=["green", "red"],
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

        # ---------- DOWNLOAD ----------
        st.download_button(
            "⬇ Download Predictions",
            df.to_csv(index=False),
            file_name="sentiment_predictions.csv",
            mime="text/csv"
        )
