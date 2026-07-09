import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    font-size:48px;
    font-weight:bold;
    color:#4F8BF9;
    text-align:center;
}

.subtitle{
    font-size:18px;
    color:#BBBBBB;
    text-align:center;
    margin-bottom:25px;
}

.result-card{
    padding:25px;
    border-radius:15px;
    background-color:#1c1f26;
    border:1px solid #333;
    box-shadow:0px 0px 15px rgba(0,0,0,.4);
}

.big-text{
    font-size:30px;
    font-weight:bold;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:3em;
    font-size:18px;
    background:#4F8BF9;
    color:white;
}

.stTextArea textarea{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Dataset Word Index
# ------------------------------
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_my_model():
    MODEL_PATH = Path(__file__).parent / "models" / "rnn_model.h5"
    return load_model(MODEL_PATH)

model = load_my_model()

# ------------------------------
# Helper Functions
# ------------------------------
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=200
    )
    return padded_review


def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed, verbose=0)

    score = float(prediction[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"

    confidence = score if score > 0.5 else 1 - score

    return sentiment, score, confidence


# ------------------------------
# Header
# ------------------------------
st.markdown(
    "<div class='title'>IMDB Movie Review Sentiment Analysis</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Deep Learning based RNN Sentiment Classifier</div>",
    unsafe_allow_html=True
)

st.divider()

left, right = st.columns([2.2,1])

# ------------------------------
# Left Column
# ------------------------------
with left:

    review = st.text_area(
        "Enter Movie Review",
        height=250,
        placeholder="Type your movie review here..."
    )

    c1, c2 = st.columns(2)

    predict_btn = c1.button("Analyze Sentiment")
    clear_btn = c2.button("Clear")

    if clear_btn:
        st.rerun()

# ------------------------------
# Right Column
# ------------------------------
with right:

    st.subheader("Sample Reviews")

    st.success(
        "This movie was absolutely amazing. The acting was brilliant and the story was engaging."
    )

    st.error(
        "Worst movie I've ever watched. Completely boring and a waste of time."
    )

    st.info("""
**Model**

- Dataset: IMDB
- Architecture: Simple RNN
- Max Length: 200 words
- Output: Binary Classification
""")

# ------------------------------
# Prediction
# ------------------------------
if predict_btn:

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    with st.spinner("Analyzing Review..."):

        sentiment, score, confidence = predict_sentiment(review)

    st.divider()

    st.markdown(
        "<div class='result-card'>",
        unsafe_allow_html=True
    )

    if sentiment == "Positive":

        st.markdown(
            "<div class='big-text'>😊 Positive Review</div>",
            unsafe_allow_html=True
        )

        st.success("The model predicts this review is Positive.")

    else:

        st.markdown(
            "<div class='big-text'>😞 Negative Review</div>",
            unsafe_allow_html=True
        )

        st.error("The model predicts this review is Negative.")

    st.write("### Confidence")

    st.progress(float(confidence))

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Prediction Score",
        f"{score:.4f}"
    )

    metric2.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

st.caption(
    "Made by Aman using Streamlit, TensorFlow, and the IMDB Movie Reviews Dataset."
)