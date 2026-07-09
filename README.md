# 🎬 IMDB Movie Review Sentiment Analysis

A modern **Deep Learning-based Sentiment Analysis** web application that classifies IMDB movie reviews as **Positive 😊** or **Negative 😞** using a **Simple Recurrent Neural Network (Simple RNN)** built with **TensorFlow/Keras** and deployed using **Streamlit**.

## 🌐 Live Demo

🚀 **Try the application here:**

**https://simple-rnn-sentiment-analysis20.streamlit.app/**

---

## 📖 Overview

Sentiment Analysis is a Natural Language Processing (NLP) task that determines whether a piece of text expresses a positive or negative opinion.

This project uses the **IMDB Movie Reviews Dataset** and a **Simple RNN** model trained to classify movie reviews based on their sentiment.

The application provides a clean and interactive web interface where users can enter any movie review and instantly receive:

- ✅ Predicted Sentiment
- 📊 Prediction Score
- 📈 Confidence Level

---

## ✨ Features

- 🎬 Movie Review Sentiment Classification
- 🤖 Deep Learning using Simple RNN
- ⚡ Real-time Prediction
- 📊 Confidence Score Visualization
- 🎨 Modern Streamlit UI
- 📱 Responsive Layout
- ☁️ Cloud Deployed using Streamlit Community Cloud

---

## 🛠️ Tech Stack

| Technology         | Purpose                        |
| ------------------ | ------------------------------ |
| Python             | Programming Language           |
| TensorFlow / Keras | Deep Learning Framework        |
| Simple RNN         | Sentiment Classification Model |
| Streamlit          | Web Application                |
| NumPy              | Numerical Computation          |
| IMDB Dataset       | Training Dataset               |

---

## 📂 Project Structure

```
Simple_RNN_Sentiment_Analysis/
│
├── models/
│   └── rnn_model.h5
│
├── app.py
├── requirements.txt
├── dl-7-imdb-rnn-sentiment-analysis.ipynb
└── README.md
```

---

## 🧠 Model Architecture

The model consists of:

```
Input Layer
      │
Embedding Layer
      │
Simple RNN Layer
      │
Dense Layer (Sigmoid)
      │
Positive / Negative
```

---

## 📊 Dataset

The project uses the **IMDB Movie Reviews Dataset** provided by TensorFlow/Keras.

- 50,000 Movie Reviews
- Binary Classification
- Positive Reviews
- Negative Reviews

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/AmanPathan24/Simple_RNN_Sentiment_Analysis.git
```

Move into the project directory

```bash
cd Simple_RNN_Sentiment_Analysis
```

Install the dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 💻 Usage

1. Open the application.
2. Enter a movie review.
3. Click **Analyze Sentiment**.
4. View the predicted sentiment along with confidence score.

---

## 📸 Application Preview

### Home Page

- Modern User Interface
- Movie Review Input
- Sample Reviews
- Model Information

### Prediction

The application displays:

- 😊 Positive / 😞 Negative Prediction
- Prediction Probability
- Confidence Percentage
- Progress Indicator

---

## 📈 Example

### Input

```
This movie was absolutely amazing. The acting was brilliant and the storyline kept me engaged throughout.
```

### Output

```
Sentiment:
😊 Positive

Confidence:
97.81%
```

---

## 🔮 Future Improvements

- 🔥 LSTM Model
- 🚀 GRU Model
- 🤖 Transformer-based Sentiment Analysis
- 🌍 Multi-language Support
- 📊 Sentiment Probability Graph
- 🎤 Voice Input
- 📄 Review History

---

## 📚 Learning Objectives

This project demonstrates:

- Natural Language Processing (NLP)
- Text Preprocessing
- Word Embeddings
- Sequence Padding
- Recurrent Neural Networks (RNN)
- Binary Classification
- TensorFlow Model Deployment
- Streamlit Application Development
