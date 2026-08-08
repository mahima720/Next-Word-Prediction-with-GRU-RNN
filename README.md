# 🔮 Next Word Prediction with GRU & Deep Learning

This repository features a Gated Recurrent Unit (GRU) Deep Learning model built with TensorFlow/Keras to perform real-time next-word prediction based on natural language sequences. The application is deployed via an interactive, modern Streamlit dashboard.

# 🔗 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://next-word-prediction-with-gru.streamlit.app/)

![UI](/images/image1.png)
![UI](/images/image2.png)

## 🚀 Features

* **Sequential Language Modeling:** Utilizes a GRU (Gated Recurrent Unit) network to capture long-term context and temporal dependencies in text.

* **Efficient Inference:** Uses @st.cache_resource for optimized asset loading and an inverse index-to-word map for fast runtime lookups.

* **Pre-padding Pipeline:** Implemented pad_sequences with padding='pre' to maintain consistent model input dimensions.

* **Interactive UI:** Built a modern Streamlit interface featuring dynamic text highlights and styled CSS result cards.

## 🛠️ Tech Stack

* **Language:** Python 3.x

* **Deep Learning:** TensorFlow, Keras

* **NLP & Processing:** Keras Preprocessing (Tokenizer, pad_sequences), NumPy

* **Serialization:** Pickle

* **Web Framework:** Streamlit

## Conclusion

This project demonstrates the power of Recurrent Neural Architectures (GRU) in understanding context and modeling sequential patterns in natural language. By converting text into token sequences and passing them through an Embedding and GRU layer, the model effectively learns vocabulary probabilities to output the most likely next word. The application showcases an end-to-end NLP deployment pipeline ready for real-world text completion workflows.