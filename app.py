import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Setup
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮", layout="centered")

# Custom CSS for modern visual polish
st.markdown("""
    <style>
        .stApp {
            max-width: 700px;
            margin: 0 auto;
        }
        .result-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px 20px;
            margin-top: 15px;
        }
        .highlight {
            color: #2563EB;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Cache heavy resources so they only load once on startup
@st.cache_resource
def load_assets():
    model = load_model('next_word_GRU.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # Create an inverse mapping for fast index-to-word lookup
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return model, tokenizer, index_to_word

model, tokenizer, index_to_word = load_assets()

# Function to predict the next word
def predict_next_word(model, tokenizer, index_to_word, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return None  # Handle out-of-vocabulary or empty inputs
        
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])  # Extract scalar integer
    
    return index_to_word.get(predicted_word_index, None)

# Streamlit App UI
st.title("Next Word Prediction With LSTM")
st.caption("Enter a phrase below to generate the next word in the sequence.")

input_text = st.text_input("Enter the sequence of Words", placeholder="e.g., To be or not to")

if st.button("Predict Next Word", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text first.")
    else:
        # Spinner active while inference runs
        with st.spinner("Analyzing text and predicting..."):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, index_to_word, input_text, max_sequence_len)
        
        if next_word:
            st.success(f"**Predicted Next Word:** `{next_word}`")
            st.markdown(
                f"""
                <div class="result-card">
                    <span style="color: #64748B; font-size: 0.85rem;">COMPLETE PHRASE</span><br>
                    <span style="font-size: 1.15rem;">{input_text} <span class="highlight">{next_word}</span></span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Could not predict the next word. The input words might not be in the vocabulary.")