import streamlit as st
import pickle
import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spam Detection", layout="wide")

#AUTO-TRAINING LOGIC
#Check if files exist. If not, trigger the training script.
required_files = ['model.pkl', 'vectorizer.pkl', 'scaler.pkl', 'confusion_matrix.png', 'feature_importance.png']

if not all(os.path.exists(f) for f in required_files):
    with st.spinner("First-time setup: Training Hybrid SVM Model & Generating Graphs... (This takes ~15 seconds)"):
        try:
            from spam_classifier import train_and_evaluate
            train_and_evaluate()
            st.success("Training Complete! Loading App...")
            time.sleep(1) # Small pause for UX
            st.rerun() # Refresh the app to load new files
        except Exception as e:
            st.error(f"Fatal Error during training: {e}")
            st.stop()

#LOAD RESOURCES
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_resources():
    with open('model.pkl', 'rb') as f: m = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f: v = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: s = pickle.load(f)
    return m, v, s

model, vectorizer, scaler = load_resources()

#HELPER FUNCTIONS
def clean_text(text):
    if not isinstance(text, str): return ""
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def get_meta_features(text):
    if not isinstance(text, str): return [0, 0, 0, 0, 0]
    return [
        len(text),
        1 if re.search(r'http[s]?://|www\.', text) else 0,
        sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        text.count('!'),
        1 if re.search(r'[$£€]', text) else 0
    ]

#UI LAYOUT
st.title("Spam Detector")
#st.markdown("### Ardentix Assignment Submission")

# Tabs
tab1, tab2 = st.tabs(["Live Detector", "Model Performance"])

# TAB 1: Detector
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Test Suspicious Messages")
        user_input = st.text_area("Message Content:", height=150, placeholder="Paste SMS text here...")
        
        if st.button("Analyze Risk", type="primary"):
            if user_input:
                txt_vec = vectorizer.transform([clean_text(user_input)])
                meta_vec = scaler.transform(np.array([get_meta_features(user_input)]))
                final_vec = hstack([txt_vec, meta_vec])
                
                prediction = model.predict(final_vec)[0]
                probability = model.predict_proba(final_vec)[0][1]
                
                st.divider()
                if prediction == 1:
                    st.error(f"**SPAM DETECTED**")
                    st.write(f"**Confidence Level:** {probability:.1%}")
                else:
                    st.success(f"**SAFE / HAM**")
                    st.write(f"**Safety Score:** {1-probability:.1%}")
            else:
                st.warning("Please enter text.")

    with col2:
        st.info("**System Architecture**")
        st.markdown("""
        **Hybrid SVM Engine**
        * Trained on UCI Dataset + 2024 Injection
        * Uses TF-IDF + Meta-Features (Caps, URLs)
        """)

# TAB 2: Visualizations
with tab2:
    st.header("Evaluation Metrics")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption="Test Set Accuracy", use_column_width=True)
            
    with col_b:
        st.subheader("Feature Importance")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", caption="Top Predictors", use_column_width=True)