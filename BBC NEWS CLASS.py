import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    le = joblib.load('label_encoder.joblib')
    return model, preprocessor, le

model, preprocessor, le = load_artifacts()

# App title
st.title("BBC News Article Classifier 📰")
st.markdown("Predict news article categories using ML")

# File upload section
uploaded_file = st.file_uploader("Upload news articles (CSV)", type=["csv"])
text_input = st.text_area("Or enter article text directly:")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Prediction function
def make_predictions(text_data):
    # Preprocess
    X_transformed = preprocessor.transform(text_data)
    # Predict
    preds_encoded = model.predict(X_transformed)
    # Decode
    preds = le.inverse_transform(preds_encoded)
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_transformed)
    else:
        probs = np.zeros((len(text_data), len(le.classes_)))
    return preds, probs

# Visualization function
def plot_results(preds, probs):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Class distribution
    class_counts = pd.Series(preds).value_counts()
    ax[0].bar(class_counts.index, class_counts.values)
    ax[0].set_title("Predicted Class Distribution")
    ax[0].tick_params(axis='x', rotation=45)

    # Confidence scores
    if probs is not None:
        max_probs = np.max(probs, axis=1)
        ax[1].hist(max_probs, bins=20)
        ax[1].set_title("Prediction Confidence Distribution")

    st.pyplot(fig)

# Process inputs
if uploaded_file or text_input:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Text' not in df.columns:
            st.error("CSV must contain 'Text' column")
        else:
            texts = df['Text']
    else:
        texts = [text_input]

    if st.button("Classify Articles"):
        preds, probs = make_predictions(texts)
        st.session_state.predictions = (preds, probs, texts)

# Display results
if st.session_state.predictions:
    preds, probs, texts = st.session_state.predictions

    # Show predictions table
    results_df = pd.DataFrame({
        'Article': texts,
        'Predicted Category': preds,
        'Confidence': np.max(probs, axis=1) if probs is not None else ['N/A']*len(preds)
    })
    st.dataframe(results_df.style.format({'Confidence': '{:.2%}'}))

    # Show visualizations
    st.subheader("Analysis")
    plot_results(preds, probs)

    # Show sample predictions
    st.subheader("Sample Prediction Details")
    sample_idx = st.slider("Select article", 0, len(texts)-1, 0)
    st.markdown(f"**Article Text:**\n{texts[sample_idx]}")
    st.markdown(f"**Predicted Category:** {preds[sample_idx]}")

    if probs is not None:
        prob_df = pd.DataFrame({
            'Category': le.classes_,
            'Probability': probs[sample_idx]
        }).sort_values('Probability', ascending=False)

        st.bar_chart(prob_df.set_index('Category'))