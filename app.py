import streamlit as st
import joblib
from cleaning import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load("Positive-Or-Negative-Log-Reg-Model.pkl")
vectorizer = joblib.load("TFIDF-Vectorizer.pkl")

phrase = st.text_area("Text Field",placeholder="Please enter your text here...",height=1)
button = st.button("Analyze")

if button:
    if phrase:
        cleaned_phrase = preprocess_text(phrase)
        transformed_text = vectorizer.transform([cleaned_phrase])
        prediction = model.predict(transformed_text)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f'This Review is : {sentiment}')
    else:
        st.write("Please input text into the field")
