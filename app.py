import streamlit as st
import joblib
from cleaning import preprocess_text

model = joblib.load("Positive-Or-Negative-Log-Reg-Model.pkl") #load the model
vectorizer = joblib.load("TFIDF-Vectorizer.pkl") #load the vectorizer

st.title("Movie Review Sentiment Predictor")
phrase = st.text_area("Enter your review here",placeholder="Please enter your text here...",height=1) #text input area
button = st.button("Analyze")

if button:
    if phrase:
        cleaned_phrase = preprocess_text(phrase) #Cleans the user's input for prediction 
        transformed_text = vectorizer.transform([cleaned_phrase]) #Transforms the cleaned phrase into features
        prediction = model.predict(transformed_text)[0] #Predicts if the text is positive or negative

        if prediction == 1:
            st.success("This review is positive!")
        elif prediction == 0:
            st.error("This review is negative!")
    else:
            st.write("Please input text into the field")
