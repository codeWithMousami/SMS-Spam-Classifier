import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()

# Custom CSS for the background image
background_css = """
<style>
body {
    background-image: url('email.jpg');
    background-size: cover;
}
</style>
"""

# Render the background image CSS
st.markdown(background_css, unsafe_allow_html=True)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stopwords and punctuations
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ''.join(y)

with open('Vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('Email/SMS Spam Classifier ')
input_sms = st.text_area('Enter The message ')

if st.button('Predict'):
    # 1-Preprocess
    transformed_sms = transform_text(input_sms)

    # 2-Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3- Predict
    result = model.predict(vector_input)[0]

    # 4- Display
    if result == 0:
        st.header('Spam')
    else:
        st.header('Not Spam')
