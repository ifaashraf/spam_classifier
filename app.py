import streamlit as st
st.set_page_config(page_title="Spam Detector", page_icon="📩")
import pandas as pd

# Title
st.title("Spam Message Detector 📩")

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Prepare data
X = df['message']
y = df['label']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X, y)

# User input
user_input = st.text_input("Enter a message:")

if st.button("Check"):
    if user_input:
        data = cv.transform([user_input])
        result = model.predict(data)[0]
        
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")