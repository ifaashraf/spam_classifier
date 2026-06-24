import pandas as pd

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham':0, 'spam':1})

X = df['message']
y = df['label']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

msg = ["Free entry in 2 a weekly competition!"]
msg = cv.transform(msg)
print("Spam" if model.predict(msg)[0] == 1 else "Not Spam")

msg = input("Enter a message: ")
msg = cv.transform([msg])
result = "Spam" if model.predict(msg)[0] == 1 else "Not Spam"
print("Prediction:", result)