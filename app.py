# Importing Libraries
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from pickle import load
from flask import Flask, render_template, request

ps = PorterStemmer()

# Text Clean
def text_clean(txt):
    txt = txt.lower()
    txt = word_tokenize(txt)
    txt = [t for t in txt if t not in punctuation]
    txt = [t for t in txt if t not in stopwords.words("english")]
    txt = [ps.stem(t) for t in txt]
    txt = " ".join(txt)
    return txt

# Load the fitted TfidfVectorizer
with open("vector.pkl", "rb") as f:
    tv = load(f)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = load(f)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    msg = ""
    if request.method == "POST":
        # Prediction
        txt = request.form["review"].strip()  # Strip whitespace
        if not txt:
            msg = "Please enter some text to check."
        else:
            ctxt = text_clean(txt)
            vtxt = tv.transform([ctxt])
            res = model.predict(vtxt)
            msg = "The mail is a Normal Email" if res[0] == "ham" else "The mail is a Spam Email"
        
    return render_template("home.html", msg=msg)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
