import streamlit as st
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from pickle import load

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Text cleaning function
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

# Custom CSS to disable textarea resizing
st.markdown(
    """
    <style>
    textarea {
        resize: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Spam Classifier App by Vinesh Ryapak")
st.write("Enter the text you want to classify as spam or ham:")

# Text input
user_input = st.text_area("")

# Prediction
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to check.")
    else:
        cleaned_text = text_clean(user_input)
        vectorized_text = tv.transform([cleaned_text])
        prediction = model.predict(vectorized_text)
        result = "The mail is a Normal Email" if prediction[0] == "ham" else "The mail is a Spam Email"
        st.success(result)

st.markdown("---")
st.write("Made by Vinesh Ryapak")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Streamlit app is running")
