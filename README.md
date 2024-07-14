## MailGuard: Intelligent Spam Detection

### Overview
MailGuard is a Streamlit-based application for classifying emails as spam or ham using a Multinomial Naive Bayes model and TfidfVectorizer for text vectorization.

### Usage
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/vinesh70/MailGuard-Intelligent-Spam-Detection.git
   cd MailGuard-Intelligent-Spam-Detection

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the App:**
   ```bash
   streamlit run app.py

### Input Examples:
1. Tone Club: Your subs has now expired 2 re-sub reply MONOC 4 monos or POLYC 4 polys 1 weekly @ 150p per week Txt STOP 2 stop This msg free Stream 0871212025016
2. XMAS Prize draws! We are trying to contact U. Todays draw shows that you have won a £2000 prize GUARANTEED. Call 09058094565 from land line. Valid 12hrs only
3. SMS SERVICES For your inclusive text credits pls gotto www.comuk.net login 3qxj9 unsubscribe with STOP no extra charge help 08702840625 comuk.220cm2 9AE


### How It Works:
1. Text Cleaning: Lowercasing, tokenization, removal of punctuation and stop words, and stemming using NLTK's PorterStemmer.
2. Vectorization: Convert cleaned text into numerical vectors using TfidfVectorizer.
3. Model Training: Multinomial Naive Bayes classifier trained on vectorized text data.
4. Prediction: Input text is cleaned and vectorized, then fed into the trained model to predict whether it's spam or ham.


### Deployment:
This app can be deployed for free on Streamlit Sharing, Heroku, or other cloud platforms. Follow the respective platform's deployment guide to get your app online.


### Feedback:
Your feedback and contributions are welcome! Feel free to create issues or pull requests.


### Contributing:
Feel free to open issues or submit pull requests if you have any improvements or suggestions.


### Author:
**Author**: [Vinesh Ryapak](https://www.linkedin.com/in/vinesh-ryapak-73693a227/)
