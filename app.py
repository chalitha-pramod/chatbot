import PyPDF2
from flask import Flask, render_template, request, redirect, url_for, session
import secrets
from flask_bootstrap import Bootstrap
from newspaper import Article
import random
import string
import numpy as np
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io


warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article.text


def get_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        pdf_text += pdf_reader.pages[page_num].extract_text()
    return pdf_text

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))


greeting_input = ["hi", "hello", "hey", "hola"]
greeting_response = ["howdy", "hey there", "hi", "hello :)"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_input:
            return random.choice(greeting_response)


def response(user_response, sent_tokens):
    robo_response = ''
    sent_tokens.append(user_response)
    tfidfvec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = tfidfvec.fit_transform(sent_tokens)
    val = cosine_similarity(tfidf[-1], tfidf)
    idx = val.argsort()[0][-2]
    flat = val.flatten()
    flat.sort()
    score = flat[-2]
    if score == 0:
        robo_response = robo_response + "Sorry, I don't understand."
    else:
        robo_response = robo_response + sent_tokens[idx]

    sent_tokens.remove(user_response)
    return robo_response

secret_key_1 = secrets.token_hex(16)

app = Flask(__name__)
app.secret_key = secret_key_1
bootstrap = Bootstrap(app)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    input_type = request.form.get('input_type')
    if input_type == 'url':
        url = request.form.get('url')
        session['url'] = url
    else:
        pdf_file = request.files.get('pdf_file')
        if pdf_file:
            session['pdf_text'] = get_pdf_text(pdf_file)
        else:
            session['pdf_text'] = None
    session['input_type'] = input_type
    return render_template("chatbot.html", input_type=input_type, url=session.get('url', ''))


@app.route("/get")
def get_bot_response():
    url = session.get('url')
    user_input = request.args.get('msg')
    if user_input == 'bye':
        return ' See you later :)'
    elif greeting(user_input) is not None:
        return  greeting(user_input)
    else:
        url = request.args.get('url')
        if not url:
            return 'Please enter a Wikipedia URL in the format /get?url=<url>'
        article_text = get_article_text(url)
        sent_tokens = nltk.sent_tokenize(article_text)
        return response(user_input, sent_tokens)


@app.route("/get_pdf", methods=["POST"])
def get_bot_response_pdf():
    pdf_text = session.get('pdf_text')
    user_input = request.form.get('msg')
    if user_input == 'bye':
        return 'See you later :)'
    elif greeting(user_input) is not None:
        return greeting(user_input)
    else:
        if not pdf_text:
            return 'Please upload a PDF file'
        sent_tokens = nltk.sent_tokenize(pdf_text)
        return response(user_input, sent_tokens)




if __name__ == "__main__":
    app.run(debug=True)      