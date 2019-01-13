import re
from os import path

import nltk
import numpy as np
import wikipedia
from PIL import Image
from django.shortcuts import render, redirect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('averaged_perceptron_tagger')

currdir = "static"
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
MAX_ARTICLES = 30


def index(request):
    return render(request, "landing.html")


def feedback(request):
    if request.method == "POST":
        title = request.POST.get("title")
        content = request.POST.get("content")
        concepts = get_concepts(content)
        articles = [get_wiki(concept) for i, concept in enumerate(concepts) if i <= MAX_ARTICLES]
        tfidf = TfidfVectorizer()
        scores = []
        print(articles)
        for article in articles:
            if article:
                tfidf_matrix = tfidf.fit_transform([content, article["content"]])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]
                scores.append({"title": article["title"], "score": round((((similarity / 1) * 10 + 10) / 20) * 10, 3)})
        titles = []
        less_related = 0
        most_related = 0
        not_related = 0
        total = len(scores)
        for score in scores:
            if score["score"] > 2.0:
                less_related += 1
                titles.append(score["title"])
            if score["score"] > 3.5:
                most_related += 1
            else:
                not_related += 1

        create_wordcloud(" ".join(str(x) for x in titles))
        create_pi_chart(less_related, most_related, not_related, total)
        return render(request, "feedback.html")
    else:
        return redirect('dashboard:landing')


def create_wordcloud(text):
    mask = np.array(Image.open(path.join(currdir, "cloud.png")))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                   max_words=200,
                   mask=mask,
                   stopwords=stopwords)
    wc.generate(text)
    wc.to_file(path.join(currdir, "wordle.png"))


def create_pi_chart(less_related, most_related, not_related, total):
    most_related = (most_related/total)*100
    less_related = (less_related/total)*100
    not_related = (not_related/total)*100
    df = pd.DataFrame({'Concepts Distribution': [float("{0:.2f}".format(most_related)),
                                                 float("{0:.2f}".format(less_related)),
                                                 float("{0:.2f}".format(not_related))], },
                      index=[str(most_related), str(less_related), str(not_related)])
    p = df.plot.pie(y='Concepts Distribution', figsize=(6, 6), colors=['cadetblue', 'skyblue', 'lightcoral'])
    p.get_figure().savefig(path.join(currdir, "pi_chart.png"))
    plt.close()


def get_wiki(query):
    try:
        page = wikipedia.page(query)
        return {"title": page.title, "content": page.content}
    except:
        print(f"Too broad concept {query}")
    return ""


def get_concepts(content):
    words_extractor = WordsExtractor(content)
    return words_extractor.extracted_words


class WordsExtractor:

    """This class is responsible for exctracting & cleaning
        words from text string sent by word cloud generator module
        and store list of words extracted from string in
        extracted_words variable."""

    def __init__(self, input_txt):

        """This is constructor for words extractor class
            it takes text string and act as controller to
            expose text string to whole class."""

        tokenized_words = self.get_tokenized_text(input_txt)
        cleaned_words = self.remove_stop_words(tokenized_words)
        self.extracted_words = self.tag_words(cleaned_words)

    def get_tokenized_text(self, input_txt):

        """This function does tokenization of text string
            and return tokenized words as list.
        Parameters
        ----------
        input_txt : iterable of characters(string).
        Returns
        -------
        tokenized text : list of strings each string
            representing a word.
        """

        return input_txt.lower().split()

    def _clean_in(self, tokenized_txt):

        """This function clean word and discard
            non words and return remaining words as list.
        Parameters
        ----------
        tokenized_txt : iterable of tokenized strings.
        Returns
        -------
        tokenized_txt text : list of cleaned words strings
             each string representing a word.
        """

        tokenized_txt = [re.sub('\W+', ' ', x).strip() for x in tokenized_txt if x]
        return [x for x in tokenized_txt if x]

    def remove_stop_words(self, tokenized_txt):

        """This function removes stop words from tokenized
            words list and return remaining words as list.
        Parameters
        ----------
        tokenized_txt : iterable of tokenized strings.
        Returns
        -------
        cleaned text : list of cleaned non stop words strings
             each string representing a word.
        """

        tokenized_txt = self._clean_in(tokenized_txt)
        return [word for word in tokenized_txt if word not in STOPWORDS]

    def tag_words(self, cleaned_words):

        """This function tags words from cleaned words list
            and return only nouns, verbs and pronoun as list.
        Parameters
        ----------
        cleaned_words : iterable of cleaned strings.
        Returns
        -------
        tegged_words : list of tagged words strings each string
            representing a noun, verb or pronoun.
        """

        tagged_words = nltk.pos_tag(cleaned_words)
        tags = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD",
                "VBG", "VBN", "VBP", "VBZ"]
        return [word for word, tag in tagged_words if tag in tags]
