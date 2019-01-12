import re
import os
import nltk
import numpy as np
import wikipedia
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from django.shortcuts import render, redirect

currdir = path.dirname(__file__) + "/static"
STOPWORDS = os.path.join(currdir, "stop_words_list.txt")


def index(request):
    return render(request, "landing.html")


def feedback(request):
    if request.method == "POST":
        title = request.POST.get("title")
        content = request.POST.get("content")
        concepts = get_concepts(content)
        articles = [get_wiki(concept) for concept in concepts]
        # TODO: Implement cosine similarity between articles & title of content to know how much relevant it is
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


def get_wiki(query):
    page = wikipedia.page(query)
    return page.content


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

        with open(STOPWORDS, "r") as stop_words_file:
            stop_words = stop_words_file.read()
            return [word for word in tokenized_txt if word not in stop_words]

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
