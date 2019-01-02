import numpy as np
import wikipedia
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from django.shortcuts import render, redirect

currdir = path.dirname(__file__) + "/static"


def index(request):
    return render(request, "landing.html")


def feedback(request):
    if request.method == "POST":
        title = request.POST.get("title")
        content = request.POST.get("content")
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
