#!/usr/bin/env python
# coding: utf-8

# In[1]:


# .py file with helper functions


# In[12]:


import numpy as np
import pandas as pd
import re
import os
import wikipedia
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
from imdb import IMDb


# In[15]:


#function to extract list of wiki page titles
def titles_list(filepath):
    movies = pd.read_excel(open(filepath,'rb'))
    titles_list = movies['wikipedia page title']
    return titles_list


# In[21]:


# function to extract imdb movie ids
def imdb_id_list(filepath):
    movies = pd.read_excel(open(filepath,'rb'))
    id_list = movies['imdb id']
    return id_list


# In[36]:


#function to pull the text of plot section from a movie wiki page
def plot_puller_wiki(movie):
    movie_plot = wikipedia.WikipediaPage(title = movie).section('Plot')
    movie_plot = movie_plot.replace('\n','').replace("\'","").replace("\\","").lower()
    return movie_plot


# In[37]:


def plot_puller_imdb(imdb_movie_id):
    ia = IMDb()
    movie = ia.get_movie(imdb_movie_id)
    movie_plot = str(movie['synopsis'])
    movie_plot = movie_plot.replace('\n','').replace("\'","").replace("\\","").lower()
    return movie_plot


# In[19]:


# function to aggregate plots of all movies to be inclued in the corpus
def plot_aggregator_wiki(titles_list):
    plot_agg = ''
    for i, movie in enumerate(titles_list):
        movie_plot = plot_puller_wiki(movie)
        plot_agg += movie_plot
    return plot_agg


# In[42]:


def plot_aggregator_imdb(imdb_movie_id_list):
    plot_agg = ''
    for i, movie_id in enumerate(imdb_movie_id_list):
        movie_plot = plot_puller_imdb(movie_id[2:])
        plot_agg += movie_plot
    return plot_agg


# In[63]:


#%time plots = plot_aggregator_imdb(idls[:-1])


# In[64]:


#plots


# In[65]:


# function to clean up and tokenize the raw text of the corpus
def preprocess_corpus_text(raw_string):
    transtable = str.maketrans('', '', string.punctuation)
    raw_string = re.sub(r'(?<=[.,])(?=[^\s])', r' ', raw_string)
    stop_words=set(stopwords.words('english'))
    sentence_tokens = sent_tokenize(raw_string)
    word_tokens  = []
    for sentence in sentence_tokens:
        clean_sentence = sentence.translate(transtable)
        tok = word_tokenize(clean_sentence)
        word_tokens.append(tok)
    final_tokens = []
    for sentence in word_tokens:
        ntk = [w for w in sentence if not w in stop_words]
        final_tokens.append(ntk)
    return final_tokens


# In[66]:


#plot_tokens = preprocess_corpus_text(plots)
#plot_tokens[-100:-1]


# In[ ]:


# function for stemming and lemmatization


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




