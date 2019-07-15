#!/usr/bin/env python
# coding: utf-8

# .py file with helper functions
# In[106]:
import numpy as np
import pandas as pd
import re
import os
import wikipedia
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem import WordNetLemmatizer
from itertools import chain
from difflib import get_close_matches as gcm
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


# In[39]:
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# In[108]:
# function to lemmatize words according to corresponding part of speech
# will try to include different lemmatizers i.espacey, stanford, textblob
def lemma(word):
    if get_wordnet_pos(word) == 'r':
        try:
            possible_adj = []
            for ss in wordnet.synsets(word):
              for lemmas in ss.lemmas(): # all possible lemmas
                  for ps in lemmas.pertainyms(): # all possible pertainyms
                      possible_adj.append(ps.name())
            word = gcm(word,possible_adj)[0]
        except:
            word = word
    lemmatizer = WordNetLemmatizer()
    pos = get_wordnet_pos(word)
    lemmatized_word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    return lemmatized_word


# In[67]:
# function to clean up and tokenize the raw text of the corpus
def preprocess_corpus_text(raw_string,lemmatize=True):
    transtable = str.maketrans('', '', string.punctuation)
    raw_string = re.sub(r'(?<=[.,])(?=[^\s])', r' ', raw_string)
    stop_words=set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()
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
    if lemmatize:
        lemmatized_tokens = []
        for i in range(len(final_tokens)):
            wordtoks = []
            for word in final_tokens[i]:
                wordlemma = lemma(word)
                wordtoks.append(wordlemma)
            lemmatized_tokens.append(wordtoks)
        return lemmatized_tokens
    else:
        return final_tokens


# In[118]:


#teststr = 'the next day steve is summoned to the brooklyn bunker to see phillips and stark. steve is approached by a beautiful female officer who wishes to thank him for his service the best way she knows how. peggy walks in on steve kissing the enlisted-woman and angrily storms away. steve apologetically follows her to starks lab, insisting that he gets nervous around women and asks why he should apologize if carter and stark have a thing going. stark quickly shoots down the rumored relationship and takes steve to his weapons engineering lab. he remarks that rogers has become attached to the triangular shield, which steve says is a handy tool in the field. on a table are several prototype shields with sophisticated components, however steve finds a plain, circular shield on a lower shelf.'


# In[119]:


#preprocess_corpus_text(teststr,lemmatize=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




