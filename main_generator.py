#!/usr/bin/env python
# coding: utf-8

# # Plot summary generator

# In[229]:


## adding directory to system path --- execute only once
#import sys,os
#os.chdir('C:\\Users\\Rohit Suresh\\Jupyter projects\\Plot summary generator')
#sys.path.append(os.getcwd())
#sys.path


# In[230]:


# import necessary packages and libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
importlib.reload(plot_summary_generator_utils)
from plot_summary_generator_utils import *


# # Train embedding matrix
# ## 1. Obtain plot summaries
# 
# Plot summaries will be obtained from Wikipedia. We will use the 'wikipedia' package to operate on wikipedia pages.

# In[2]:


filepath = 'corpus_movies_list.xlsx'
imdb_ids = imdb_id_list(filepath)
imdb_ids


# In[3]:


get_ipython().run_line_magic('time', 'corpus_raw_text = plot_aggregator_imdb(imdb_ids)')


# In[4]:


corpus_raw_text


# In[220]:


corpus_tokens = preprocess_corpus_text(corpus_raw_text)


# In[221]:


corpus_tokens[-50:]


# In[7]:


# import necessary libraries
from gensim.models import Word2Vec


# In[86]:


get_ipython().run_line_magic('time', 'embed = Word2Vec(corpus_tokens, min_count = 3, sg = 1, iter=100, negative=10, sorted_vocab=1)')


# In[87]:


print(embed)


# In[88]:


embed_words = list(embed.wv.vocab)
print(embed_words[:25])


# In[89]:


print(embed['thanos'])


# In[90]:


word_vectors = embed.wv


# In[91]:


res = word_vectors.most_similar(positive=['wasp','man'],negative=['woman'])


# In[92]:


res[:3]


# In[93]:


res1 = word_vectors.most_similar(positive=['barton','woman'],negative=['man'])


# In[94]:


res1[:3]


# In[117]:


res2 =word_vectors.most_similar(positive=['thor','tony'],negative=['pepper'])


# In[118]:


res2[:3]


# In[100]:


res3 = word_vectors.similar_by_word('peter')


# In[101]:


res3[:3]


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





# In[231]:


teststr1 = 'the next day steve is summoned to the brooklyn bunker to see phillips and stark. steve is approached by a beautiful female officer who wishes to thank him for his service the best way she knows how. peggy walks in on steve kissing the enlisted-woman and angrily storms away. steve apologetically follows her to starks lab, insisting that he gets nervous around women and asks why he should apologize if carter and stark have a thing going. stark quickly shoots down the rumored relationship and takes steve to his weapons engineering lab. he remarks that rogers has become attached to the triangular shield, which steve says is a handy tool in the field. on a table are several prototype shields with sophisticated components, however steve finds a plain, circular shield on a lower shelf.'


# In[232]:


t = preprocess_corpus_text(teststr1,lemmatize=True)


# In[215]:


t


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




