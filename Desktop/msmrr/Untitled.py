#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


pip install --user -U nltk


# In[10]:


import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[11]:


movies.head(2)


# In[12]:


movies.shape


# In[13]:


credits.head()


# In[14]:


movies = movies.merge(credits,on='title')


# In[15]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[16]:


movies.head()


# In[17]:


import ast


# In[18]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L


# In[19]:


movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies.head()


# In[20]:


movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


# In[21]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[22]:


movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[23]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# In[24]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[25]:


movies.sample(5)


# In[26]:


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[27]:


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)


# In[28]:


movies.head()


# In[29]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
    
vector = cv.fit_transform(new['tags']).toarray()
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


# In[31]:


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


# In[35]:


recommend('Avatar')


# In[37]:


pickle.dump(new,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




