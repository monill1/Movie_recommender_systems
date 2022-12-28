#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd


# In[47]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[48]:


movies.head(1)


# In[49]:


credits.head(1)


# In[50]:


credits.head(1)['crew'].values


# In[51]:


credits.head(1)['cast'].values


# In[52]:


movies = movies.merge(credits, on = 'title')


# In[53]:


movies.head(1)


# In[54]:


# now removing collumn which is not useful for recommendation
# preparing the list of useful collumns


# In[55]:


# genres
# id
# keywords
# title
# overview
# cast
# crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[56]:


movies.info()


# In[57]:


movies.head()


# In[58]:


movies.isnull().sum()


# In[59]:


movies.dropna(inplace=True)


# In[60]:


movies.duplicated().sum()


# In[61]:


# Formatting


# In[62]:


movies.iloc[0].genres


# In[63]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# in this format

# ['Action','Adventure','Fantasy','SciFi']


# In[64]:


def convert(obj):
    L = []
    for i in obj:
        L.append(i['name'])
    return L


# In[65]:


convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[ ]:


# first we have to convert string of list to list


# In[66]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[67]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[69]:


movies['genres'] = movies['genres'].apply(convert)


# movies.head()

# In[70]:


movies.head()


# In[71]:


# now do same on keywords


# In[73]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[74]:


movies.head()


# In[76]:


# now formate the cast collumn
movies['cast'][0]
# This is for first movie avatar now we only want first three name 


# In[77]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[79]:


movies['cast'] = movies['cast'].apply(convert3)


# In[80]:


movies.head()


# In[83]:


# now move on to crew
movies['crew'][0]
# in this i only need where job = 'Director'


# In[84]:


def fetch_director(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[86]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[88]:


movies.head()
# convert to what we want


# In[90]:


movies['overview'][0]
# now we also want to convert it into list


# In[92]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[94]:


movies.head()
# Now all comlumns are in list


# In[97]:


movies.iloc[0]


# In[98]:


# now i have to remove all the space between word for machine to understant nicly


# In[101]:


# 'Sam Worthington' to 'SamWorthington' and so on
# for less confusion between names for recommender


# In[103]:


movies['genres'] = movies['genres'].apply(lambda x : [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(" ","") for i in x])


# In[104]:


movies.head()


# In[105]:


# now make new columns with name of 'Tags'
# and it is concatination of last 4 columns


# In[106]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[107]:


movies.head()


# In[108]:


new_df = movies[['movie_id', 'title', 'tags']]


# In[109]:


new_df


# In[111]:


# now converting list of tags into string
new_df['tags'] = new_df['tags'].apply(lambda x : " ".join(x))


# In[112]:


new_df.head()


# In[113]:


new_df['tags'][0]


# In[114]:


# now conver it to lowercase


# In[116]:


new_df['tags'] = new_df['tags'].apply(lambda x : x.lower())


# In[118]:


new_df.head()


# ### Now do an vactorisation
# 

# In[143]:


# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[144]:


cv.fit_transform(new_df['tags']).toarray()


# In[145]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[146]:


vectors


# In[147]:


# movies in vextor 
vectors[0]


# In[148]:


cv.get_feature_names()


# In[129]:


# now we don't wnat it different like actor and actors and also for other words also
# Remove this


# In[130]:


# now apply stamming for this grammar differntiate


# In[131]:


import nltk


# In[132]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[137]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[136]:


ps.stem('dancing')


# In[142]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[140]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[149]:


from sklearn.metrics.pairwise import cosine_similarity


# In[153]:


similarity = cosine_similarity(vectors)


# In[154]:


similarity.shape


# In[155]:


similarity[0]


# In[ ]:


def recommend(movie):
    return


# In[158]:


new_df[new_df['title'] == 'Avatar'].index[0]


# In[160]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[163]:


sorted(similarity[0], reverse = True)


# In[165]:


sorted(list(enumerate(similarity[0])), reverse = True)


# In[166]:


sorted(list(enumerate(similarity[0])), reverse = True, key=lambda x:x[1])


# In[167]:


# now fetch first 5 of them
sorted(list(enumerate(similarity[0])), reverse = True, key=lambda x:x[1])[1:6]


# In[173]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[174]:


recommend('Avatar')


# In[172]:


new_df.iloc[1216].title


# In[182]:


recommend('Shutter Island')


# In[183]:


import pickle


# In[184]:


pickle.dump(new_df, open('movies.pkl','wb'))


# In[191]:


new_df['title'].values


# In[192]:


new_df.to_dict()


# In[194]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))


# In[195]:


pickle.dump(similarity,open('similarity.pkl', 'wb'))


# In[ ]:




