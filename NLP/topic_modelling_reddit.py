# coding: utf-8

import os
import re
import string


import pandas as pd
import json
import gensim
from gensim import corpora

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation



lines = []
folder_path = "./data/"
for filename in os.listdir(folder_path):
    if "RC" in filename:
        with open(folder_path + filename) as f:
            lines += f.readlines()

data = [json.loads(line) for line in lines]

filtered_data = [x for x in data if x["subreddit"] in ["reddit.com"]]
df = pd.DataFrame(filtered_data)
df = df[df["ups"] > 5]

def clean_builder():
    non_standart_char = re.compile(r'[^a-zA-Z0-9]')
    stop = set(stopwords.words('english'))
    stop.update(("to","cc","subject","http","from","sent"))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    porter= PorterStemmer()
    
    def clean(text):
        text=text.strip()
        text = non_standart_char.sub(' ', text)
        stop_free = " ".join([i for i in text.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        #stem = " ".join(porter.stem(token) for token in normalized.split())
        return normalized
    
    return clean

clean = clean_builder()
text_clean=[]
for text in df['body']:
    text_clean.append(clean(text).split)
    
dictionary = corpora.Dictionary(text_clean)
text_term_matrix = [dictionary.doc2bow(text) for text in text_clean]

Lda = gensim.models.LdaMulticore
for N in [64,128,256]:
    print(N)
    ldamodel = Lda(text_term_matrix, num_topics=N, id2word = dictionary, workers=9)
    print(ldamodel.print_topics(num_topics=N, num_words=10))
    try:
        ldamodel.save("lda_" + str(N) + ".model")
    except:
        pass
