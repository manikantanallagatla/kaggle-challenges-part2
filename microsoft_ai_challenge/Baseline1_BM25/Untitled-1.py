import pandas as pd

data =  pd.read_csv('data.tsv', delimiter= "\t", encoding='utf-8', header=None)

data[1] = [w.lower() for w in data[1]]
data[2] = [w.lower() for w in data[2]]

data.head()
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *

nltk.download('stopwords')
stpwords = stopwords.words('english') + list(string.punctuation); print(stpwords)
data['query_words'] = [sentence.split(" ") for sentence in data[1]]
data['passage_words'] = [sentence.split(" ") for sentence in data[2]]
data.head()
data['query_words_filtered'] = [" ".join([w for w in query if w not in stpwords]) for query in data['query_words'] ]
data['passage_words_filtered'] = [" ".join([w for w in query if w not in stpwords]) for query in data['passage_words'] ]
data.head()

data["queries"] = data['query_words_filtered']
data["passages"] = data['passage_words_filtered']

data["queries"] = ["".join([w for w in query if w not in string.punctuation]) for query in data['queries']]
data["passages"] = ["".join([w for w in passage if w not in string.punctuation]) for passage in data['passages']]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()