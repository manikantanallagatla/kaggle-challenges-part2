
import pandas as pd
import spacy
data =  pd.read_csv('dataMin.tsv', delimiter= "\t", encoding='utf-8', header=None)
data.head()

from gensim.models.keyedvectors import KeyedVectors
from DocSim import DocSim

# Using the pre-trained word2vec model trained using Google news corpus of 3 billion running words.
# The model can be downloaded here: https://bit.ly/w2vgdrive (~1.4GB)
# Feel free to use to your own model.
googlenews_model_path = '/Users/sangu/Downloads/GoogleNews-vectors-negative300.bin.gz'

model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True)

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *

nltk.download('stopwords')
stpwords = stopwords.words('english') + list(string.punctuation); print(stpwords)
ds = DocSim(model,stopwords=stpwords)

# data = data.dropna()
# len(data)
# from nltk.stem.porter import *
# import re
# words = re.compile(r"\w+",re.I)
#
# stemmer = PorterStemmer()
# queries = []
# passages = []
# for q in data[4].tolist():
#         queries.append([stemmer.stem(i.lower()) for i in words.findall(q)])
# for q in data[5].tolist():
#         passages.append([stemmer.stem(i.lower()) for i in words.findall(q)])
#
# queries = [" ".join([w for w in sentence]) for sentence in queries]
# passages = [" ".join([w for w in sentence]) for sentence in passages]
# len(queries)
# len(data[4])
# len(passages)
# len(data[5])
# queries[:4]
# passages[:4]

# data[4] = queries
# data[5] = passages
data.head()

docIDFDict = {}
avgDocLength = 0

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = TfidfVectorizer()
docIDFDict = vectorizer.fit_transform(data[6])

docIDFDict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

docIDFDict['often']

avgDocLength = sum([len(x) for x in data[6] ])/len(data[6])

avgDocLength
# new_data = pd.concat([pos_data,min_neg_data])
#
# len(new_data)
#
# new_data = new_data.sample(frac=1).reset_index(drop=True)
#
# new_data.head()
# new_data.to_csv('dataMin.tsv', sep= "\t", encoding='utf-8', header=None)
# new_data[[4,5]]
# new_data[[4,5]].to_csv('datacdssm.tsv', sep= "\t", encoding='utf-8', header=None, index=False)

def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :

    global docIDFDict,avgDocLength

    query_words= Query.strip().lower().split(delimiter)
    passage_words = Passage.strip().lower().split(delimiter)
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

def GetDocSimScore(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :

    source_doc = Query
    # target_docs = ['delete a invoice', 'how do i remove an invoice', "purge an invoice"]
    target_docs = [str(Passage)]

    sim_scores = ds.calculate_similarity(source_doc, target_docs)
    if(len(sim_scores) == 1):
        score = (sim_scores[0]['score'])
        return score

    return 0

score1 = []
for query, passage in zip(data[5], data[6]):
    score1.append(GetBM25Score(query, passage))
data['score1'] = score1

score2 = []
for query, passage in zip(data[5], data[6]):
    score2.append(GetDocSimScore(query, passage))
data['score2'] = score2
data[:20]

cdssm = pd.read_csv('cdssm_out.score.txt', delimiter= "\t", encoding='utf-8', header=None)
dssm = pd.read_csv('dssm_out.score.txt', delimiter= "\t", encoding='utf-8', header=None)

data['score3'] = cdssm
data['score4'] =  dssm

x_train = pd.DataFrame()
x_test = pd.DataFrame()
x_train['score1'] = data['score1']
x_train['score2'] = data['score2']
x_train['score3'] = data['score3']
x_train['score4'] = data['score4']

y_train = data[3]

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

def RunBM25OnEvaluationSet(outputfile, df):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    #f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for index, row in df.iterrows():
        #print(row[0])
        Query = row['queries']
        Passage = row['passages']
        score = row['score']
        tempscores.append(score)
        lno+=1
        if(lno%10==0):
            tempscores = [str(s) for s in tempscores]
            scoreString = "\t".join(tempscores)
            qid = str(row[0])
            fw.write(qid+"\t"+scoreString+"\n")
            tempscores=[]
        if(lno%5000==0):
            print(lno)
    print(lno)
    #f.close()
    fw.close()

test_data = pd.read_csv('eval1_unlabelled.tsv', delimiter= "\t", encoding='utf-8', header=None)
test_data.head()
test_data[1] = [w.lower() for w in test_data[1]]
test_data[2] = [w.lower() for w in test_data[2]]

test_data['query_words'] = [sentence.split(" ") for sentence in test_data[1]]
test_data['passage_words'] = [sentence.split(" ") for sentence in test_data[2]]

test_data['queries'] = [" ".join([w for w in query if w not in stpwords]) for query in test_data['query_words'] ]
test_data['passages'] = [" ".join([w for w in query if w not in stpwords]) for query in test_data['passage_words'] ]

test_data["queries"] = ["".join([w for w in query if w not in string.punctuation]) for query in test_data['queries']]
test_data["passages"] = ["".join([w for w in passage if w not in string.punctuation]) for passage in test_data['passages']]

from nltk.stem.porter import *
import re
words = re.compile(r"\w+",re.I)

stemmer = PorterStemmer()
queries = []
passages = []
for q in test_data['queries'].tolist():
        queries.append([stemmer.stem(i.lower()) for i in words.findall(q)])
for q in test_data['passages'].tolist():
        passages.append([stemmer.stem(i.lower()) for i in words.findall(q)])

test_data['queries'] = [" ".join([w for w in sentence]) for sentence in queries]
test_data['passages'] = [" ".join([w for w in sentence]) for sentence in passages]
test_data.head()

testdata = pd.DataFrame()
len(test_data)

test_score1 = []
for query, passage in zip(test_data['queries'], test_data['passages']):
    test_score1.append(GetBM25Score(query, passage))
testdata['score1'] = test_score1

test_score2 = []
for query, passage in zip(test_data['queries'], test_data['passages']):
    test_score2.append(GetDocSimScore(query, passage))
testdata['score2'] = test_score2
testdata[:20]

testcdssm = pd.read_csv('val_cdssm_out.score.txt', delimiter= "\t", encoding='utf-8', header=None)
testdssm = pd.read_csv('val_dssm_out.score.txt', delimiter= "\t", encoding='utf-8', header=None)

testdata['score3'] = testcdssm
testdata['score4'] =  testdssm

d_test = xgb.DMatrix(testdata)
p_test = bst.predict(d_test)

p_test_100 = [x*100 for x in p_test]

testdata['score1'][:5]
y[:5]
y = p_test_100 + testdata['score1']
test_data['score'] = y
RunBM25OnEvaluationSet("answer.tsv", test_data)
