import pandas as pd

data =  pd.read_csv('dataMin.tsv', delimiter= "\t", encoding='utf-8', header=None)

data.head()
# data[5] = [w.lower() for w in data[]]
# data[6] = [w.lower() for w in data[2]]
#
# data.head()
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *

nltk.download('stopwords')
stpwords = stopwords.words('english') + list(string.punctuation); print(stpwords)
# data['query_words'] = [sentence.split(" ") for sentence in data[1]]
# data['passage_words'] = [sentence.split(" ") for sentence in data[2]]
# data.head()
# data['query_words_filtered'] = [" ".join([w for w in query if w not in stpwords]) for query in data['query_words'] ]
# data['passage_words_filtered'] = [" ".join([w for w in query if w not in stpwords]) for query in data['passage_words'] ]
# data.head()
# data[0].head()
# data["queries"] = data['query_words_filtered']
# data["passages"] = data['passage_words_filtered']
#
# data["queries"] = ["".join([w for w in query if w not in string.punctuation]) for query in data['queries']]
# data["passages"] = ["".join([w for w in passage if w not in string.punctuation]) for passage in data['passages']]

docIDFDict = {}
avgDocLength = 0

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = TfidfVectorizer()
docIDFDict = vectorizer.fit_transform(data[6])

docIDFDict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

docIDFDict['often']

avgDocLength = sum([len(x) for x in data[6] ])/len(data[6])

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

def RunBM25OnEvaluationSet(outputfile, df):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    #f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for index, row in df.iterrows():
        #print(row[0])
        Query = row['queries']
        Passage = row['passages']
        score = GetBM25Score(Query,Passage)
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
test_data[1] = [w.lower() for w in test_data[1]]
test_data[2] = [w.lower() for w in test_data[2]]

test_data['query_words'] = [sentence.split(" ") for sentence in test_data[1]]
test_data['passage_words'] = [sentence.split(" ") for sentence in test_data[2]]

test_data['queries'] = [" ".join([w for w in query if w not in stpwords]) for query in test_data['query_words'] ]
test_data['passages'] = [" ".join([w for w in query if w not in stpwords]) for query in test_data['passage_words'] ]

test_data["queries"] = ["".join([w for w in query if w not in string.punctuation]) for query in test_data['queries']]
test_data["passages"] = ["".join([w for w in passage if w not in string.punctuation]) for passage in test_data['passages']]

RunBM25OnEvaluationSet("answer.tsv", test_data)

data = data.drop(columns = [1, 2, 'query_words_filtered', 'query_words', 'passage_words', 'passage_words_filtered'])
data.head()
data.to_csv('datacleaned.tsv', sep= "\t", encoding='utf-8', header=None)
