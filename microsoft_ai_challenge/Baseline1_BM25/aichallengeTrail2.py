import pandas as pd

# %% Load stopwords
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import *

nltk.download('stopwords')
stpwords = stopwords.words('english') + list(string.punctuation); print(stpwords)

# %% Loading test set
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

test_data.head()
test_data[['queries', 'passages']].to_csv('validationdata2.txt', sep= "\t", encoding='utf-8', header=None, index=False)

cdssm_score = pd.read_csv('cdssm_out.score.txt', sep= "\t", encoding='utf-8', header=None)
dssm_score = pd.read_csv('dssm_out.score.txt', sep= "\t", encoding='utf-8', header=None)

def score(outputfile, df):

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

answers = cdssm_score + dssm_score
answers[:3]
test_data['score'] = answers
score("answer.tsv", test_data)

data = data.drop(columns = [1, 2, 'query_words_filtered', 'query_words', 'passage_words', 'passage_words_filtered'])
data.head()
data.to_csv('datacleaned.tsv', sep= "\t", encoding='utf-8', header=None)
