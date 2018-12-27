##### import
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm
%matplotlib inline
import random
import sys
import time

##### load data
labeled_images = pd.read_csv('train.csv')
labeled_images
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.9, random_state=0)

##### data visualization
i= random.randint(1,500)
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.hist(train_images.iloc[i])

##### svm classifier
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
clf.score(test_images,test_labels)

##### clean data
test_images[test_images>0]=1
train_images[train_images>0]=1

img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])

##### load test data and submission
test_data=pd.read_csv('test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data)
results
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)

##### improve SVM by grid search
svm = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':(1,2,4,6,8,10), 'gamma': (0.1,0.4,0.8, 1.0,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
clf = GridSearchCV(svm, parameters)
clf.fit(train_images, train_labels.values.ravel())


from IPython.display import HTML
import base64

def create_download_link( df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = f'<a target="_blank">{title}</a>'
    return HTML(html)
