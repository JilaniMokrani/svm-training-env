import librosa.display
import scipy.io.wavfile as wavfile
import numpy
from os import environ
import os
import os.path
from os import walk
from scipy import stats
import numpy as np
import librosa 
import numpy as np
from scipy.stats import norm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from pycm import *

dataPath = environ["DATA_SOURCE"]

print(os.listdir(dataPath))
a = []
b = []
i = 0
min = 10000
for label in os.listdir(dataPath):
    fileList = os.listdir(dataPath + '/' + label)
    for wavFile in fileList:
        element = pickle.load(open(dataPath + "/" + label + "/" + wavFile,"rb"))
        if(element.shape[1] < min):
            min = element.shape[1]
        a.append(element)
        b.append(i)
    i = i + 1

print (min)

for i in range(len(a)):
    l = min * a[i].shape[0]
    a[i]=a[i].flatten()[:l]
    
clf = svm.SVC(kernel="rbf",verbose=1)

clf.fit(a,b)

print("finished training")

predictions = []

for wavFile in a:
    pred = clf.predict([wavFile])[0]
    print (pred)
    predictions.append(pred)

print(classification_report(b,predictions,target_names=os.listdir(dataPath)))

pickle.dump(clf,open(environ["MODEL_PATH"] + "/model","wb"))
pickle.dump(a,open(environ["MODEL_PATH"] + "/a","wb"))
pickle.dump(b,open(environ["MODEL_PATH"] + "/b","wb"))
pickle.dump(predictions,open(environ["MODEL_PATH"] + "/predictions","wb"))