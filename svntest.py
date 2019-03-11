import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataread
from sklearn import svm

def show_accuracy(x,y,tit):
    count=x.shape[0]
    plt.scatter(range(count), x)
    plt.scatter(range(count),y)
    plt.title(tit)
    plt.show()


inputlist = dataread.input('learn')
outputlist = dataread.output('learn')

# test set
inputtest = dataread.input('validate')
outputtest = dataread.output('validate')

#clf = svm.SVC(C=10, kernel='linear', decision_function_shape='ovr')
clf = svm.SVC(C=5, kernel='rbf', gamma=1, decision_function_shape='ovo')
clf.fit(inputlist, outputlist)

print(clf.score(inputlist, outputlist))
y_hat = clf.predict(inputlist)
#show_accuracy(y_hat, outputlist, 'train')

print(clf.score(inputtest, outputtest))
y_hat = clf.predict(inputtest)
#show_accuracy(y_hat, outputtest, 'test')

print(np.array(outputtest))
print(np.array(y_hat))