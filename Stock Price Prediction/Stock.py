
# coding: utf-8

# In[2]:


import sys
#print('Python: {}'.format(sys.version))
# scipy
import scipy
#print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
#print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
#print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
#print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier




# In[1]:


# Load dataset
url = "ABBAN.csv"
dataset = pandas.read_csv(url, header=None)
array = dataset.values
dataset.head(20)

array.reshape(-1, 1)
X = array[:,0]
Y = array[:,1]

validation_size = 0.50
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

#40.9	321650
names = []
model2 = LinearRegression();
model2.fit(numpy.array(X_train).reshape(-1,1),numpy.array(Y_train).reshape(-1,1))

out=model2.predict(numpy.array(X_validation).reshape(-1,1))
a=0
for i in range(0,len(X_validation)):
    a=a+((out[i]-Y_validation[i])/Y_validation[i])**2
a=a/len(X_validation)
a=a**(1/2.0)
print((1-a)*100,"% accuracy")
for i in range(0,len(X_validation)):
    print(Y_validation[i],out[i])
 #





