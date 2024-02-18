#!/usr/bin/env python
# coding: utf-8

# In[235]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# # Part 1

# In[236]:


def q2():
    
    X = pd.read_csv('features.csv')
    Y = pd.read_csv('labels.csv')
    
    return X.values,Y.values


# In[237]:


def q3(X):
    plt.imshow(X[:50],cmap='gray');
    return


# In[238]:


def q4(X,Y):
    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.2, random_state = 0)
    return xtrain,xtest,ytrain[:,0],ytest[:,0];


# # Part 2

# In[254]:


def q5(x,y,type_):
    if(type_ == "QDA"):
        clf = QDA() # builds an object for results

    elif(type_ == "LDA"):
        clf = LDA();
    
    elif(type_ == "GNB"):
        clf = GNB();
        
    clf.fit(x,y) # study data and it's parameters
    return clf.predict(x) # classify


# In[255]:


def q6(y,y_,string,type_):
    cm = confusion_matrix(y,y_)
    print("\n=================================");
    print("%s Type Classifier:"%(type_));
    print("=================================");
    print("Q6: %s Set classification accuracy = "%(string), np.diagonal(cm).sum()/sum(sum(cm)));
    print("\n%s Set Confusion Matrix\n\n"%(string),cm)
    print("\n%s Set full Classification Report:\n\n"%(string),classification_report(y,y_))
    return;


# In[256]:


def main():
    
    # for part 1:
    
    # q2
    X,Y = q2();
    
    # q3
    q3(X)
    
    # q4
    xtrain,xtest,ytrain,ytest = q4(X,Y);
    
    # for part 2:
    
    # q5
    type_ = "QDA";
    ytrain_ = q5(xtrain,ytrain,type_);
    ytest_ = q5(xtest,ytest,type_);
    
    # q6
    q6(ytrain,ytrain_,"Train",type_);
    q6(ytest,ytest_,"Test",type_);

    # q7
    type_ = "LDA";
    ytrain_ = q5(xtrain,ytrain,type_);
    ytest_ = q5(xtest,ytest,type_);
    q6(ytrain,ytrain_,"Train",type_);
    q6(ytest,ytest_,"Test",type_);
    
    # q8
    type_ = "GNB";
    ytrain_ = q5(xtrain,ytrain,type_);
    ytest_ = q5(xtest,ytest,type_);
    q6(ytrain,ytrain_,"Train",type_);
    q6(ytest,ytest_,"Test",type_);
    
    return


# In[257]:


main()


# In[ ]:





# In[ ]:





# In[ ]:




