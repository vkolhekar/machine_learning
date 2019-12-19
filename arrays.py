import numpy as np                                                              #for array computations
import pandas as pd  
import math                                                                     #to read and process data
import matplotlib.pyplot as plt                                                 #for plotting data and observations
from sklearn.impute import SimpleImputer                                        #to replace missing values with either mean or whatever the user's choice
from sklearn.preprocessing import LabelEncoder                                  #to encode string and categorical data into numeric values
from sklearn.preprocessing import OneHotEncoder                                 #to create dummy variables so that priorites are not assigned to string data
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split                            #to split dataset into train data and test data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression                               #we are using linear regression to predict salary of employees in this assignment
import statsmodels.regression.linear_model as sm                                #for Backwards Elimination optimization

def find_max(arr,arr_len):
    maxP=0
    for i in range(0,arr_len):
        if((arr[i]).astype(float)>maxP):
            maxP=(arr[i]).astype(float)
            #print("maxP at %d iteration is %f"%(i,maxP))
    #print("\n\nFinal maxP value is ",maxP)
    return maxP


dataset = pd.read_csv('50_Startups.csv')

X=dataset.iloc[:,:-1].values                                                    #-1 represents last col not included
#print(X)
y=dataset.iloc[:,4].values 

labelencoder_obj=LabelEncoder()
X[:,3]=labelencoder_obj.fit_transform(X[:,3])                                   #encoding all countries into unique nos
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],    #https://datascience.stackexchange.com/questions/41113/deprecationwarning-the-categorical-features-keyword-is-deprecated-in-version
remainder='passthrough') 
X = np.array(transformer.fit_transform(X), dtype=np.int)     

X=X[:,1:]

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
SL=0.05
maxP=0
rows,cols = X_opt.shape
#print(len(X_opt[0]))
for i in range(0,cols):
    #print("\n\n")
    regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
    p_len=len(regressor_OLS.pvalues)
    #print("pvalue array is = ",regressor_OLS.pvalues)
    #print("length of p value array is ",p_len)
    
    maxP=find_max(regressor_OLS.pvalues,p_len)
    #maxP = max(regressor_OLS.pvalues).astype(float)
    if(maxP>SL):
        #print("inside mxp>sl")
        for j in range(0,cols-i):
            #print("inside j loop")
            if (regressor_OLS.pvalues[j].astype(float) == maxP):
                #print("inside main condition")
                #print("print col to be deleted is ",X_opt[:,j])
                X_opt=np.delete(X_opt,j,1)
    #print("New X_opt is \n",X_opt)
print("\n\nfinal X_opt is \n",X_opt)
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
print(regressor_OLS.summary)