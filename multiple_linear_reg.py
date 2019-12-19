################IMPORT LIBRARY CLASSES#############
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

################USER DEFINED FUNCTIONS#############

def preprocessing(X):

    '''##############Replacing missing values###############
    imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)          #verbose is 0 is columns, 1 is rows
    imputer=imputer.fit(X[:,1:3])                                                   #includes cols 1 and 2, excludes 3
    X[:,1:3]=imputer.transform(X[:,1:3])                                            #replace the missing values with the calculated mean
    #print(X)'''

    ############Processing string and categorical data########
    labelencoder_obj=LabelEncoder()
    X[:,3]=labelencoder_obj.fit_transform(X[:,3])                                   #encoding all countries into unique nos
    transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [3])],    #https://datascience.stackexchange.com/questions/41113/deprecationwarning-the-categorical-features-keyword-is-deprecated-in-version
    remainder='passthrough') 
    X = np.array(transformer.fit_transform(X), dtype=np.int)                        #fitting the values to X
    '''labelencoder_obj2=LabelEncoder()                                        
    Y=labelencoder_obj2.fit_transform(Y)                                            #assigning numeric values to yes/no col

    #print("\n\nY=")
    #print(Y)               '''                             
    return X

###############FINDING MAX P VALUE################
def find_max(arr,arr_len):
    maxP=0
    for i in range(0,arr_len):
        if((arr[i]).astype(float)>maxP):
            maxP=(arr[i]).astype(float)
            #print("maxP at %d iteration is %f"%(i,maxP))
    #print("\n\nFinal maxP value is ",maxP)
    return maxP

################BACKWARDS ELIMINATION###############
def backwards_elimination(X,y,sl):
    #print("\n\n\n")
    X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)                    #adding x0=1 for coeff b0 in the multiple linear regression equation
    #X_opt=X[:,[0,3]]                                                                #starting with all independent variables, eliminating ivs one by one 
    X_opt=X
    SL=0.05
    maxP=0
    rows,cols = X_opt.shape
    #print("\t\tX_opt\n\n\n",X_opt)
                             
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
    #print("\n\nfinal X_opt is \n",X_opt)
    regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()    
    return regressor_OLS
################MAIN################################
def main():

    ################Importing dataset###################
    dataset=pd.read_csv("50_Startups.csv")
    #print(dataset)                                                                 #for debugging purpose
    X=dataset.iloc[:,:-1].values                                                    #-1 represents last col not included
    #print(X)
    y=dataset.iloc[:,4].values                                                      #includes only the last col
    #print(Y)

    X=preprocessing(X)                                                              #required data preprocessing

    ############Avoiding dummy variablr trap###############
    X=X[:,1:]
    #print(X)
    ############Splitting testing and training data########
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=42)


    regressor=LinearRegression()                                                    #constructor call
    regressor.fit(X_train,y_train)
    y_pred=regressor.predict(X_test)                                                #predicting the test data

    ###########Building Optimal Model using Backwards Elimination#############
    regressor_OLS=backwards_elimination(X,y,0.05)
    print(regressor_OLS.summary())




main()