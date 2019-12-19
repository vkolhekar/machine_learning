################IMPORT LIBRARY CLASSES#############
import numpy as np                                                              #for array computations
import pandas as pd  
import math                                                                     #to read and process data
from sklearn.impute import SimpleImputer                                        #to replace missing values with either mean or whatever the user's choice
from sklearn.preprocessing import LabelEncoder                                  #to encode string and categorical data into numeric values
from sklearn.preprocessing import OneHotEncoder                                 #to create dummy variables so that priorites are not assigned to string data
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split                            #to split dataset into train data and test data
from sklearn.preprocessing import StandardScaler
################MAIN################################

################Importing dataset###################
dataset=pd.read_csv("Data.csv")
#print(dataset)                                                                 #for debugging purpose
X=dataset.iloc[:,:-1].values                                                    #-1 represents last col not included
#print(X)
Y=dataset.iloc[:,3].values                                                      #includes only the last col
#print(Y)

##############Replacing missing values###############
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)          #verbose is 0 is columns, 1 is rows
imputer=imputer.fit(X[:,1:3])                                                   #includes cols 1 and 2, excludes 3
X[:,1:3]=imputer.transform(X[:,1:3])                                            #replace the missing values with the calculated mean
#print(X)

############Processing string and categorical data########
labelencoder_obj=LabelEncoder()
X[:,0]=labelencoder_obj.fit_transform(X[:,0])                                   #encoding all countries into unique nos
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],    #https://datascience.stackexchange.com/questions/41113/deprecationwarning-the-categorical-features-keyword-is-deprecated-in-version
remainder='passthrough') 
X = np.array(transformer.fit_transform(X), dtype=np.int)                        #fitting the values to X
labelencoder_obj2=LabelEncoder()                                        
Y=labelencoder_obj2.fit_transform(Y)                                            #assigning numeric values to yes/no col
#print(X)
#print("\n\nY=")
#print(Y)                                            

############Splitting testing and training data########
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

###########Feature scaling############################
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print(X_train)