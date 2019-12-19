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
################MAIN################################

################Importing dataset###################
dataset=pd.read_csv("Salary_Data.csv")
#print(dataset)                                                                 #for debugging purpose
X=dataset.iloc[:,:-1].values                                                    #-1 represents last col not included
#print(X)
Y=dataset.iloc[:,1].values                                                      #includes only the last col
#print(Y)

'''##############Replacing missing values###############
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)          #verbose is 0 is columns, 1 is rows
imputer=imputer.fit(X[:,1:3])                                                   #includes cols 1 and 2, excludes 3
X[:,1:3]=imputer.transform(X[:,1:3])                                            #replace the missing values with the calculated mean
#print(X)'''

'''############Processing string and categorical data########
labelencoder_obj=LabelEncoder()
X[:,0]=labelencoder_obj.fit_transform(X[:,0])                                   #encoding all countries into unique nos
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],    #https://datascience.stackexchange.com/questions/41113/deprecationwarning-the-categorical-features-keyword-is-deprecated-in-version
remainder='passthrough') 
X = np.array(transformer.fit_transform(X), dtype=np.int)                        #fitting the values to X
labelencoder_obj2=LabelEncoder()                                        
Y=labelencoder_obj2.fit_transform(Y)                                            #assigning numeric values to yes/no col
#print(X)
#print("\n\nY=")
#print(Y)                                            '''

############Splitting testing and training data########
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=42)



###########Feature scaling############################
'''sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print(X_train)'''                                                               #not required for this dataset

##########Linear Regression##########################
regressor=LinearRegression()                                                    #initializing the object with default constructor
regressor.fit(X_train,y_train)

########Prediction###############################
y_pred=regressor.predict(X_test)                                                #predicting values on test data

########Visualizing the training results###########
plt.scatter(X_train,y_train,color='red')                                        #plotting train data
plt.plot(X_train,regressor.predict(X_train),color='blue')                       #for plotting regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


######Visualizing the test results###############
plt.scatter(X_test,y_test,color='red')                                        #plotting train data
plt.plot(X_train,regressor.predict(X_train),color='blue')                       #for plotting regression line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()