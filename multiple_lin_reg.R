#Importing dataset
dataset=read.csv("50_Startups.csv")

#Encoding the categorical data
dataset$State=factor(dataset$State, levels=c('Florida','New York','California'),labels=c(1,2,3))
library(caTools)
set.seed(123)
split=sample.split(dataset$Profit,SplitRatio=0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)
print("\n\nTraining set\n\n")
print(training_set)
print("Test set\n\n")
print(test_set)
#Fitting multiple Linear Regression model to dataset
regressor=lm(formula= Profit~ .,                                                        #profit~R.D.Spend+Administration+Marketing.Spend+State
             data=training_set)
print(summary(regressor))

#predicting test results
y_pred=predict(regressor,newdata=test_set)
print("y_pred")
print(y_pred)
print("Difference between orignal and predicted")
print(test_set$Profit-y_pred[1])
library("treemap")
tm<-treemap(dataset,
        index=c("R.D.Spend"),
        vSize="Profit",
        vColor="Marketing.Spend",
        draw=TRUE,
        type="value"
        )

print(tm)