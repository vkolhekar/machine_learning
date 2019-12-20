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
