dataset=read.csv("Position_Salaries.csv")                                   #importing the dataser
dataset=dataset[2:3]                                                        #keeping only the useful columns



library(rpart)                                                              #for decision tree regression
regressor=rpart(formula = Salary~.,
                data    = dataset,
                control = rpart.control(minsplit=1))    
print(summary(regressor))
 
y_pred=predict(regressor,data.frame(Level=6.5))                             #Predicting a new result
print(y_pred)


# Visualising the Decision Tree Regression results (higher resolution)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
ggsave('dec_tree_reg.pdf')
# Plotting the tree
plot(regressor)
text(regressor)