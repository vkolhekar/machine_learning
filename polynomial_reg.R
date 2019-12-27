dataset=read.csv("Position_Salaries.csv")                                   #importing the dataser
dataset=dataset[2:3]                                                        #keeping only the useful columns

lin_reg=lm(formula = Salary ~.,                                             #linear regression for base comparison
            data=dataset)
print(summary(lin_reg))

dataset$Level2 = dataset$Level^2                                            #adding level^2 column as Level2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
print(dataset)
poly_reg = lm(formula = Salary ~.,
                data = dataset)
print(summary(poly_reg))

#Visualizing Linear Regression Results
library(ggplot2)
ggplot()+
    geom_point(aes(x=dataset$Level,y=dataset$Salary,),colour='red')+
    geom_line(aes(x=dataset$Level,y=predict(lin_reg,newdata=dataset),colour='blue'))+
    ggtitle('Truth or Bluff (Linear Regression)')+
    xlab('Level')+
    ylab('Salary')
ggsave('lin_reg.pdf')

#Visualizing Polynomial Regression Results
library(ggplot2)
ggplot()+
    geom_point(aes(x=dataset$Level,y=dataset$Salary,),colour='red')+
    geom_line(aes(x=dataset$Level,y=predict(poly_reg,newdata=dataset),colour='blue'))+
    ggtitle('Truth or Bluff (Polynomial Regression)')+
    xlab('Level')+
    ylab('Salary')
ggsave('poly_reg.pdf')