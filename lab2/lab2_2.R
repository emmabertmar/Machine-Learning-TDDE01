install.packages('readxl')
install.packages('kknn')
install.packages('tree')

library(readxl)
library(kknn)
library(tree)

#Each row in dataset is the image of a handwritten digit, each column represents the pixel intensity values and
#And the last column shows the actual digit 0-9.

#### Exxercise 1####
data = read.csv2("/home/arre/Universitet/TDDE01/Lab2/bank-full.csv", header = TRUE, stringsAsFactors = TRUE)

data$duration = c() #remove duration variable

colnames(data)[ncol(data)] <- "output" #give a label to last column, this is the target variable


n = dim(data)[1] #total number of instances
set.seed(12345)
id=sample(1:n, floor(n*0.4))
train = data[id,] #train 40%

id1 = setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.3))
valid = data[id2,] #valid 30%

id3 = setdiff(id1, id2)
test = data[id3,] #test 40%


####Excercise 2 #####
missclass=function(X,X1){ 
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}

#Train the models with the different settings
tree1 = tree(output~., data = train)
tree2 = tree(output~., data=train, minsize=7000)
tree3 = tree(output~., data=train, mindev=0.0005)

#Predict training data points
pred1_train <- predict(tree1, type="class")
pred2_train <- predict(tree2, type="class")
pred3_train <- predict(tree3, type="class")

#Predict validation data points
pred1_valid <- predict(tree1, newdata = valid, type="class")
pred2_valid <- predict(tree2, newdata = valid, type="class")
pred3_valid <- predict(tree3, newdata = valid, type="class")

#Calculate missclassification rate for default
mc1_train <- missclass(pred1_train, train$output) # = 0.1048441
mc1_valid <- missclass(pred1_valid, valid$output) # = 0.1092679

#Calculate missclassification rate for minsize=7000
mc2_train <- missclass(pred2_train, train$output) #=0.1048441
mc2_valid <- missclass(pred2_valid, valid$output) #=0.1092679

#Calculate missclassification rate for mindev=0.0005
mc3_train <- missclass(pred3_train, train$output) #0.09400575
mc3_valid <- missclass(pred3_valid, valid$output) #0.1119221


####Exercise 3#####

#code from lecture 2b, get deviance for different k values
deviance_train=rep(0, 50)
deviance_valid=rep(0, 50)
for(k in 2:50){
#prune (remove non critical parts)
 pruned_tree = prune.tree(tree3, best=k)
 pred=predict(pruned_tree, newdata = valid, type="tree")
 deviance_train[k] = deviance(pruned_tree)
 deviance_valid[k] = deviance(pred)
}


#Plot the deviance for training and validation data
plot(2:50, deviance_train[2:50], type="b", col="orange",
     xlab="Number of leaves", ylab="Deviance",
     main="Training vs Validation Deviance", ylim = range(7000, 12000))
lines(2:50, deviance_valid[2:50], type="b", col="blue")
legend("topright", legend=c("Training", "Validation"),
       col=c("orange", "blue"), pch=16)

optimal_leaves = which.min(deviance_valid[-1]) + 1 #leaves, gives 21+1=22 optimal leaves because of index 2:50 in loop.

#Prune tree with optimal amount of leaves, plot tree aswell
optimal_tree=prune.tree(tree3, best = optimal_leaves)
plot(optimal_tree)
text(optimal_tree, pretty=0)
summary(optimal_tree)
#missclass rate 0.1039

######Exercise 4 ########
#train the optimal tree
predicted_tree = predict(optimal_tree, newdata = test, type = "class")
confusion_matrix = table(test$output, predicted_tree)

accuracy=1-missclass(predicted_tree, test$output) #accuracy = 0.8910351

precision = confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[1,2]) #TP/(TP+FP)= 0.66667
recall = confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[2,1]) #TP/(TP+FN) = 0.135

F1 = 2* (precision*recall)/(precision+recall) #F1 = 0,224554 quite low => bad at detecting the positive class


  #####Exercise 5 ####### to be finished
#Loss matrix needs to be swapped from the one given in assigment correct order no, yes
LossMatrix = matrix(c(0, 1, 5, 0), nrow = 2, ncol = 2, byrow = TRUE)
colnames(LossMatrix) = rownames(LossMatrix) = levels(test$output)

# multiply with loss matrix to weigh the results as we need to
testPrediction = predict(optimal_tree, newdata = test) %*% LossMatrix

# Code from tutorial 2
testPredictionI = apply(testPrediction, MARGIN=1, FUN = which.min)
Pred = levels(test$output)[testPredictionI]

confusion_matrix = table(test$output, Pred)

accuracy=1-missclass(Pred, test$output) #accuracy = 0.8910351

precision = confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[1,2])
recall = confusion_matrix[2,2]/(confusion_matrix[2,2] + confusion_matrix[2,1])

F1 = 2* (precision*recall)/(precision+recall) #F1 = 0,486205 increased => model is better at detecting the positive class. 

  
  ##### Execise 6 ######
  
optimal_dt_predict = predict(optimal_tree, newdata = test, type="vector")

#Train a logistic model and predict
logistic_regression = glm(output~., train, family = "binomial")
prob_logistic_regression = predict(logistic_regression, newdata=test, type="response")

#true positive and false positive rate for decision tree
TPR_DT = numeric(20)
FPR_DT = numeric(20)

#true positive and false positive rate for logistic regression model
TPR_LR = numeric(20)
FPR_LR = numeric(20)

index = 1

for (i in seq(0.05, 0.95, 0.05)) {
  #Decision tree
  #y_hat yes if p(Y='yes'|X)>pi otherwise no
  y_hat = ifelse(optimal_dt_predict[,2] > i, "yes", "no")
  #Confusion matrix for decision tree
  cm_DT = table(
    factor(test$output, levels = c("no", "yes")),
    factor(y_hat, levels = c("no", "yes"))
  )
  
  #Calculate false and true positive rates
  FPR_DT[index] = cm_DT[1,2]/(cm_DT[1,2]+ cm_DT[1, 1]) #FP/(FP+TN)
  TPR_DT[index] = cm_DT[2,2]/(cm_DT[2,2]+ cm_DT[2, 1]) #TP/(TP+FN)
  
  
  #Logistic regression
  #yes if p(Y='yes'|X)>pi otherwise no
  pred_LR = ifelse(prob_logistic_regression > i, "yes", "no")
  
  #Confusion matrix for logistic regression
  cm_LR = table(
    factor(test$output, levels=c("no","yes")),
    factor(pred_LR, levels=c("no","yes"))
  )
  
  #Calculate false and true positive rates
  FPR_LR[index] = cm_LR[1,2]/(cm_LR[1,2]+ cm_LR[1, 1]) #FP/(FP+)
  TPR_LR[index] = cm_LR[2,2]/(cm_LR[2,2]+ cm_LR[2, 1]) #TP/(TP+FN)
  
  index = index + 1

}

# Plot Decision Tree
plot(FPR_DT, TPR_DT, type="l", col="red", lwd=2,
     xlab="False Positive Rate", ylab="True Positive Rate",
     xlim=c(0,1), ylim=c(0,1),
     main="ROC Curves")

# Add Logistic Regression
lines(FPR_LR, TPR_LR, col="blue", lwd=2)


# Legend
legend("bottomright", legend=c("Decision Tree","Logistic Regression"),
       col=c("red","blue"), lty=1, pch=c(16,17))




  

