install.packages('readxl')
install.packages('kknn')
library(readxl)
library(kknn)

#Each row in dataset is the image of a handwritten digit, each column represents the pixel intensity values and
#And the last column shows the actual digit 0-9.

#### Exxercise 1####
data = read.csv("/home/arre/Universitet/TDDE01/Lab1/optdigits.csv", header = FALSE)

colnames(data)[ncol(data)] <- "Digit" #give a label to last column

data[,ncol(data)] <- as.factor(data[,ncol(data)]) #makes the target variable categorical

n = dim(data)[1] #total number of instances
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train = data[id,]

id1 = setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.25))
valid = data[id2,]

id3 = setdiff(id1, id2)
test = data[id3,]

missclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}

#### Exercise 2####


fitted_traindata = kknn(Digit~., train, train, k=30, kernel = "rectangular")
fitted_testdata = kknn(Digit~., train, test, k=30, kernel = "rectangular")

pred_train <-fitted_traindata$fitted.values #the average label between the 30 nearest neighbours
pred_test <-fitted_testdata$fitted.values


#Each row predicted digit, each column actual digit
cm_test <- table(test$Digit, pred_test)
cm_train <- table(train$Digit, pred_train)
  

test_miss_rate <-missclass(pred_test, test$Digit)
train_miss_rate <-missclass(pred_train, train$Digit)

print(cm_test)
print(cm_train)
print(test_miss_rate)
print(train_miss_rate)

#End

#### Exercise 3####
train_probs_8 = fitted_traindata$prob[, "8"]
train_8_indices <-which(train$Digit == "8")

#vector where all elements correspond to one example where 8 is the true digit
train_probs_actual_8 <-train_probs_8[train_8_indices] 


easiest_8idx <- order(train_probs_actual_8, decreasing = TRUE)[1:2]
hardest_8idx <- order(train_probs_actual_8)[1:3]

easiest_rows <- train_8_indices[easiest_8idx]
hardest_rows <- train_8_indices[hardest_8idx]

#Case 1
pixels <- train[easiest_rows[1], -ncol(train)]
pixels_matrix<- matrix(as.numeric(pixels), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(pixels_matrix, Colv = NA, Rowv = NA, scale = "none")

#Case 25
pixels <- train[easiest_rows[2], -ncol(train)]
pixels_matrix<- matrix(as.numeric(pixels), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(pixels_matrix, Colv = NA, Rowv = NA, scale = "none")

#Case 7
pixels <- train[hardest_rows[1], -ncol(train)]
pixels_matrix<- matrix(as.numeric(pixels), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(pixels_matrix, Colv = NA, Rowv = NA, scale = "none")

#Case 30
pixels <- train[hardest_rows[2], -ncol(train)]
pixels_matrix<- matrix(as.numeric(pixels), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(pixels_matrix, Colv = NA, Rowv = NA, scale = "none")

#Case 72
pixels <- train[hardest_rows[3], -ncol(train)]
pixels_matrix<- matrix(as.numeric(pixels), nrow = 8, ncol = 8, byrow = TRUE)
heatmap(pixels_matrix, Colv = NA, Rowv = NA, scale = "none")

####Exercise 4####
error_rate_train <- numeric(30)
error_rate_valid <- numeric(30)
cross_entropy <- numeric(30)
for (i in 1:30){
  fit_train = kknn(Digit~., train, train, k=i, kernel = "rectangular", scale = TRUE)
  fit_valid = kknn(Digit~., train, valid, k=i, kernel = "rectangular", scale = TRUE)
  
  pred_train2 <- fit_train$fitted.values
  pred_valid <- fit_valid$fitted.values
  
  error_rate_train[i] <-missclass(pred_train2, train$Digit)
  error_rate_valid[i] <-missclass(pred_valid, valid$Digit)
  
  true_labels<-as.numeric(valid$Digit)
  n<-nrow(valid)
  
  y_true <- matrix(0, nrow = n, ncol = ncol(fit_valid$prob))
  for (k in 1:nrow(valid)) {
    y_true[k, true_labels[k]] <- 1 
  }
  
  cross_entropy[i] = -mean(rowSums(y_true*log(fit_valid$prob + 1e-15)))
  
}
best_k <- which.min(error_rate_valid)
fit_test = kknn(Digit~., train, test, k=best_k, kernel = "rectangular", scale = TRUE)
pred_test <-fit_test$fitted.values
error_test2 <-missclass(pred_test, test$Digit)

plot(1:30, error_rate_valid, type = "b", col="orange", main="Training error vs Validation error vs K", xlab="K", 
     ylab="Missclassification error", ylim = range(c(error_rate_train, error_rate_valid)))
lines(1:30, error_rate_train, type="b", col="blue")
points(best_k, error_test2, type = "b", col="red")
legend("bottomright", legend = c("Validation error", "Train error", "Test error"),
       col = c("orange", "blue", "red"), pch = 16, cex = 0.7)

plot(1:30, cross_entropy, type="b", col="green4", main="Cross-entropy vs K", xlab = "K", ylab = "Cross entropy")
optimal_k <- which.min(cross_entropy)
