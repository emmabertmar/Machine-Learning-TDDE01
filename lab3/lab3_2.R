# Using template:
# Lab 3 block 1 of 732A99/TDDE01/732A68 Machine Learning
# Author: jose.m.pena@liu.se
# Made for teaching purposes

library(kernlab)
set.seed(1234567890)

# Read data 
data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo, ]

# Split data into training, validation and test sets
train <- spam[1:3000, ] 
valid <- spam[3001:3800, ]
train_valid <- spam[1:3800, ]
test <- spam[3801:4601, ]

# Find best C parameter value
by <- 0.3
err_va <- NULL

for(i in seq(from = by, to = 5, by = by)){ 
  # Fit SVM with RBF kernel and sigma = 0.05
  filter <- ksvm(type ~ ., 
                 data = train, 
                 kernel = "rbfdot", 
                 kpar = list(sigma = 0.05),
                 C = i, 
                 scaled = FALSE)
  mailtype <- predict(filter, valid[,-58])
  
  # Error rate on validation set
  t <- table(mailtype, valid[,58])
  err_va <-c(err_va, (t[1,2]+t[2,1])/sum(t))
}

# Train on training set. Test on validation set.
filter0 <- ksvm(type ~ ., 
                data = train, 
                kernel = "rbfdot", 
                kpar = list(sigma = 0.05), 
                C = which.min(err_va) * by, 
                scaled = FALSE)
mailtype <- predict(filter0, valid[,-58])
t <- table(mailtype, valid[,58])
err0 <- (t[1,2]+t[2,1])/sum(t)
err0

# Train on training set. Test on testing set.
filter1 <- ksvm(type ~ .,
                data = train,
                kernel = "rbfdot",
                kpar = list(sigma = 0.05),
                C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter1, test[,-58])
t <- table(mailtype, test[,58])
err1 <- (t[1,2]+t[2,1])/sum(t)
err1

# Train on training and validation set. Test on test set.
filter2 <- ksvm(type ~ .,
                data = train_valid,
                kernel = "rbfdot",
                kpar = list(sigma = 0.05),
                C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter2, test[,-58])
t <- table(mailtype, test[,58])
err2 <- (t[1,2]+t[2,1])/sum(t)
err2

# Train on the entire data set. Test on test set.
filter3 <- ksvm(type ~ .,
                data = spam,
                kernel = "rbfdot",
                kpar = list(sigma = 0.05),
                C = which.min(err_va) * by,
                scaled = FALSE)
mailtype <- predict(filter3, test[,-58])
t <- table(mailtype, test[,58])
err3 <- (t[1,2]+t[2,1])/sum(t)
err3



# Implementation of SVM predictions
sv_indexes = alphaindex(filter3)[[1]] # Support vector indexes
coefficients = coef(filter3)[[1]]     # alpha * y
intercept = - b(filter3)              # Intercept (b() in kernlab returns negative intercept)
k = NULL
rbf_kernel = rbfdot(sigma = 0.05)


# We produce predictions for the first 10 points in the dataset
for(i in 1:10){ 
  k2 = NULL
  x_new = as.numeric(spam[i, -58])  # New data point
  
  for(j in 1:length(sv_indexes)){
    x_j = as.numeric(spam[sv_indexes[j], -58]) # Support vector on index j
    kernel_value = rbf_kernel(x_j, x_new)      # Compute K(x_j, x_new)  
    k2 = c(k2, coefficients[j] * kernel_value)
  }
  k = c(k, sum(k2) + intercept)
}

# Compare result with this prediction
answ = predict(filter3, spam[1:10, -58], type = "decision")
round(answ, digits = 7)
k




