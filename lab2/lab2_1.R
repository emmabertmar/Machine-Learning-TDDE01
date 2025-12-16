
library(dplyr)
library(glmnet)

# Load data
dataframe = read.csv("tecator.csv")

# Select data that is to be used
data = dataframe %>%
  select(Fat, Channel1:Channel100)

# Partition into 50% training data and 50% test data
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.5))
train_df = data[id,]
test_df = data[-id,]


# ---------------- Task 1: Linear regression ---------------- 
# Fit a linear regression to the training data 
model = lm(Fat ~ ., data = train_df)
summary(model)

# Estimate the training and test errors
# Get predictions for training data and calculate MSE
predictions_train = predict(model, train_df)
MSE_train = mean((train_df$Fat - predictions_train)^2) 

# Get predictions for test data and calculate MSE
predictions_test  = predict(model, test_df)
MSE_test = mean((test_df$Fat - predictions_test)^2)

print(paste("traning MSE = ", MSE_train))
print(paste("test MSE = ", MSE_test))


# ---------------- Task 3: Lasso regression ---------------- 
# Prepare training data, remove unnecessary columns
x = as.matrix(train_df %>% select(-Fat))
y = as.matrix(train_df %>% select(Fat))

# Fit a Lasso regression model with alpha = 1
model_lasso = glmnet(x, y, alpha = 1, family = 'gaussian')
model_lasso

# Plot the coefficients for Lasso regression
plot(model_lasso,
     xvar = "lambda", 
     label = TRUE,
     main = "",
     ylab = "Coefficients")

title(main = "Lasso Regression",
      line = 2.5)



# ---------------- Task 4: Ridge regression ---------------- 
# Fit a Ridge regression model with alpha = 0
model_ridge = glmnet(x, y, alpha = 0, family = 'gaussian')
model_ridge

# Plot the coefficients for Ridge regression
plot(model_ridge, 
     xvar = "lambda", 
     label = TRUE, 
     main = "", 
     ylab = "Coefficients")

title(main = "Ridge Regression", 
      line = 2.5)


# ---------------- Task 5: Cross-Validation ---------------- 
# Find optimal lambda with cross-validation. 10 folds by default.
cv_lasso = cv.glmnet(x, y, alpha = 1, family = 'gaussian')
  
# Optimal lambda and number of variables
coef(cv_lasso, s = "lambda.min")
cv_lasso$lambda.min

# Plot the cross-validation for Lasso regression
plot(cv_lasso)

title(main = "Lasso Regression cross-validation", 
      line = 2.5)


# Prepare test data, remove unnecessary columns
x_test = as.matrix(test_df %>% select(-Fat))
y_test = as.matrix(test_df %>% select(Fat))

# Make predictions for minimum lambda value
predicted_test = predict(cv_lasso, newx = x_test, s = "lambda.min", type = "response")

plot(x = as.vector(y_test),
     y = as.vector(predicted_test), 
     xlab = "Actual test values",
     ylab = "Predicted test values",
     xlim = c(min(y_test), max(y_test)),
     ylim = c(min(predicted_test), max(predicted_test)),
     col = "coral1",
     pch = 16,  # filled circles
     cex = 1.2,  # size of dots
     main = "Actual test vs Predicted test")

abline(a = 0, b= 1, col = "black", lty = 2)

