library(caret)
library(ggplot2)

# Load data
communities = read.csv("communities.csv", header = TRUE)
crime_rate <- communities[,101] # Crime Rate

# ---------------- Task 1: PCA with eigen() ---------------- 
# Scale and center the data with caret
scaler = preProcess(communities[ ,-101], method = c("center", "scale"))
communities_scaled <- predict(scaler, communities[,-101])

# Calculate the covariance matrix
cov <- cov(communities_scaled)

# Perform PCA with eigen()
pca_results_1 <- eigen(cov) # solving equation C x u = lambda * u
eigen_values <- pca_results_1$values

# Find position in vector where 95% of variance can be obtained
total_variance <- sum(eigen_values) # 100 after scaling
variance_proportion <- eigen_values/total_variance
cumulative_variance <- cumsum(eigen_values/total_variance)

# Amount of components needed for 95% of the variance
components_for_95_percent <- which(cumulative_variance >= 0.95)[1]

#The proportion of PC1
variance_proportion[1]

#The proportion of PC2
variance_proportion[2]



# ---------------- Task 2: PCA with princomp() ---------------- 
# Trace plot of loading scores
pca_results_2 = princomp(communities_scaled)
loading_scores_sorted <- sort(pca_results_2$loadings, decreasing=TRUE)
plot(pca_results_2$loadings[,1], main="Trace Plot of PC1 Loading scores", ylab="Loading Score")

# Trace plot of PC1 loading scores
sorted_pc1_loadings <- sort(pca_results_2$loadings[,1], decreasing = TRUE)
barplot(sorted_pc1_loadings, 
        main = "PC1 Feature Contributions (Loadings)",
        xlab = "Features",
        ylab = "PC1 Loading Score",
        las = 2,       # rotates x-axis labels vertically
        cex.names = 0.5)

# Trace plot of PC1 with abs(loading scores)
sorted_pc1_loadings = sort(abs(pca_results_2$loadings[, 1]), decreasing = TRUE)
barplot(sorted_pc1_loadings, 
        main = "PC1 Feature Contributions (Loadings)",
        xlab = "Features",
        ylab = "|PC1 Loading Score|",
        las = 2,
        cex.names = 0.5)

# set some values (loading, abs(loading), sort them etc)
pc1_loadings = pca_results_2$loadings[ ,1]
abs_pc1_loadings = abs(pc1_loadings)
sorted_pc1_loadings = sort(abs_pc1_loadings, decreasing = TRUE)

top_5_contributing_features = sorted_pc1_loadings[1:5]
print(top_5_contributing_features)
# Greatest contributing features are:
# medFamInc (Median Family Income)
# medIncome (Median Income)
# PctKids2Par (Percentage of Kids with Two Parents)
# pctWInvInc (Percentage with Investment Income)
# PctPopUnderPov (Percentage of Population Under Poverty)


# Create dataframe with PCA scores and crime rate for PC1 and PC2
scores_df <- data.frame(
  PC1 = pca_results_2$scores[,1],
  PC2 = pca_results_2$scores[,2],
  crime_rate = crime_rate
)

# Plot of the PC scores
ggplot(scores_df, aes(x=PC1, y=PC2, color=crime_rate))  +
  geom_point(alpha=0.7, size=3) + 
  scale_color_gradient(low="blue", high="red")+
  labs(title="PCA Scores Colored by Violent Crime Rate",
       x = "Principal Component 1 (PC1)",
       y = "Principal Component 2 (PC2)",
       color = "Violent Crimes per Pop") +
  theme_minimal()



# ---------------- Task 3: Linear regression ---------------- 
# Partition into 50% training data, 50% test data
n = dim(communities)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.5))
train = communities[id,]
test = communities[-id,] 

# Scale data with caret
model_scaler = preProcess(train, method = c("center", "scale"))
train_scaled <- predict(model_scaler, train) # wrong before
test_scaled <- predict(model_scaler, test)   # wrong before

# Fit linear regression model
model <- lm(ViolentCrimesPerPop~., train_scaled)
preds_train <- predict(model, train_scaled)
preds_test <- predict(model, test_scaled)

# Training and test MSE
MSE_train = mean((train_scaled$ViolentCrimesPerPop - preds_train)^2)
MSE_test = mean((test_scaled$ViolentCrimesPerPop - preds_test)^2)
MSE_train
MSE_test



# ---------------- Task 4: Cost function ---------------- 
# Prepare the data matrices
X_train <- as.matrix(train_scaled[, -which(names(train_scaled) == "ViolentCrimesPerPop")])
X_test <- as.matrix(test_scaled[, -which(names(test_scaled) == "ViolentCrimesPerPop")])
y_train <- train_scaled$ViolentCrimesPerPop
y_test <- test_scaled$ViolentCrimesPerPop

# Initialize storage vectors
train_errors <- c()
test_errors <- c()

# Cost function that tracks MSE at each iteration
cost_function <- function(theta) {
  
  # Training MSE (this is what BFGS minimizes)
  predictions_train <- X_train %*% theta
  mse_train <- mean((y_train - predictions_train)^2)
  
  # Test MSE
  predictions_test <- X_test %*% theta
  mse_test <- mean((y_test - predictions_test)^2)
  
  # Store both errors using global assignment
  train_errors <<- c(train_errors, mse_train)
  test_errors <<- c(test_errors, mse_test)
  
  # Return training error for optimization
  return(mse_train)
}

# Initialize theta to zero vector
theta_init <- rep(0, ncol(X_train))

# Run BFGS optimization
opt_result <- optim(
  par = theta_init,
  fn = cost_function,
  method = "BFGS"
)


# Find optimal iteration (minimum test error - early stopping criterion)
optimal_iteration <- which.min(test_errors)

cat("Total iterations:", length(train_errors), "\n")
cat("Optimal iteration (early stopping):", optimal_iteration, "\n")
cat("Training error at optimal iteration:", train_errors[optimal_iteration], "\n")
cat("Test error at optimal iteration:", test_errors[optimal_iteration], "\n")

# Plot MSE (discarding first 500 iterations as suggested)
start_iter <- 500
plot_data <- data.frame(
  iteration = start_iter:length(train_errors),
  train_error = train_errors[start_iter:length(train_errors)],
  test_error = test_errors[start_iter:length(test_errors)]
)

plot(plot_data$iteration, plot_data$train_error,
     type = "l", col = "black", lwd = 2,
     xlab = "Iteration Number",
     ylab = "Mean Squared Error",
     main = "Training and Test Errors vs Iteration Number",
     ylim = range(c(plot_data$train_error, plot_data$test_error)))

lines(plot_data$iteration, plot_data$test_error, col = "blue", lwd = 2)

# Add vertical line at optimal iteration
abline(v = optimal_iteration, col = "red", lty = 2, lwd = 2)

legend("topright", 
       legend = c("Training Error", "Test Error", "Optimal (Early Stopping)"),
       col = c("black", "blue", "red"),
       lty = c(1, 1, 2),
       lwd = 2)




# Compare with Task 3 results
cat("\n=== Comparison with Task 3 ===\n")
cat("Task 3 - Training MSE:", MSE_train, "\n")
cat("Task 3 - Test MSE:", MSE_test, "\n")
cat("Task 4 (optimal) - Training MSE:", train_errors[optimal_iteration], "\n")
cat("Task 4 (optimal) - Test MSE:", test_errors[optimal_iteration], "\n")


# Alternative visualization with ggplot2
plot_data_long <- data.frame(
  iteration = rep(start_iter:length(train_errors), 2),
  error = c(train_errors[start_iter:length(train_errors)],
            test_errors[start_iter:length(test_errors)]),
  type = rep(c("Training", "Test"), each = length(start_iter:length(train_errors)))
)

ggplot(plot_data_long, aes(x = iteration, y = error, color = type)) +
  geom_line(size = 1) +
  geom_vline(xintercept = optimal_iteration, linetype = "dashed", 
             color = "red", size = 1) +
  labs(title = "Training and Test Errors During BFGS Optimization",
       x = "Iteration Number",
       y = "Mean Squared Error",
       color = "Dataset") +
  theme_minimal() +
  annotate("text", x = optimal_iteration, y = max(plot_data_long$error) * 0.9,
           label = paste("Optimal:", optimal_iteration), 
           color = "red", hjust = -0.1)

