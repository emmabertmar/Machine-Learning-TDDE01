
# Read csv file and add column names
dataframe = read.csv("pima-indians-diabetes.csv", header = FALSE)
colnames(dataframe) <- c("times_pregnant", "glucose_2h", "blood_pressure", 
                  "skinfold_thickness", "serum_insulin", "mass_index", 
                  "pedigree", "age", "diabetes")



# ---------------- Task 1: Scatterplot ---------------- 
# Plot of Age and Plasma Glucose, colored by diabetes status
plot(x = dataframe$age, y = dataframe$glucose_2h,
     xlab = "Age",
     ylab = "Plasma Glucose Concentration",
     xlim = c(min(dataframe$age), max(dataframe$age)),
     ylim = c(min(dataframe$glucose_2h), max(dataframe$glucose_2h)),
     col = ifelse(dataframe$diabetes == 0, "dodgerblue3", "coral1"),
     pch = 16,  # filled circles
     cex = 0.7,  # size of dots
     main = "Plasma Glucose vs Age")

legend("bottomright", legend = c("Diabetes", "Not diabetes"),
       col = c("coral1", "dodgerblue3"), pch = 16, cex = 0.7)

  

# ---------------- Task 2: Logistic regression model ----------------
# Fit a logistic regression model
model = glm(diabetes ~ glucose_2h + age,
            data = dataframe, family = "binomial")
summary(model)

# Predict probabilities from the model and classify using threshold r = 0.5
probability = predict(model, type = "response")
prediction = ifelse(probability > 0.5, 1, 0)

# Confusion matrix
confusion_matrix <- table(dataframe$diabetes, prediction)
print(confusion_matrix)

# Misclassification rate: Proportion of incorrectly classified observations
incorrect <- sum(confusion_matrix) - sum(diag(confusion_matrix))
misclassification <- incorrect/sum(confusion_matrix)
print(paste("Misclassification rate = ", misclassification))
  
# Scatterplot: Classifications from model
plot(x = dataframe$age, y = dataframe$glucose_2h,
     xlab = "Age",
     ylab = "Plasma Glucose Concentration",
     xlim = c(min(dataframe$age), max(dataframe$age)),
     ylim = c(min(dataframe$glucose_2h), max(dataframe$glucose_2h)),
     col = ifelse(prediction == 0, "dodgerblue3", "coral1"),
     pch = 16,  # filled circles
     cex = 0.7,  # size of dots
     main = "Plasma Glucose vs Age")

legend("bottomright", legend = c("Predicted diabetes", "Predicted not diabetes"),
       col = c("coral1", "dodgerblue3"), pch = 16, cex = 0.7)



# ---------------- Task 3: Decision boundary ---------------- 
# Add decision boundary line from logistic regression coefficients
abline(a = -coef(model)[1]/coef(model)[2],  # intercept
       b = -coef(model)[3]/coef(model)[2],  # slope
       col = "black", lwd = 1.5)


# ---------------- Task 4: Scatterplots with basis thresholds ---------------- 
# Predict using threshold r = 0.2
prediction2 = ifelse(probability > 0.2, 1, 0)

# Confusion matrix
confusion_matrix2 <- table(dataframe$diabetes, prediction2)
print(confusion_matrix2)

# Misclassification rate
incorrect2 <- sum(confusion_matrix2) - sum(diag(confusion_matrix2))
misclassification2 <- incorrect2/sum(confusion_matrix2)
print(paste("Misclassification rate 2 = ", misclassification2))

# Scatterplot
plot(x = dataframe$age, y = dataframe$glucose_2h,
     xlab = "Age",
     ylab = "Plasma Glucose Concentration",
     xlim = c(min(dataframe$age), max(dataframe$age)),
     ylim = c(min(dataframe$glucose_2h), max(dataframe$glucose_2h)),
     col = ifelse(prediction2 == 0, "dodgerblue3", "coral1"),
     pch = 16,  # filled circles
     cex = 0.7,  # size of dots
     main = "Plasma Glucose vs Age"
)

legend("bottomright", legend = c("Predicted diabetes", "Predicted not diabetes"),
       col = c("coral1", "dodgerblue3"), pch = 16, cex = 0.7)



# Predict using threshold r = 0.8 
prediction3 = ifelse(probability > 0.8, 1, 0)

# Confusion matrix
confusion_matrix3 <- table(dataframe$diabetes, prediction3)
print(confusion_matrix3)

# Misclassification rate
incorrect3 <- sum(confusion_matrix3) - sum(diag(confusion_matrix3))
misclassification3 <- incorrect3/sum(confusion_matrix3)
print(paste("Misclassification rate 3 = ", misclassification3))

# Scatterplot
plot(x = dataframe$age, y = dataframe$glucose_2h,
     xlab = "Age",
     ylab = "Plasma Glucose Concentration",
     xlim = c(min(dataframe$age), max(dataframe$age)),
     ylim = c(min(dataframe$glucose_2h), max(dataframe$glucose_2h)),
     col = ifelse(prediction3 == 0, "dodgerblue3", "coral1"),
     pch = 16,  # filled circles
     cex = 0.7,  # size of dots
     main = "Plasma Glucose vs Age"
)

legend("bottomright", legend = c("Predicted diabetes", "Predicted not diabetes"),
       col = c("coral1", "dodgerblue3"), pch = 16, cex = 0.7)



# ---------------- Task 5: Basis function expansion ---------------- 
# Add basis features to dataset
dataframe$z1 = dataframe$glucose_2h^4
dataframe$z2 = dataframe$glucose_2h^3 * dataframe$age
dataframe$z3 = dataframe$glucose_2h^2 * dataframe$age^2
dataframe$z4 = dataframe$glucose_2h^4 * dataframe$age^3
dataframe$z5 = dataframe$age^4

# Fit logistic regression with basis-expanded features
model2 = glm(diabetes ~ glucose_2h + age + z1 + z2 + z3 + z4 + z5,
            data = dataframe, family = "binomial")
summary(model2)

# Calculate all probabilities for each person (with their x1, x2)
probability_basis = predict(model2, type = "response")
prediction_basis = ifelse(probability_basis > 0.5, 1, 0)

# Confusion matrix
confusion_matrix_basis <- table(dataframe$diabetes, prediction_basis)
print(confusion_matrix_basis)

# Misclassification rate
incorrect_basis <- sum(confusion_matrix_basis) - sum(diag(confusion_matrix_basis))
misclassification_basis <- incorrect_basis/sum(confusion_matrix_basis)
print(paste("Misclassification rate = ", misclassification_basis))

# Scatterplot
plot(x = dataframe$age, y = dataframe$glucose_2h,
     xlab = "Age",
     ylab = "Plasma Glucose Concentration",
     xlim = c(min(dataframe$age), max(dataframe$age)),
     ylim = c(min(dataframe$glucose_2h), max(dataframe$glucose_2h)),
     col = ifelse(prediction_basis == 0, "dodgerblue3", "coral1"),
     pch = 16,  # filled circles
     cex = 0.7,  # size of dots
     main = "Plasma Glucose vs Age"
)

legend("bottomright", legend = c("Predicted diabetes", "Predicted not diabetes"),
       col = c("coral1", "dodgerblue3"), pch = 16, cex = 0.7)


