install.packages('readxl')
install.packages('kknn')
install.packages('neuralnet')

library(readxl)
library(kknn)
library(neuralnet)

######Exercise 1###########
set.seed(1234567890)
Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin=sin(Var)) #calculates sin for each data point. 
tr <- mydata[1:25,] # Training
te <- mydata[26:500,] # Test
# Random initialization of the weights in the interval [-1, 1]
set.seed(1234567890)
winit <- runif(31, -1, 1)
nn <- neuralnet(Sin ~ Var, data=tr, hidden=10, startweights = winit, act.fct = "logistic")
# Plot of the training data (black), test data (blue), and predictions (red)
print(nn$weights)
plot(tr, cex=2, col="black")
points(te, col = "blue", cex=1)
points(te[,1],predict(nn,te), col="red", cex=1)
legend("bottomright", legend = c("Train points", "Test points", "Nerual network predictions"), 
       col=c("black", "blue", "red"), pch=16)


#####Exercise 2#########
linear <- function(x) x
ReLU <- function(x) ifelse(x > 0, x, 0.01 * x)
softplus <- function(x) log(1+exp(x))

nn_linear <- neuralnet(Sin ~ Var, data=tr, hidden=10, startweights = winit, act.fct = linear)
nn_ReLU <- neuralnet(Sin ~ Var, data=tr, hidden=10, startweights = winit, act.fct = ReLU)
nn_softplus <- neuralnet(Sin ~ Var, data=tr, hidden=10, startweights = winit, act.fct = softplus)

plot(te, cex=1, col="black", ylim= c(-1.5, 1.5), main="Linear vs ReLU vs softplus")
points(te[,1], predict(nn_linear, te), col="blue", cex=1)
points(te[,1], predict(nn_ReLU, te), col="red", cex=1)
points(te[,1], predict(nn_softplus, te), col="green", cex=1)
legend("bottomright", legend = c("Test points", "Linear", "ReLU", "softplus"),
       col=c("black", "blue", "red", "green"), pch=16)


######Exercise 3###########
set.seed(1234567890)
Var <- runif(500, 0, 50)
mydata <- data.frame(Var, Sin=sin(Var)) #calculates sin for each data point. 

set.seed(1234567890)
winit <- runif(31, -1, 1)
plot(mydata, cex=1, col="black", main="Prediction on new range [0, 50]", ylim=c(-15, 10))
points(mydata[,1],predict(nn,mydata), col="red", cex=1)
abline(h=-7.5, col="green", lwd=2)
legend("bottomright", legend = c("True points", "NN predictions"), col=c("black", "red"), 
       , pch=16)

########Exercise 4#######
nn$result.matrix
nn$weights


#########Exercise 5#######
set.seed(1234567890)
Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin=sin(Var)) #calculates sin for each data point. 

set.seed(1234567890)
winit <- runif(31, -1, 1)

#before we used Sin~Var, for inverse we use Var~Sin (predict x from sin)
nn_inverse <- neuralnet(Var ~ Sin, data=mydata, hidden=10, startweights = winit, threshold = 0.1)

plot(mydata$Sin,mydata$Var, cex=1, col= "black", main = "Nn predicting x from sin(x)",
     ylim=c(-1, 10), xlim = c(-1.0, 5))
points(mydata[,1], predict(nn_inverse, mydata), col="red", cex=1)
legend("bottomright", legend = c("True points", "Predicted"), col = c("black", "red"), pch=16)

