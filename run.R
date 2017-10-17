# (c) 2015 by Guenter Klambauer
library(gmatrix)
setDevice(0)

## Simulated data example
y <- c(rep(0,49),rep(1,51))
X <- matrix(NA,nrow=100,ncol=20)
for (i in 1:100){
   X[i, ] <- rnorm(n=20,sd=0.2)
}
X[,5] <- X[,5]+y
X[,7] <- X[,7]+y
  
  
source("nnet-gpu.R")
nnm <- nnetgpu(X,y,hiddenUnits=c(50,20),epochs=5,fineTuneEpochs=0,verbose=0,eta=0.2,batch=16,activationF="ReLU")
  
# check training
table(y,predictNNgpu(X,nnm,useBestWeights=TRUE)>0.5)

## IRIS data example
Y <- cbind(as.numeric(rep("setosa",nrow(iris))==iris$Species), as.numeric(rep("versicolor",nrow(iris))==iris$Species),as.numeric(rep("virginica",nrow(iris))==iris$Species))
X <- as.matrix(iris[,1:4])
X <- t(t(X) - apply(X,2,mean) / apply(X,2,sd))

nnm <- nnetgpu(X,Y,hiddenUnits=c(100,100),epochs=5,fineTuneEpochs=0,verbose=0,eta=0.01,batch=64,activationF="ReLU",activationOutput="softmax",loss="cross-entropy")

# check training error
table(apply(Y,1,which.max),apply(predictNNgpu(X,nnm,useBestWeights=TRUE),1,which.max))


