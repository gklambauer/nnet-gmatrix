# (c) 2015 by Guenter Klambauer
# R package gmatrix needed
# adapted version used for ChemBioBridge data

forwardPassGPU <- function(X,W,dropout,activationF,activationFOutput){
  s <- nrow(X)
  L <- length(W)
  act <- g(X,type="s")
  inputUnits <- ncol(X)
  outputUnits <- ncol(W[[1]])
  for (l in 1:L){
    #other layers	
    if (l < L){
      outputUnits <- ncol(W[[l]])
      netI <- gmatrix(0,nrow=s,ncol=outputUnits,type="s")
      gmm(act,W[[l]],netI)
      if (dropout[l]>0) {
        dropMask <- gmatrix(grbinom(n=outputUnits*s,size=1,prob=1-dropout[l+1]), nrow=s, type="s")
        act <- activationF(netI) * dropMask
      } else {
        act <- activationF(netI)
      }
      inputUnits <- outputUnits
    } else {
      outputUnits <- ncol(W[[l]])
      netI <- gmatrix(0,nrow=s,ncol=outputUnits,type="s")
      gmm(act,W[[l]],netI)
      if (dropout[l]>0) {
        dropMask <- gmatrix(grbinom(n=outputUnits*s,size=1,prob=1-dropout[l+1]), nrow=s, type="s")
        act <- activationFOutput(netI) * dropMask
      } else {
        act <- activationFOutput(netI)
      }
      inputUnits <- outputUnits
    }
  }	
  
  # output layers
  return(h(act))
}



computeError <- function(targets,output,lossf,learnType){
  nc <- ncol(targets)
  output[is.na(output)] <- mean(output,na.rm=TRUE)
  errorMat <- lossf(output,targets)
  errorMat <- pmax(errorMat,0)
  currentError <- mean(errorMat,na.rm=TRUE)
  cM <- colMeans(errorMat,na.rm=TRUE)
  cM[is.na(cM)] <- 0
  errorVec <- cM
  errorVecFN <- rep(0,nc)
  errorVecFP <- rep(0,nc)
  errorMatC <- 0
  currentErrorC <- 0
  a1 <- c(0)
  a2 <- c(0)
  errV <- list()
  errV$errorMat <- errorMat
  errV$currentError <- currentError
  errV$errorVec <- errorVec
  errV$errorMatC <- errorMatC
  errV$currentErrorC <- currentErrorC
  errV$errorVecFN <- errorVecFN
  errV$errorVecFP <- errorVecFP
  errV$a1 <- a1
  errV$a2 <- a2
  return(errV)
  
}

if (FALSE){
  Y <- matrix(c(rep(0,49),rep(1,51)),ncol=1)
  X <- matrix(NA,nrow=100,ncol=20)
  for (i in 1:100){
    X[i, ] <- rnorm(n=20,sd=0.2)
  }
  X[,5] <- X[,5]+Y
  X[,7] <- X[,7]+Y
  Xval=NULL
  Yval=NULL
  eta=0.1
  mu=0.9
  gamma=0.01
  tau=0.95
  activationFunction="sigmoid"
  activationOutput="sigmoid"
  loss="logistic"
  learnType="classification"
  hiddenUnits=c(100,100)
  epochs=100
  fineTuneEpochs=10
  batch=128
  verbose=FALSE
  dropout=c(0,0,0)
  replace=FALSE
  pp1=2.0
  pp2=16
  ss=1.2
}


nnetgpu <- function(X,Y,Xval=NULL,Yval=NULL,eta=0.1,W=NULL, activationFunction="sigmoid",activationOutput="sigmoid", 
                    loss="logistic",learnType="classification",init="MS",detNorm=FALSE, saveWeights=NULL,saveWeightsEvery=10,
                    hiddenUnits=c(100,100),epochs=100,fineTuneEpochs=10,batch=128,verbose=10,dropout=c(0,0,0),
                    replace=FALSE,pp1=2.0,pp2=16,monitoring=FALSE,validationErrorMode="subset"){
  
  library(gmatrix)
  
  # batch size
  if (!(is.numeric(batch) & batch >=1 & batch <= nrow(X)) ) stop("Batch size incorrect.")
  batch <- as.integer(batch)
  
  if (is.data.frame(X)) X <- as.matrix(X)
  # nbr of samples
  n <- nrow(X)
  # nbr of features
  m <- ncol(X)
  # nbr of weight matrices
  L <- length(hiddenUnits)+2
  
  # target values
  if (is.vector(Y)) Y <- matrix(Y,ncol=1)
  if (nrow(X)!=nrow(Y)) stop("Number of labels does not match number of samples.")
  
  if (any(is.na(X))) stop("NAs in input data.")
  
  # output units
  outunits <- ncol(Y)
  
  units <- c(m,hiddenUnits,outunits)	
  
  doDropout <- FALSE
  if (any(dropout>0)) {
    doDropout <- TRUE
  }
  if (length(dropout)!=L-1){
    stop("Length of dropout vector must be number of hidden units plus one!")
  }
  dropout <- c(dropout,0)
  
  # validation set
  if (!is.null(Xval)) {
    if (is.data.frame(Xval)) Xval <- as.matrix(Xval)
    if (is.vector(Yval)) Yval <- matrix(Yval,ncol=1)
    if (nrow(Xval)!=nrow(Yval)) stop("Number of labels of validation set does not match number of samples.")
    if (ncol(Y)!=ncol(Yval)) stop("Number of outputs is different for validation set.")
    if (any(is.na(Xval))) stop("NAs in validation data.")
  }
  
  # log File
  if (!is.null(saveWeights)) logFile <- paste0(saveWeights,".log") else logFile <- paste0(basename(tempfile()),".log")
  cat(timestamp(),file=logFile)
  # sigmoid function
  if (activationFunction=="sigmoid"){
    activationF <- function(x){
      return( 1/(1+exp(-x))-0.5)
    }
    activationFDeriv <- function(x){
      a <- activationF(x)
      return(a*(1-a) )
    }
    activationF2Deriv <- function(x){
      a <- activationF(x)
      return((1-2*a)*a*(1-a) )
    }
  } else if (activationFunction=="ReLU"){
    activationF <- function(x){
      return( (x>0) * x)
      #return(ifelse(x>0,x,g(0,type="s")))
    }
    activationFDeriv <- function(x){
      y <- (x>0)
      #type(y) <- "s"
      return(y)
      #return(ifelse(x>0,g(1,type="s"),g(0,type="s")))
    }
    activationF2Deriv <- function(x){
      y <- pp1*exp(-pp2*x^2)
      return(y)
    }
  } else if (activationFunction=="eLU"){
    activationF <- function(x){
      y <- (x>0) * x + (x<0) * ( exp(x * (x<0)) -1) 
      return(y)
    }
    activationFDeriv <- function(x){
      y <- (x>0) * g(1,type="s") + (x<0) * (activationF(x)+1)
      return(y)
    }
    activationF2Deriv <- function(x){
      stop("not implemented")
      return(y)
    }
  } else if (activationFunction=="eLUapprox"){
    activationF <- function(x){
      return( (x > (-1)) * x)
    }
    activationFDeriv <- function(x){
      return(x > (-1))
    }
    activationF2Deriv <- function(x){
      stop("not implemented")
      return(y)
    }
  } else if (activationFunction=="linear"){
    activationF <- function(x){
      return(x)
    }
    activationFDeriv <- function(x){
      y <- rep(1.0,length(x))
      return(y)
    }
    activationF2Deriv <- function(x){
      return(0)
    }
  } else if (activationFunction=="softmax"){
    activationF <- function(x){
      y <- x
      offset <- g(apply(abs(x),1,max),type="s")
      s <- log(rowSums(exp(x - offset))) + offset
      y <- exp(y-s)
      return(y)
    }
    activationFDeriv <- function(x){
      a <- activationF(x)
      return(a*(1-a) )
    }
    activationF2Deriv <- function(x){
      a <- activationF(x)
      return((1-2*a)*a*(1-a) )
    }
  } else {
    stop("ActivationFunction must be \"sigmoid\" or \"ReLU\" or eLU or eLUapprox or \"linear\" or \"softmax\"")
  }
  
  # sigmoid function output
  if (activationOutput=="sigmoid"){
    activationFOutput <- function(x){
      return( 1/(1+exp(-x)))
    }
    activationFOutputDeriv <- function(x){
      a <- activationFOutput(x)
      return(a*(1-a) )
    }
    activationFOutput2Deriv <- function(x){
      a <- activationF(x)
      return((1-2*a)*a*(1-a) )
    }
  } else if (activationOutput=="ReLU"){
    activationFOutput <- function(x){
      #return(ifelse(x>0,x,g(0,type="s")))
      return( (x>0) * x)
    }
    activationFOutputDeriv <- function(x){
      #return(ifelse(x>0,g(1,type="s"),g(0,type="s")))
      y <- (x>0)
      #type(y) <- "s"
      return(y)
    }
    activationFOutput2Deriv <- function(x){
      y <- pp1*exp(-pp2*x^2)
      return(y)
    }
  } else if (activationOutput=="linear"){
    activationFOutput <- function(x){
      return(x)
    }
    activationFOutputDeriv <- function(x){
      y <- rep(1.0,length(x))
      return(y)
    }
    activationFOutput2Deriv <- function(x){
      return(0)
    }
  } else if (activationOutput=="softmax"){
    activationFOutput <- function(x){
      xs <- (x > 20) * g(20,type="s") + (x <= 20) * x
      offset <- g(apply(xs,1,max),type="s")
      ss <- log(rowSums(exp(xs - offset))) + offset
      y <- exp(xs-ss)
      return(y)
    }
    activationFOutputDeriv <- function(x){
      a <- activationFOutput(x)
      return(a*(1-a) )
    }
    activationFOutput2Deriv <- function(x){
      a <- activationF(x)
      return((1-2*a)*a*(1-a) )
    }
  } else {
    stop("ActivationFunctionOutput must be \"sigmoid\" or \"ReLU\" or \"linear\" or \"softmax\"")
  }
  
  if (loss=="quadratic"){
    lossf <- function(x,y){
      return((x-y)^2)
    }
    #lossfd <- function(x,y){
    #	return(2*(x-y))
    #}
  } else if (loss=="logistic"){
    lossf <- function(x,y){
      x <- pmin(pmax(x,0.001),0.999)
      rv <- -(y*log(x)+(1-y)*log(1-x))
      return(rv)
    }
    # lossfd: not needed
  } else if (loss=="cross-entropy"){
    lossf <- function(x,y){
      return(-y*log(x+1e-14))
    }
    # lossfd: not needed
  } else {
    stop('Error function not known. Use "quadratic" or "logistic" or "cross-entropy"')
  }
  
  
  #minError
  minError <- Inf
  bestW1 <- list()
  
  trainErrorPerStep <- c()
  valErrorPerStep <- c()
  #accuracyPerStep <- c()
  AUCPerStep <- c()
  
  #minErrorC
  if (learnType=="classification") {
    minErrorC <- Inf
    bestCW1 <- c()
  }
  
  #Learning Rate
  eta <- eta/batch
  
  
  ### definition of GPU objects
  # init weight matrix
  
  if (is.character(init) & is.null(W)){
    message("Using given strategy as initialization.")
    ggk <- rep(NA,L-1)
    #if (activationFunction!="ReLU") activationCorr <- 1 else activationCorr <- 2
    if (init=="MS") {
      #for (i in 1:(L-1)) ggk[i] <- sqrt(2)/sqrt(min(units[i],units[i+1]))
      for (i in 1:(L-1)) ggk[i] <- sqrt(2 * 1/(1-dropout[i]) ) /sqrt(units[i])
    } else if (init=="MS2") {
      for (i in 1:(L-1)) ggk[i] <- sqrt(2 * 1/(1-dropout[i]) )/sqrt(sqrt(units[i]*units[i+1]))
    } else if (init=="Clamp") {
      for (i in 1:(L-1)) ggk[i] <- sqrt(exp(1) * 1/(1-dropout[i]) )/sqrt(units[i])
    } else if (init=="MS-ELU") {
      warnings("With initializiation MS-ELU the input variance must be normalized to one.")
      for (i in 1:(L-1)) ggk[i] <- sqrt( 1/(1-dropout[i]) )/sqrt(0.645 * units[i])
         
    } else if (init=="SV") {
      for (i in 1:(L-1)) {
        if (units[i]==units[i+1]){
          ggk[i] <- 1*sqrt(2)/sqrt(units[i])
        } else {
          mi <- min(units[i],units[i+1])
          ma <- max(units[i],units[i+1])
          ggk[i] <- sqrt(2* 1/(1-dropout[i]) ) * sqrt(exp(-1/mi*sum(log(1- (1:mi)/ma))) * 1/ma)
        } 
      }	
    } else {
      stop("Not implemented initialization method.")
    }		
    W <- list()
    deltaW <- list()
    #momW <- list()
    for (i in 1:(L-1)){
      W[[i]] <- gmatrix(grnorm(n=units[i]*units[i+1],sd=ggk[i],type="s"),nrow=units[i],ncol=units[i+1],type="s")	
      deltaW[[i]] <- gmatrix(0,nrow=units[i],ncol=units[i+1],type="s")
    }
    #deltaW <- momW
    
  } else if (is.numeric(init) & is.null(W)){
    ssg <- as.numeric(init)		
    W <- list() 
    deltaW <- list()
    #momW <- list()
    for (i in 1:(L-1)){
      W[[i]] <- gmatrix(grnorm(n=units[i]*units[i+1],sd=ssg,type="s"),nrow=units[i],ncol=units[i+1],type="s")	
      deltaW[[i]] <- gmatrix(0,nrow=units[i],ncol=units[i+1],type="s")
    }
    #deltaW <- momW
  } else if (!is.null(W)){
    W <- lapply(W,g,type="s")
    deltaW <- list()
    for (i in 1:(L-1)){
      deltaW[[i]] <- gmatrix(0,nrow=units[i],ncol=units[i+1],type="s")
    }
    
    message("Using given weight matrices as initialization.")
  } else {
    stop("Initialization type not recognized.")
  }
  
  
  netI <- list()	
  deltaHidden <- list()
  act <- list()
  dropMask <- list()
   for (i in 1:L){
    netI[[i]] <- gmatrix(0,nrow=batch,ncol=units[i],type="s")
    deltaHidden[[i]] <- gmatrix(0,nrow=batch,ncol=units[i],type="s")
    act[[i]] <- gmatrix(0,nrow=batch,ncol=units[i],type="s")
    dropMask[[i]] <- gmatrix(0,nrow=batch,ncol=units[i],type="s")
  }
  
  
  TlossEp <- c(0)
  TclassEp <- c(0)
  VlossEp <- c(0)
  VclassEp <- c(0)
  
  
  nbrUpdates <- 0
  for (epoch in 1:(epochs+fineTuneEpochs)){
    # find minibatches
    if (n%%batch==0){
      permIdx <- sample(1:n,replace=replace)
    } else {
      permIdx <- c(sample(1:n,replace=replace) , sample(1:n,size=batch-(n%%batch), replace=replace))
      if (length(permIdx) %% batch != 0) browser()
    }
    idxList <- lapply(1:ceiling(n/batch), function(i) {
      idx <- permIdx[(batch*(i-1)+1):(batch*(i))]
      idx <- idx[!is.na(idx)]
      return(idx)
    })
    
    
    currentError <- 0
    errorVec <- rep(0,ncol(Y))
    
    
    lll <- length(idxList)
    for (i in 1:lll){
      idx <- idxList[[i]]
      x.tmp <- X[idx, ,drop=FALSE]
      if (!(nrow(x.tmp)==batch & ncol(x.tmp)==m & all(!is.na(x.tmp)) & length(idx)==batch & all(idx %in% 1:n))) browser()
      y <- as.matrix(Y[idx, , drop=FALSE])
      if (class(Y)=="dgCMatrix"){
        y[y ==    0] <- NA
        y[y == (-1)] <- 0
      }
      x <- gmatrix(x.tmp,nrow=nrow(x.tmp),type="s")
      
      if (dropout[1]>0) {
        x <- x * gmatrix(grbinom(n=m*nrow(x),size=1,prob=1-dropout[1]),nrow=nrow(x),type="s")
        if(any(is.nan(h(x)))) {
          x <- g(X[idx, ,drop=FALSE],type="s")
          x <- x * gmatrix(grbinom(n=m*nrow(x),size=1,prob=1-dropout[1]),nrow=nrow(x),type="s")
          if(any(is.nan(h(x)))) {
            x <- g(X[idx, ,drop=FALSE],type="s")
            x <- x * gmatrix(grbinom(n=m*nrow(x),size=1,prob=1-dropout[1]),nrow=nrow(x),type="s")
            if(any(is.nan(h(x)))) {
              x <- g(X[idx, ,drop=FALSE],type="s")
              x <- x * gmatrix(grbinom(n=m*nrow(x),size=1,prob=1-dropout[1]),nrow=nrow(x),type="s")
            }
          }
          
        }
      }
      
      ### FORWARD PASS
      act[[1]] <- x			
      for (l in 1:(L-1)){
        if (l < (L-1)){
          gmm(act[[l]],W[[l]],netI[[l+1]])
          if (dropout[l+1]>0) {
            dropMask[[l+1]] <- gmatrix(grbinom(n=units[l+1]*batch,size=1,prob=1-dropout[l+1]), nrow=batch, type="s")
            act[[l+1]] <- activationF(netI[[l+1]]) * dropMask[[l+1]]
          } else {
            act[[l+1]] <- activationF(netI[[l+1]])
          }
        } else {
          gmm(act[[l]],W[[l]],netI[[l+1]])
          if (dropout[l+1]>0) {
            dropMask[[l+1]] <- gmatrix(grbinom(n=units[l+1]*batch,size=1,prob=1-dropout[l+1]), nrow=batch, type="s")
            act[[l+1]] <- activationFOutput(netI[[l+1]]) * dropMask[[l+1]]
          } else {
            act[[l+1]] <- activationFOutput(netI[[l+1]])
          }
        }
      }
      
      
      act.output.cpu <- h(act[[L]])
      if (any(is.na(act.output.cpu) | !is.finite(act.output.cpu))) message("NAs in activations of output layer!")
      
      if (loss=="quadratic" & activationOutput != "linear") {
        delta.output.cpu <- (act.output.cpu - y) * as.vector(activationFOutputDeriv(h(netI[[L]])))
        #delta2.output.cpu <-  as.vector(activationFOutputDeriv(netI.output))^2 + (act.output.cpu - y) * as.vector(activationFOutput2Deriv(netI.output))
      } else if (loss=="quadratic" & activationOutput=="linear") {
        delta.output.cpu <- (act.output.cpu - y)
        #delta2.output.cpu <- rep(1,outunits)
      } else if (loss=="logistic" & activationOutput=="sigmoid") {
        delta.output.cpu <- (act.output.cpu - y)
        #delta2.output.cpu <- act.output.cpu*(1- act.output.cpu)
      } else if (loss=="cross-entropy" &  activationOutput=="softmax" ) {
        #browser()
        delta.output.cpu <- (act.output.cpu - y)
        #delta2.output.cpu <- act.output.cpu*(1- act.output.cpu)
      } else {
        stop("Not implemented")
      }
      
      
      if (any(is.na(delta.output.cpu) | !is.finite(delta.output.cpu))) delta.output.cpu <- ifelse(is.na(delta.output.cpu), 0, delta.output.cpu)
      #delta.output <- g(delta.output.cpu,type="s") #move deltas to gpu
      #act.output.cpu <- h(act.output)
      #if (anyNA) delta2.output.cpu <- ifelse(is.na(delta2.output.cpu), 0, delta2.output.cpu)
      #delta2.output <- g(delta2.output.cpu,type="s") #move delta2s to gpu
      
      errV <- computeError(y, act.output.cpu, lossf=lossf, learnType=learnType)
      
      if (!all(is.na(errV$errorMat))) {
        currentError <- 1/i*((i-1)*currentError + errV$currentError)
      } else {
        currentError <- 1/i*((i-1)*currentError)
      }
      errorVec <- 1/i*( (i-1)*errorVec + errV$errorVec)
      
      
      cat("\n#######\nEpoch:",epoch,"Batch:",i,"\n TRAIN: \n")
      cat(" Training loss:",errV$currentError, "\n")
      cat("\n")
      cat("\n#######\nEpoch:",epoch,"Batch:",i,"\n TRAIN: \n",file=logFile,append=TRUE)
      cat(" Training loss:",errV$currentError, "\n",file=logFile,append=TRUE)
      cat("\n",file=logFile,append=TRUE)
     
      
      
      trainErrorPerStep <- c(trainErrorPerStep,errV$currentError)
      if (!is.null(Xval) & (nbrUpdates %% 100==0) ) {
        if (any(dropout!=0)){
          for (l in 1:length(W)){
            W[[l]] <- W[[l]]*(1-dropout[l])
          } 
        }
        
        if (validationErrorMode=="subset"){
          idxRandVal <- sample(1:nrow(Xval),batch)
          YvalSelected <- as.matrix(Yval[idxRandVal, ])
          if (class(Y)=="dgCMatrix"){
            YvalSelected[YvalSelected ==    0] <- NA
            YvalSelected[YvalSelected == (-1)] <- 0
          }
          
          val.output <- forwardPassGPU(Xval[idxRandVal, ], W, dropout=rep(0,length(dropout)),activationF=activationF,activationFOutput=activationFOutput)
          errV <- computeError(YvalSelected, val.output, lossf=lossf, learnType=learnType)
          currentError <- errV$currentError
          VlossEp <- c(VlossEp,errV$currentError)
          
          cat("VALIDATION \n")
          cat("Approx. validation Loss:", errV$currentError, "\n")
          AUCPerStepTmp <- rep(NA,ncol(YvalSelected))
          idxAUC <- which(apply(YvalSelected,2,function(x) all(c(0,1) %in% x)))
          AUCPerStepTmp[idxAUC] <- sapply(idxAUC,function(mm) return(AUC(val.output[,mm],YvalSelected[,mm]) ) )
          cat("\nERROR:", paste(AUCPerStepTmp,collapse=","),"\n")
          
          cat("\n")
          cat("VALIDATION \n",file=logFile,append=TRUE)
          cat("Approx. validation Loss:", errV$currentError, "\n",file=logFile,append=TRUE)

          cat("\nERROR:", paste(AUCPerStepTmp,collapse=","),"\n",file=logFile,append=TRUE)
          cat("\n",file=logFile,append=TRUE)

          valErrorPerStep <- c(valErrorPerStep,errV$currentError)
          AUCPerStep <- cbind(AUCPerStep,AUCPerStepTmp)

        } else if (validationErrorMode=="full") {
          valIdxList <- c(seq(1,nrow(Xval),batch),nrow(Xval)+1)
          val.output <- matrix(NA,nrow=nrow(Yval),ncol=ncol(Yval))
          if (nbrUpdates <= 1){
            YvalSelected <- as.matrix(Yval)
            if (class(Y)=="dgCMatrix"){
              YvalSelected[YvalSelected ==    0] <- NA
              YvalSelected[YvalSelected == (-1)] <- 0
            }	
          }
          
          for (bb in 1:(length(valIdxList)-1)){
            idxRandVal <- (valIdxList[bb]:(valIdxList[bb+1]-1))
            val.output[idxRandVal, ] <- forwardPassGPU(Xval[idxRandVal, ], W, dropout=rep(0,length(dropout)),activationF=activationF,activationFOutput=activationFOutput)
          }
          errV <- computeError(YvalSelected, val.output, lossf=lossf, learnType=learnType)
          currentError <- errV$currentError
          VlossEp <- c(VlossEp,errV$currentError)
          
          cat("VALIDATION \n")
          cat("Approx. validation Loss:", errV$currentError, "\n")
          AUCPerStepTmp <- rep(NA,ncol(YvalSelected))
          idxAUC <- which(apply(YvalSelected,2,function(x) all(c(0,1) %in% x)))
          AUCPerStepTmp[idxAUC] <- sapply(idxAUC,function(mm) return(AUC(val.output[,mm],YvalSelected[,mm]) ) )
          cat("\nMEAN AUC:", mean(AUCPerStepTmp,na.rm=TRUE),"\n")          
	  cat("\n")
          cat("VALIDATION \n",file=logFile,append=TRUE)
          cat("Approx. validation Loss:", errV$currentError, "\n",file=logFile,append=TRUE)
    	  cat("\nMEAN AUC:", mean(AUCPerStepTmp,na.rm=TRUE),"\n",file=logFile,append=TRUE)          
  		cat("\n",file=logFile,append=TRUE)
          
          valErrorPerStep <- c(valErrorPerStep,errV$currentError)
          AUCPerStep <- cbind(AUCPerStep,AUCPerStepTmp)
          
        } else {
          stop("ValidationErrorMode unknown.") 
        }
        
        if (any(dropout!=0)){
          for (l in 1:length(W)){
            W[[l]] <- W[[l]] / (1-dropout[l])
          } 
        }
        
      }
      
      
      
      ### BACKPROP
      
      nbrUpdates <- nbrUpdates+1
      deltaHidden[[L]] <- g(delta.output.cpu,type="s")
      
      
      if (monitoring) browser()
      for (l in (L-1):1){
        gmm(deltaHidden[[l+1]],W[[l]],deltaHidden[[l]],trA=FALSE,trB=TRUE)
        if (dropout[l]>0) {
          deltaHidden[[l]] <- deltaHidden[[l]] * (activationFDeriv(netI[[l]])*dropMask[[l]])
        } else {
          deltaHidden[[l]] <- deltaHidden[[l]] * activationFDeriv(netI[[l]])
        }
        gmm(act[[l]],deltaHidden[[l+1]],deltaW[[l]],trA=TRUE,trB=FALSE)
      }
           
      
      # UPDATE WEIGHTS
      if ((nbrUpdates-1) %% verbose == 0 & verbose > 0) {
        for (l in 1:(L-1)){
          W[[l]] <- W[[l]] - eta*deltaW[[l]]
          ndim <- min(nrow(W[[l]]),ncol(W[[l]]))
          Ndim <- max(nrow(W[[l]]),ncol(W[[l]]))
          SSDX <- sd(W[[l]])
          if (ncol(W[[l]]) > nrow(W[[l]])) EEX <- sqrt(eigen(tcrossprod(W[[l]]))$values) else EEX <- sqrt(eigen(crossprod(W[[l]]))$values)
          #if (is.null(saveWeights)){
          cat("Layer:",l,"Estimated Spectral radius:", sqrt(min(nrow(W[[l]]),ncol(W[[l]]))) *SSDX ,"\n")
          cat("Product of singular values (Det):", prod(EEX),"\n")
          cat("Layer:",l,"Estimated lower bound of SV:", (1-sqrt(ndim/Ndim)) *sqrt(Ndim) * SSDX, "Estimated upper bound of SV:", (1+sqrt(ndim/Ndim)) *sqrt(Ndim) * SSDX,"\n")
          cat("Layer:",l,"Calculated SV: Min:", min(EEX),"Max:",max(EEX),"\n")
          #} else {
          cat("Layer:",l,"Estimated Spectral radius:", sqrt(min(nrow(W[[l]]),ncol(W[[l]]))) *SSDX ,"\n",file=logFile,append=TRUE)
          cat("Product of singular values (Det):", prod(EEX),"\n",file=logFile,append=TRUE)
          cat("Layer:",l,"Estimated lower bound of SV:", (1-sqrt(ndim/Ndim)) *sqrt(Ndim) * SSDX, "Estimated upper bound of SV:", (1+sqrt(ndim/Ndim)) *sqrt(Ndim) * SSDX,"\n",file=logFile,append=TRUE)
          cat("Layer:",l,"Calculated SV: Min:", min(EEX),"Max:",max(EEX),"\n",file=logFile,append=TRUE)
          #}
        }
        
      } else {
        for (l in 1:(L-1)) W[[l]] <- W[[l]] - eta*deltaW[[l]]
      }
      
      if (detNorm){
        for (l in 1:(L-1)){
          sss <- sd(W[[l]])
          if ( init=="Clamp"){
            W[[l]] <- sqrt(exp(1))/sqrt( nrow(W[[l]]) ) *1/sss * W[[l]] 
          } else if (init=="MS") {
            W[[l]] <- sqrt(2)/sqrt( nrow(W[[l]]) ) *1/sss * W[[l]] 
          } else if (is.numeric(init)){
            W[[l]] <- init *1/sss * W[[l]] 
          } else {
            stop("Not implemented!")
          }          
        }
      }
      
    }
    
    if (is.null(Xval)){
      if (currentError < minError & currentError > 0) {
        bestW <- lapply(W,h)
        minError <- currentError
      }
    } else {
      if (currentError < minError & currentError > 0) {
        bestW <- lapply(W,h)
        minError <- currentError
      }
    }
    
    
    
    ggc(silent=FALSE)
    
  }
  
  if (epoch==epochs) {
    for (l in 1:length(W)){
      W[[l]] <- W[[l]]*(1-dropout[l])
    }                   
    dropout <- rep(0,length(dropout))
    doDropout=FALSE
  }
  
  Ytrain <- matrix(NA,nrow=nrow(Y),ncol=ncol(Y))
  for (i in 1:length(idxList)){
    ### Forward Pass
    idx <- idxList[[i]]	
    Ytrain[idx, ] <- forwardPassGPU(X[idx, ,drop=FALSE], W, dropout=rep(0,length(dropout)),activationF=activationF,activationFOutput=activationFOutput)
  }
  if (!is.null(Yval)){
    val.output <- NA
  } else {
    val.output <- NA
  }
  
  result <- list()
  result$ytrain <- Ytrain
  result$yval <- val.output
  result$weights <- lapply(W,h)
  result$activationf <- activationF
  result$activationfOutput <- activationFOutput
  result$bestWeights <- bestW
  #result$bestWeightsC <- bestC.W
  result$dropout <- dropout
  result$TlossEp <- TlossEp
  result$TclassEp <- TclassEp
  result$VlossEp <- VlossEp
  result$VclassEp <- VclassEp
  result$trainErrorPerStep <- trainErrorPerStep
  result$valErrorPerStep <- valErrorPerStep
  #result$accuracyPerStep <- accuracyPerStep
  result$AUCPerStep <- AUCPerStep
  return(result)
}


predictNNgpu <- function(X,nnm,batch=128, replace=FALSE, useBestWeights=FALSE){
  n <- nrow(X)
  m <- ncol(rr$weights[[length(rr$weights)]])
  if (n%%batch==0){
      permIdx <- sample(1:n,replace=replace)
    } else {
      permIdx <- c(sample(1:n,replace=replace) , sample(1:n,size=batch-(n%%batch), replace=replace))
      if (length(permIdx) %% batch != 0) browser()
    }
    idxList <- lapply(1:ceiling(n/batch), function(i) {
      idx <- permIdx[(batch*(i-1)+1):(batch*(i))]
      idx <- idx[!is.na(idx)]
      return(idx)
  })
    
  W <- list()
  if (!useBestWeights){
    for (i in 1:length(nnm$weights)){
      W[[i]] <- g(nnm$weights[[i]],type="s")
    }		
  } else {
    for (i in 1:length(nnm$weights)){
      W[[i]] <- g(nnm$bestW[[i]],type="s")
    }
  }
 
  predictions <- matrix(NA,nrow=n,ncol=m)
  for (i in 1:length(idxList)){
    ### Forward Pass
    idx <- idxList[[i]]	
    predictions[idx, ] <- forwardPassGPU(X[idx, ], W, dropout=rep(0,length(nnm$dropout)),activationF=nnm$activationf,activationFOutput=nnm$activationfOutput)
  }
  return(predictions)
}
