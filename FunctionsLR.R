# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  N <- nrow(X)            # the number of training samples
  Nt <- nrow(Xt)          # the number of testing samples
  K <- length(unique(y))  # the number of categories
  p <- ncol(X)            # the length of a sample
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if (!identical(X[,1], rep(1,times=N)) || !identical(Xt[,1], rep(1,times=Nt))) stop("Error: Wrong intercepts in training or testing matrices!")
  # Check for compatibility of dimensions between X and Y
  if (!identical(N, length(y))) stop("Error: Not compatible for multiplication: training data sets and labels!")
  # Check for compatibility of dimensions between Xt and Yt
  if (!identical(Nt, length(yt))) stop("Error: Not compatible for multiplication: testing data sets and labels!")
  # Check for compatibility of dimensions between X and Xt
  if (!identical(p, ncol(Xt))) stop("Error: Not compatible in length: training and testing data!")
  # Check eta is positive
  if (eta <= 0) stop("Error: Bad choice on learning/damping rate!")
  # Check lambda is non-negative
  if (lambda < 0) stop("Error: Bad choice on ridge parameter!")
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)) {
    beta_init <- matrix(0, nrow=p , ncol=K)
  } else {
    if (!identical(nrow(beta_init), p) || !identical(ncol(beta_init), K)) {
      stop("Error: Not compatible: Wrong size of initial beta.")
    }
  }
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  ### Initialization
  error_train <- rep(0, times=numIter+1)
  error_test <- rep(0, times=numIter+1)
  objective <- rep(0, times=numIter+1)
  ### Initial values
  exp_mat <- exp(X %*% beta_init)
  pk <- exp_mat / rowSums(exp_mat)    # muti-sigmoid functions
  pos_ind <- cbind(1:N, (y+1))        # position of indicator function to be one
  objective[1] <- -mean(log(pk[pos_ind])) + 0.5*lambda*norm(beta_init, type="F")^2
  error_train[1] <- sum(max.col(pk) != (y+1)) / N * 100
  error_test[1] <- sum(max.col(Xt %*% beta_init) != (yt+1)) / Nt * 100
  beta <- beta_init
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  for (iter in seq_len(numIter)) {
    # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
    ## Update step
    pk_adj <- pk
    pk_adj[pos_ind] <- pk_adj[pos_ind] - 1   # adjust p_k to p_k-indicator function
    grad <- t(X) %*% pk_adj / N + lambda*beta    # gradient vector as columns of the matrix
    for (i in seq_len(K)) {
      diag_vec <- pk[ ,i] - pk[ ,i]^2
      hessian <- lambda * diag(p) + t(X) %*% (X * diag_vec) / N
      beta[ ,i] <- beta[ ,i] - eta * solve(hessian) %*% grad[ ,i]
    }
    
    exp_mat <- exp(X %*% beta)
    pk <- exp_mat / rowSums(exp_mat)         # n x K matrix with each row as pk
    
    ## Report step
    objective[iter+1] <- -mean(log(pk[pos_ind])) + 0.5*lambda*norm(beta, type="F")^2
    error_train[iter+1] <- sum(max.col(pk) != (y+1)) / N * 100
    error_test[iter+1] <- sum(max.col(Xt %*% beta) != (yt+1)) / Nt * 100
  }
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}