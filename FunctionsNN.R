# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, times = hidden_p);   # intercepts after the input layer
  b2 <- rep(0, times = K);          # intercepts after the hidden layer
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  set.seed(seed)
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p);
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K);
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0 
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  
  # [ToDo] Calculate loss when lambda = 0
  # loss = ...
  N <- nrow(scores);                   # num of the sample
  exp_mat <- exp(scores);
  pk <- exp_mat / rowSums(exp_mat);    # muti-sigmoid functions
  pos_ind <- cbind(1:N, (y+1));        # position of indicator function to be one
  loss <- -mean(log(pk[pos_ind]));     # loss when lambda is zero
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  # error = ...
  error <- sum(sapply(1:nrow(pk), function(i) which.max(pk[i, ])) != (y + 1)) / N * 100;
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  # grad = ...
  grad <- pk;  
  grad[pos_ind] <- grad[pos_ind] - 1;   # adjust p_k to p_k-indicator function
  grad <- grad / N;                     # gradient vector as columns of the matrix
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){

  # [To Do] Forward pass
  N <- nrow(X);              # the number of samples
  N_ones <- rep(1, N);       # all one vector
  # From input to hidden 
  hd <- X %*% W1 + outer(N_ones, b1, '*');
  # ReLU
  relu_hd <- matrix(pmax(0, hd), nrow = N, ncol = length(b1));
  # From hidden to output scores
  output <- relu_hd %*% W2 + outer(N_ones, b2, '*');
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y, output, K);
  
  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  db2 <- colSums(out$grad);              # sum of \partial f / \partial o
  dW2 <- t(relu_hd) %*% out$grad + lambda * W2;

  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  dhd <- (out$grad %*% t(W2));  # grad for hidden
  dhd[relu_hd == 0] <- 0;
  db1 <- colSums(dhd);
  dW1 <- t(X) %*% dhd + lambda * W1;
  
  # print(max(abs(dW1)))  # Check gradient magnitude for W1
  # print(max(abs(dW2)))  # Check gradient magnitude for W2

  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  # [ToDo] Forward pass to get scores on validation data
  Nval <- nrow(Xval);           # the number of samples
  N_ones <- rep(1, Nval);       # all one vector
  # From input to hidden 
  hd_val <- Xval %*% W1 + outer(N_ones, b1, '*');
  # ReLU
  relu_hd_val <- matrix(pmax(0, hd_val), nrow = Nval, ncol = length(b1));
  # From hidden to output scores
  scores_val <- relu_hd_val %*% W2 + outer(N_ones, b2, '*');
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  # exp_mat_val <- exp(scores_val);
  # pk_val <- exp_mat_val / rowSums(exp_mat_val);    # muti-sigmoid functions
  error <- sum(max.col(scores_val)!= (yval + 1)) / Nval * 100;
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n/mbatch)

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  p <- ncol(X);
  K <- length(unique(y));
  init <- initialize_bw(p, hidden_p, K, scale, seed);
  W1 <- init$W1;
  W2 <- init$W2;
  b1 <- init$b1; 
  b2 <- init$b2;
  # Initialize storage for error to monitor convergence
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    # print(batchids)
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    errori <- 0;
    for (j in 1:nBatch){
      # fit NN with samples in each batch
      pass_res <- one_pass(X[batchids == j, ], y[batchids == j], K, W1, b1, W2, b2, lambda);
      # one pass SGD update
      W1 <- W1 - rate * pass_res$grads$dW1;
      W2 <- W2 - rate * pass_res$grads$dW2;
      b1 <- b1 - rate * pass_res$grads$db1;
      b2 <- b2 - rate * pass_res$grads$db2;
      # error aggregation
      errori <- errori + pass_res$error;
      # print(errori)
    }
    
    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
    error[i] <- errori / nBatch;
    error_val[i] <- evaluate_error(Xval, yval, W1, b1, W2, b2);
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}
