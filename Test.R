source("FunctionsNN.R")

out1 <- initialize_bw(16, 20, 10, 1e-3);

y <- c(1,2,3,4,5,1,2,3,4,5);
scores <- matrix(rnorm(10*5), 10, 5);
K <- 5;

out2 <- loss_grad_scores(y, scores, K)

