#
# quadratic_kernel_tests.r
#
# Andrew Roberts
#

library(kergp)
library(ggmatplot)

# Settings.   
N <- 20
N_test <- 100
f <- function(x) 10 * 2*(x - 10)^2

# Training data. 
X <- matrix(seq(-1,1, length.out=N), ncol=1)
colnames(X) <- "x"
y <- f(X) + 100*rnorm(n=N)
df <- data.frame(x=drop(X), y=drop(y))

# Test data. 
X_test <- matrix(seq(-1, 1, length.out=N_test), nrow=N_test)
y_test <- f(X_test)
colnames(X_test) <- "x"


# Quadratic kernel. 
quad_ker_func <- function(x1, x2, par) { 
  affine_comb <- sum(x1 * x2) + par[1]
  kern <- affine_comb^2
  attr(kern, "gradient") <- c(cst=2*affine_comb)
  return(kern)
}

quad_ker <- covMan(kernel = quad_ker_func,
                   hasGrad = TRUE,
                   d = 1L,
                   parLower = c(cst=-Inf),
                   parUpper = c(cst=Inf),
                   parNames = c("cst"),
                   label = "polynomial kernel 1", 
                   inputs = "x", 
                   par = c(cst=1.0))

# Kernel matrix. 
K <- covMat(quad_ker, X)
chol_test <- chol(K) # PSD, not PD. 
eps <- sqrt(.Machine$double.eps)
chol_test <- chol(hetGP:::add_diag(K, rep(eps, N))) # Hack using jitter to make PD. 

# Draw prior samples at test points. 
K_test <- covMat(quad_ker, X_test)
eig <- eigen(K_test, symmetric=TRUE)
eig$values[eig$values < 0] <- 0 # Some small negative values due to numerical roundoff. 
S <- eig$vectors %*% diag(sqrt(eig$values))
n_samp <- 5
prior_samp <- S %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw samples at x=0. 
prior_samp_0 <- drop(sqrt(covMat(quad_ker, matrix(0)))) * rnorm(10000)
hist(prior_samp_0, breaks=30)
mean(prior_samp_0) # close to 0, as expected
sd(prior_samp_0) # close to 1, as expected. 

# Draw samples at x=0, using larger value of constant param. 
coef(quad_ker) <- c(cst=10)
prior_samp_0 <- drop(sqrt(covMat(quad_ker, matrix(0)))) * rnorm(10000)
hist(prior_samp_0, breaks=30)
mean(prior_samp_0) # close to 0, as expected
sd(prior_samp_0) # close to 10, as expected.

# Ridge regression. 
K <- covMat(quad_ker, x)
K_lambda <- hetGP:::add_diag(K, rep(eps, N))
y_hat <- covMat(quad_ker, X_test, x) %*% chol2inv(K_lambda) %*% y  
plot(x, y)
lines(X_test, y_test, col="red")
lines(X_test, y_hat, col="blue")

#
# Adding kernels.  
#

default_nugget <- sqrt(.Machine$double.eps)

# Quadratic kernel plus Gaussian kernel. 
quad_plus_Gauss_ker_func <- function(x1, x2, par) { 
  
  # Gaussian part.
  kern <- kergp:::kNormFun(x1, x2, par, kergp::k1FunGauss)
  
  # Quadratic part. Currently setting the `c` parameter to 1. 
  kern <- kern + (tcrossprod(x1, x2) + 1)^2
  # affine_comb <- sum(x1 * x2) + 1
  # kern <- affine_comb^2
  
  # Jitter. Hack: add jitter to diagonal if number of rows are equal.
  if(nrow(x1) == nrow(x2)) kern <- hetGP:::add_diag(kern, rep(default_nugget, nrow(x1)))
  
  # Gradient. 
  # attr(kern, "gradient") <- c(cst=2*affine_comb)
  
  return(kern)
}


d <- 1L

quad_plus_Gauss_ker <- covMan( 
    kernel = quad_plus_Gauss_ker_func,
    hasGrad = TRUE,
    acceptMatrix = TRUE,
    d = d,
    par = c(rep(1, d), 1),
    parLower = rep(1e-8, d + 1L),
    parUpper = rep(Inf, d + 1L),
    parNames = c(paste("theta", 1L:d, sep = "_"), "sigma2"),
    label = "Gauss plus quadratic kernel"
)


# Draw prior samples at test points: default hyperparams. 
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: very small marginal variance. 
coef(quad_plus_Gauss_ker)["sigma2"] <- .000001
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: very large marginal variance. 
coef(quad_plus_Gauss_ker)["sigma2"] <- 20
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: small lengthscale. 
coef(quad_plus_Gauss_ker)["sigma2"] <- 1
coef(quad_plus_Gauss_ker)["theta_1"] <- .001
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: small lengthscale, small marginal variance.  
coef(quad_plus_Gauss_ker)["sigma2"] <- .001
coef(quad_plus_Gauss_ker)["theta_1"] <- .001
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: large lengthscale.  
coef(quad_plus_Gauss_ker)["sigma2"] <- 1
coef(quad_plus_Gauss_ker)["theta_1"] <- 10
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: large lengthscale, large marginal variance.   
coef(quad_plus_Gauss_ker)["sigma2"] <- 100
coef(quad_plus_Gauss_ker)["theta_1"] <- 10
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: large lengthscale, small marginal variance.   
coef(quad_plus_Gauss_ker)["sigma2"] <- .001
coef(quad_plus_Gauss_ker)["theta_1"] <- 10
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

# Draw prior samples at test points: middle ground.  
coef(quad_plus_Gauss_ker)["sigma2"] <- 1
coef(quad_plus_Gauss_ker)["theta_1"] <- .1
K_test <- covMat(quad_plus_Gauss_ker, X_test)
L <- t(chol(K_test))
n_samp <- 5
prior_samp <- L %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp, type="l")

