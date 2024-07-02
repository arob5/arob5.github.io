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



gp1 <- kergp::gp(y~1, data=df, inputs="x", cov=quad_ker, estim=FALSE)
