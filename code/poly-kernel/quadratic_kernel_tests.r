#
# quadratic_kernel_tests.r
#
# Andrew Roberts
#


library(kergp)
library(ggmatplot)

# Test data. 
N <- 20
x <- matrix(as.numeric(1:N), ncol=1)
colnames(x) <- "x"
y <- x^2 + 100*rnorm(n=N)
df <- data.frame(x=drop(x), y=drop(y))
x_tst <- matrix(seq(-50, 50, length.out=200), ncol=1)
y_tst <- x_tst^2
colnames(x_tst) <- "x"

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

N_test <- 100
X_test <- matrix(seq(0, N, length.out=N_test), nrow=N_test)
y_test <- X_test^2
K_test <- covMat(quad_ker, X_test)
chol_test <- chol(K_test) # PSD, not PD. 
eps <- sqrt(.Machine$double.eps)
chol_test <- chol(hetGP:::add_diag(K_test, rep(eps, N_test))) # Hack using jitter to make PD. 

# Draw prior samples. 
eig <- eigen(K_test, symmetric=TRUE)
S <- eig$vectors %*% diag(sqrt(eig$values))
n_samp <- 5
prior_samp <- S %*% matrix(rnorm(N_test*n_samp), nrow=N_test, ncol=n_samp)
matplot(X_test, prior_samp)

# Ridge regression. 
K <- covMat(quad_ker, x)
K_lambda <- hetGP:::add_diag(K, rep(eps, N))
y_hat <- covMat(quad_ker, X_test, x) %*% chol2inv(K_lambda) %*% y  
plot(x, y)
lines(X_test, y_test, col="red")
lines(X_test, y_hat, col="blue")



gp1 <- kergp::gp(y~1, data=df, inputs="x", cov=quad_ker, estim=FALSE)
