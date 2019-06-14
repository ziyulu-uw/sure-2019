import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# constants
omega = 0.5  # omega = k/m
mu = 0.01    # mu  = gamma/m
w = math.sqrt(omega - 0.25*mu**2)
N = 200  # number of time steps in one simulation
dt = 0.05  # step size in one simulation
sigma = 0.1  # noise coefficient in SDE
var_v = 0.03  # measurement noise variance
X0 = np.array([[0.0],[1.0]])
C = np.array([0.0, 1.0], ndmin=2)


lambda1 = complex(-0.5*mu,w)
lambda2 = complex(-0.5*mu,-w)

# construct matrix A
A11 = lambda2*cmath.exp(lambda1*dt) - lambda1*cmath.exp(lambda2*dt)
A12 = -cmath.exp(lambda1*dt) + cmath.exp(lambda2*dt)
A21 = lambda1*lambda2*(cmath.exp(lambda1*dt) - cmath.exp(lambda2*dt))
A22 = -lambda1*cmath.exp(lambda1*dt) + lambda2*cmath.exp(lambda2*dt)
A = np.array([[A11, A12], [A21, A22]])
A = np.divide(A, lambda2-lambda1)  # A turns out to be real
A = A.real

# construct covariance matrix R
e1 = cmath.exp(2*lambda1*dt)
e2 = cmath.exp(2*lambda2*dt)
e3 = cmath.exp((lambda1 + lambda2)*dt)
R11 = 0.5*(e1-1)/lambda1 + 0.5*(e2-1)/lambda2 - 2*(e3-1)/(lambda1 + lambda2)
R12 = -0.5*(e1-1) - 0.5*(e2-1) + 2*(e3-1)
R21 = R12
R22 = lambda1*0.5*(e1-1) + lambda2*0.5*(e2-1) - 2*lambda1*lambda2*(e3-1)/(lambda1 + lambda2)
R = np.array([[R11, R12], [R21, R22]])
R = np.multiply(R, (sigma/(lambda2-lambda1))**2)  # R turns out to be real
R = R.real


def compute_gradient(K):
    # computes dF/dK, where F = 1/2N * \sum_{n=1}^N (X_hat_n-X_n)^2
    # returns dF/dK, F

    X = X0  # initial state
    X_hat = X0  # initial state estimate
    dX_hat = np.zeros((2,2))  # initial gradient dX_hat/dK
    sum_gradients = np.zeros((1,2))
    sum_errors = 0

    for n in range(N):
        Z_hat = C @ A @ X_hat  # predicted observation
        W = np.random.multivariate_normal([0,0], R)  # gaussian system noise with mean 0 covariance R
        W = np.array(W, ndmin=2)
        W = np.transpose(W)
        X = A @ X + W  # state update
        V = np.random.normal(0, var_v)  # gaussian observation noise with mean 0 variance var_v
        Z = C @ X + V  # observation
        X_hat = A @ X_hat + K * (Z - Z_hat)  # state estimate
        diag = Z - Z_hat
        diag = np.array([1,1]) * diag[0][0]
        dX_hat = A @ dX_hat + np.diag(diag) - K @ C @ A @ dX_hat
        sum_gradients += np.transpose(X_hat - X) @ dX_hat
        error = X_hat - X
        sum_errors += error[0][0]**2 + error[1][0]**2

    return sum_gradients/N, sum_errors/(2*N)


def gradient_descent(n, alpha):
    # n -- number of gradient steps, alpha -- step size
    # performs gradient descent using dF/dK as gradient
    # returns K, a list of F at each gradient step, a list of dF/dK at each gradient step

    K = np.array([[1.0], [1.0]])  # initialize K
    err_L = []
    grad_L = []
    for i in range(n):
        grad, err = compute_gradient(K)
        err_L.append(err)
        grad_L.append(grad)
        K = K - alpha*np.transpose(grad)
        # print(K)

    return K, err_L, grad_L


def expected_gradient_descent(n, alpha, m):
    # n -- number of gradient steps, alpha -- step size, m -- number of simulations to compute expectation
    # performs gradient descent using E[dF/dK] as gradient
    # returns K, a list of E[F] at each gradient step, a list of E[dF/dK] at each gradient step

    K = np.array([[1.0], [1.0]])  # initialize K
    err_L = []
    grad_L = []
    for i in range(n):
        sum_grad = np.zeros((1,2))
        sum_err = 0
        for j in range(m):
            grad, err = compute_gradient(K)
            sum_grad += grad
            sum_err += err
        exp_grad = sum_grad/m  # E[dF/dK] is approximated by 1/m * \sum_{i=1}^N dF_i/dK
        exp_err = sum_err/m    # E[F] is approximated by 1/m * \sum_{i=1}^N F_i
        err_L.append(exp_err)
        grad_L.append(exp_grad)
        K = K - alpha * np.transpose(exp_grad)
        # print(K)

    return K, err_L, grad_L


def Gradient_descent(n,alpha):
    # calls gradient_descent(n, alpha) and plots F vs n, log(dF/dK) vs n

    K, err_L, grad_L = gradient_descent(n, alpha)
    print(K)
    print(err_L)
    x = [i for i in range(n)]
    plt.plot(x, err_L)
    plt.title("Gradient descent with {} steps and step size {}".format(str(n),str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("mean squared error of one simulation")
    plt.show()

    grad1_L = [grad_L[i][0][0] for i in range(n)]
    grad2_L = [grad_L[i][0][1] for i in range(n)]
    log_grad1_L = [math.log10(abs(grad1_L[i])) for i in range(n)]
    log_grad2_L = [math.log10(abs(grad2_L[i])) for i in range(n)]
    plt.plot(x, log_grad1_L, label='grad1')
    plt.plot(x, log_grad2_L, label='grad2')
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("log10(gradient)")
    plt.legend()
    plt.show()



def Expected_gradient_descent(n, alpha, m):
    # calls expected_gradient_descent(n, alpha, m) and plots E[F] vs n, log(E[dF/dK]) vs n

    K, err_L, grad_L = expected_gradient_descent(n, alpha, m)
    print(K)
    print(err_L)
    x = [i for i in range(n)]
    plt.plot(x, err_L)
    plt.title("Expected gradient descent with {} steps and step size {}".format(str(n), str(alpha)))
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("expected mean squared error of one simulation")
    plt.show()

    grad1_L = [grad_L[i][0][0] for i in range(n)]
    grad2_L = [grad_L[i][0][1] for i in range(n)]
    log_grad1_L = [math.log10(abs(grad1_L[i])) for i in range(n)]
    log_grad2_L = [math.log10(abs(grad2_L[i])) for i in range(n)]
    plt.plot(x, log_grad1_L, label='grad1')
    plt.plot(x, log_grad2_L, label='grad2')
    plt.xlabel("number of gradient descent steps")
    plt.ylabel("log10(gradient)")
    plt.legend()
    plt.show()


# Gradient_descent(500,0.001)
# Expected_gradient_descent(500, 0.001, 10)

