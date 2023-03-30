import copy
import math
import numpy as np


# Find w and b ---> Gradient desecnt
# Calculate gradient
def p_cal_grad(X, y, w, b, a):

    m, n = X.shape
    fwb = (np.matmul(X.reshape(m,a), w.reshape(a,1)) + b)  # using reshape to make desired order and to make more readeable the orders of matrix
    diff = fwb - y.reshape(m,1)

    djdw = np.matmul(diff.reshape(1, m), X)
    djdb = np.sum(diff)

    return (djdw / m), (djdb / m)


# Calculate cost
def p_cal_cost(X, y, w1, b1, a):

    m, n = X.shape
    fwb = (np.matmul(X.reshape(m,a), w1.reshape(a,1)) + b1)
    diff = fwb - y.reshape(m,1)
    acst = np.sum(diff**2)/(2*m)  # Average cost to get a smaller value

    return acst


# Gradient Descent
def p_find_wb(X, y, winit, binit, alpha, num_iters, fcost, fgrad, a):
    # Recording history of cost and the parameters w and b for each iteration
    recJ = []  
    w0 = copy.deepcopy(winit)
    b0 = binit
    for i in range(num_iters):
        djdw, djdb = fgrad(X, y, w0, b0, a)  # fgrad to call cal_gard
                                          # fcost to call cal_cost
        # Simultaneous update of parameters
        w0 -= alpha * djdw
        b0 -= alpha * djdb
        if i < 100000:
            recJ.append(fcost(X, y, w0, b0, a))

        if i % (math.ceil(num_iters/10)) == 0:
            print(f"Itertaion no. {i}: cost {recJ[-1]:0.4e}")
        
        # Last iterations to check variation
        if (num_iters - 10) <= i <= (num_iters - 1):
            print(f"Itertaion no. {i}: cost {recJ[-1]:0.3e}")  

    return w0, b0, recJ