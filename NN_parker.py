import numpy as np
import math
import csv

def sigmoid(z):
    return np.matrix( 1 / (1.0 + np.exp(-z)) )

def sig_prime(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

'''
INPUTS
X: d by 1 array
Y: k by 1 array where k is number of classifications
w1: m x (d+1) matrix where m is the number of hidden units
w2: k x (m+1) matrix

OUTPUTS
X: (d+1) by 1 matrix representing inputs with added constant input of "1"
Y: k by 1 matrix representing classifications
f: k x 1 matrix representing output of output unit
z1: (m+1) x 1 matrix representing output of hidden unit
a1: (m+1) x 1 matrix representing activation of hidden unit
loss: loss according to J(w) equation
'''
def feedforward(X, Y, w1, w2):    
    #(X, Y) = prepare_matrices(X, Y, w1, w2) #X is now length d+1
    
    m, d_1 = w1.shape #since w1 should have an extra row for the constant input
    d = d_1 - 1
    k, m_check = w2.shape
    if m+1 != m_check: raise AssertionError("Weight matrices improperly aligned")
    if not(X.shape == (d,1) or X.shape == (1,d)): raise AssertionError("Problem with X and w1 matrix")
    if not(Y.shape == (k,1) or Y.shape == (1,k)): raise AssertionError("Problem with Y with w2 matrix")
    
    #get matrices in expected shapes
    X = np.matrix(X).reshape(d,1)
    X = np.matrix(np.append(X,[[1]],axis=0))
    Y = np.matrix(Y).reshape(k,1)
    
    a1 = w1 * X # m x 1 matrix
    a1 = np.matrix(np.append(a1,[[0]],axis=0))
    z1 = sigmoid(a1) #a1, z1 are m+1 length matrices

    a2 = w2 * z1 # k x 1 matrix
    f = sigmoid(a2)
    
    #Calculate error
    loss = np.sum(-np.multiply(Y, np.log(f)) - np.multiply((1-Y), np.log(1-f)))
    
    return (X, Y, f, z1, a1, loss) 
'''
INPUTS
X: d by 1 array
Y: k by 1 array where k is number of classifications
w1: m x (d+1) matrix where m is the number of hidden units
w2: k x (m+1) matrix

OUTPUTS
w1_gradient: m x (d+1) matrix that represents gradient of J(w) with respect to w1
w2_gradient: k x (m+1) matrix that represents gradient of J(w) with respect to w2
'''
def backprop(X, Y, w1, w2, lamb):    
    (X, Y, f, z1, a1, loss) = feedforward(X, Y, w1, w2)
    
    # d_out are the deltas of the output units
    # d_hidden are the deltas of the hidden units
    d_out = f - Y
    delta_x_w2 = w2.T * d_out # (m+1) by 1 matrix
    d_hidden = np.multiply(sig_prime(a1), delta_x_w2) #(m+1) by 1 matrix
    
    w2_gradient = d_out * z1.T + 2*lamb*w2
    #delete last row since d_hidden includes delta of the constant units
    w1_gradient = np.delete(d_hidden * X.T, -1, axis=0) + 2*lamb*w1 
    
    return (w1_gradient, w2_gradient)

'''
We use Online Training of the Neural Net as opposed to Batch Training

INPUTS
X: d x n array
Y: k x n array
num_hidden: number of hidden nodes
w1_guess: num_hidden x (d+1) matrix for guesses of initial value for w1
w2_guess: k x (num_hidden+1) matrix for guesses of initial value for w2

OUTPUTS
w1: num_hidden x (d+1) matrix
w2: k x (num_hidden+1)
'''
def NN(X, Y, w1_guess, w2_guess, thresh, learning_rate, lamb):
    X = np.matrix(X)
    Y = np.matrix(Y)
    d,n = X.shape
    k,n_check = Y.shape
    num_hidden = w1_guess.shape[0]
    if n != n_check: raise AssertionError("X and Y shape doesn't match")
    if w1_guess.shape != (num_hidden,d+1): raise AssertionError("w1_guess is not " + str(num_hidden) + ", " + str(d))
    if w2_guess.shape != (k,num_hidden+1): raise AssertionError("w2_guess not specified properly")
    
    w1 = w1_guess
    w2 = w2_guess
    loss = 1
    num_iterations = 0
    while (loss > thresh) and (num_iterations) < 100:
        num_iterations += 1
        loss = 0
        for i in range(n):
            x = X.T[i]
            y = Y.T[i]
            w1_gradient, w2_gradient = backprop(x, y, w1, w2, lamb)
            w1 = w1 - learning_rate * w1_gradient
            w2 = w2 - learning_rate * w2_gradient
            loss += feedforward(x, y, w1, w2)[5]
        loss = loss / n
    return (w1, w2)
    
if __name__ == "__main__":
    name = "1"
    train = 'toy_multiclass_'+name+'_train.csv'
    A = None
    with open(train, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            row = [float(x) for x in row]
            if(A is None):
                A = np.matrix(row)
            else:
                A = np.concatenate((A,np.matrix(row)),axis=0)
    # use deep copy here to make cvxopt happy
    X = A[:, 0:2].copy()
    Y = A[:, 2:3].copy()
    
    #modify X and Y to be useable
    X = X.T
    Y_mod = []
    for i in Y:
        if i == 3:
            Y_mod.append([0,0,1])
        elif i == 2:
            Y_mod.append([0,1,0])
        elif i == 1:
            Y_mod.append([1,0,0])
        else:
            raise AssertionError("Invalid input")
    Y = np.matrix(Y_mod).T
    
    d, n = X.shape
    k, n_check = Y.shape
    hidden_nodes = 2
    w1 = np.random.rand(hidden_nodes,d+1)
    w2 = np.random.rand(k, hidden_nodes+1)
    thresh = .05
    learn_rate = 1
    lamb = 0.1
    z = NN(X,Y,w1,w2,thresh,learn_rate,lamb)