import numpy as np
import math
import csv
from scipy import special,  optimize
'''
INPUTS
X: d by 1 array
Y: k by 1 array where k is number of classifications
w1: m x d matrix where m is the number of hidden units
w2: k x m matrix
'''
name = "1"
train = 'toy_multiclass_'+name+'_train.csv'
#train = 'mnist_train.csv'
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

#X = A[:, 0:784].copy()
#Y = A[:, 784:785].copy()
possible_classes = 3
print A.shape
def feedforward(X, Y, w1, w2):

	def sigmoid(Z):
		return np.matrix([1 / (1.0 + math.exp(-z)) for z in Z]).T

	#add bias factor
	X = np.concatenate((X,np.ones((1,1))),axis = 0)
	z1 = w1 * X # m-length array
	o1 = sigmoid(z1)
	
	#add bias factor again
	o1 = np.concatenate((o1,np.ones((1,1))),axis = 0)
	z2 = w2 * o1 # k-length array
	o2 = sigmoid(z2)
	
	#dldo2 = np.matrix([-(Y[k,0] - o2[k,0]) for k in range(0,o2.shape[0])]).T
	dldo2 = o2 - Y
	
	#dldo2 = np.matrix([(Y[k,0]+o2[k,0])/(o2[k,0]*(o2[k,0]-1)) for k in range(0,o2.shape[0])]).T
	
	dldz2 = np.multiply(dldo2,np.multiply(o2,1-o2))
	dldw2 = np.zeros(w2.shape)
	
	#for i in range(0,dldw2.shape[0]):
	#	for j in range(0,dldw2.shape[1]):
        #           dldw2[i,j] = o1[j,0]*dldz2[i,0]

	dldw2 = dldz2 * o1.T

	dldo1 = np.zeros(o1.shape)
	#for i in range(0,o1.shape[0]):
	#	for j in range(0,dldz2.shape[0]):
	#		dldo1[i,0] += w2[j,i]*dldz2[j,0]

	dldo1 = w2.T*dldz2
	
	dldz1 = np.multiply(dldo1,np.multiply(o1,1-o1))			
		
	dldw1 = np.zeros(w1.shape)
	#for i in range(0,dldw1.shape[0]):
#   		for j in range(0,dldw1.shape[1]):
#			dldw1[i,j] = X[j,0]*dldz1[i,0]
        #print w1.shape
	dldw1 = np.delete(dldz1,-1,axis = 0) * X.T
	#print dldw1.shape	
	error = np.linalg.norm(Y-o2,2)

	return (o2,dldw1,dldw2,error)
    

def back_prop_train(X,Y,w1,w2,rate,y):
	error = 1;
	while(error > 0.05):
		o2,dldw1,dldw2,error = feedforward(X,Y,w1,w2)
		
		#regularization
		dldw1 += y*np.multiply(w1,w1)
		dldw2 += y*np.multiply(w2,w2)
		
		print error
		w1 = w1 - dldw1*rate
		w2 = w2 - dldw2*rate
		
	return w1,w2

def gradient_descent(X,Y,w1,w2,rate,lam):
	for i in range(0,400):
		dldw1 = np.zeros(w1.shape)
		dldw2 = np.zeros(w2.shape)
		error = 0
		for c in range(0,X.shape[0]):
			x = X[c,:].T
			y = np.zeros((possible_classes,1))
			y[Y[c,0]-1,0] = 1
			o2,d1,d2,e = feedforward(x,y,w1,w2)
			dldw1 += d1
			dldw2 += d2
			error += e
		
		dldw1 += lam*2*w1
		dldw2 += lam*2*w2
		#print dldw2
		
		w1 = w1 - dldw1*rate
		w2 = w2 - dldw2*rate
		print error_rate(X,Y,w1,w2)
		
	return w1,w2
	
def s_gradient_descent(X,Y,w1,w2,rate,lam):
	for i in range(0,100):
		error = 0
		for c in range(0,X.shape[0]):
			x = X[c,:].T
			y = np.zeros((possible_classes,1))
			y[Y[c,0]-1,0] = 1
			o2,d1,d2,e = feedforward(x,y,w1,w2)
			d1 += lam*2*w1
			d2 += lam*2*w2
			
			w1 = w1 - d1*rate
			w2 = w2 - d2*rate
			error += e
		print error_rate(X,Y,w1,w2)
		
	return w1,w2
	
def error_rate(X,Y,w1,w2):
	classes = 3
	error = 0
	for c in range(0,X.shape[0]):
		x = X[c,:].T
		y = np.zeros((possible_classes,1))
		y[Y[c,0]-1,0] = 1
		o2,d1,d2,e = feedforward(x,y,w1,w2)
		val = np.argmax(o2)
		if(val != Y[c,0]-1):
			error += 1
	return 1.0*error/(X.shape[0])

hidden = 2	
w1 = np.ones((hidden,X.shape[1]+1))
w2 = np.ones((possible_classes,hidden+1))
w1,w2 = gradient_descent(X,Y,w1,w2,0.04,0)
print error_rate(X,Y,w1,w2)

'''
hidden = 2
w1 = np.ones((hidden,X.shape[1]+1))
w2 = np.ones((possible_classes,hidden+1))
w1,w2 = s_gradient_descent(X,Y,w1,w2,0.1,0.0)
print error_rate(X,Y,w1,w2)
'''

'''
X = np.matrix("1;1")
Y = np.matrix("0;1")
w1 = 1*np.ones((2,3))
w2 = 1*np.ones((2,3))
print back_prop_train(X,Y,w1,w2,0.2,0)
'''
