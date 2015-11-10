import numpy as np
import math
import csv
from scipy import special,  optimize
import pylab as pl
'''
INPUTS
X: d by 1 array
Y: k by 1 array where k is number of classifications
w1: m x d matrix where m is the number of hidden units
w2: k x m matrix
'''
name = "2"
#train = 'toy_multiclass_'+name+'_train.csv'
train = 'mnist_train.csv'
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
print A.shape
X = A[:, 0:784].copy()
Y = A[:, 784:785].copy()
#X = A[:, 0:2].copy()
#Y = A[:, 2:3].copy()

#train = 'toy_multiclass_'+name+'_test.csv'
train = 'mnist_test.csv'
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

X_test = A[:, 0:784].copy()
Y_test = A[:, 784:785].copy()

#X = A[:, 0:784].copy()
#Y = A[:, 784:785].copy()

possible_classes = 6
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
	dldw2 = dldz2 * o1.T

	dldo1 = w2.T*dldz2
	
	dldz1 = np.multiply(dldo1,np.multiply(o1,1-o1))			
		
	dldw1 = np.delete(dldz1,-1,axis = 0) * X.T
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
        momentum = 0.3

        old_dldw1 =  np.zeros(w1.shape)
        old_dldw2 =  np.zeros(w2.shape)
        train_errors = []
        test_errors = []
	for i in range(0,200):
                #rate = max(2/(i+10)**(1.0/2),rate_in)
		dldw1 = old_dldw1 * momentum
		dldw2 = old_dldw2 * momentum
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

		old_dldw1 = dldw1
		old_dldw2 = dldw2
		#print dldw2
		
		w1 = w1 - dldw1*rate
		w2 = w2 - dldw2*rate
		
                train_er = error_rate(X,Y,w1,w2)
                #test_er = error_rate(X_test,Y_test,w1,w2)
                print train_er
                #train_errors.append(train_er)
                #test_errors.append(test_er)
                
		
	return w1,w2,train_errors,test_errors
	
def s_gradient_descent(X,Y,w1,w2,rate,lam):
        train_errors = []
        test_errors = []
        momentum = 0.3
        old_dldw1 =  np.zeros(w1.shape)
        old_dldw2 =  np.zeros(w2.shape)
	for i in range(0,200):
		error = 0
		for c in range(0,X.shape[0]):
                        
			x = X[c,:].T
			y = np.zeros((possible_classes,1))
			y[Y[c,0]-1,0] = 1
			o2,d1,d2,e = feedforward(x,y,w1,w2)
			d1 += old_dldw1 * momentum
                        d2 += old_dldw2 * momentum

                        old_dldw1 = d1
                        old_dldw2 = d2
                        
			d1 += lam*2*w1
			d2 += lam*2*w2
			
			w1 = w1 - d1*rate
			w2 = w2 - d2*rate
			error += e
		train_er = error_rate(X,Y,w1,w2)
                #test_er = error_rate(X_test,Y_test,w1,w2)
                print train_er
                #train_errors.append(train_er)
                #test_errors.append(test_er)
                
		
		
	return w1,w2,train_errors,test_errors
	
def error_rate(X,Y,w1,w2):
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




hidden = 10
w1 = np.ones((hidden,X.shape[1]+1))
w2 = np.ones((possible_classes,hidden+1))
w1,w2,train_e,test_e = s_gradient_descent(X,Y,w1,w2,0.1,0)



xvals = range(1,15)
e = []
'''
for i in xvals:
    hidden = i
    w1 = np.ones((hidden,X.shape[1]+1))
    w2 = np.ones((possible_classes,hidden+1))
    w1,w2,errors,_ = s_gradient_descent(X,Y,w1,w2,0.1,0)
    er = error_rate(X_test,Y_test,w1,w2)
    print er
    e.append(er)
'''

#batch
#e = [0.37, 0.33666666666666667, 0.27666666666666667, 0.27666666666666667, 0.2733333333333333, 0.2733333333333333, 0.25333333333333335, 0.25333333333333335, 0.24333333333333335]
#e += [0.24333333333333335, 0.25666666666666665, 0.25333333333333335, 0.30333333333333334, 0.6666666666666666]
#e = [0.41, 0.4033333333333333, 0.4066666666666667, 0.4066666666666667, 0.4066666666666667, 0.41, 0.4066666666666667, 0.4033333333333333, 0.4033333333333333, 0.3933333333333333, 0.39, 0.37666666666666665, 0.6666666666666666, 0.6666666666666666]

#print e

#pl.plot(xvals,e)
#pl.axis([1,10,0.3,0.5])
#pl.show()



pl.plot(range(1,len(train_e)+1),train_e)
pl.plot(range(1,len(test_e)+1),test_e)
pl.show()

'''
def score(x):
    x = np.matrix(x).T
    #print x.shape
    o2,_,_,_ = feedforward(x, np.zeros((possible_classes,1)), w1, w2)
    return np.argmax(o2)

xx,yy = np.meshgrid(np.arange(-1.5,1,0.01),np.arange(-1.5,1.5,0.01))
zz = np.array([score(x) for x in np.c_[xx.ravel(),yy.ravel()]])
zz = zz.reshape(xx.shape)
pl.figure()
pl.scatter(X[:,0],X[:,1],c =np.array(Y),cmap = pl.cm.cool,s=50)
pl.contour(xx,yy,zz,[0.5,1.5],colors = 'green',linestyles = 'solid', linewidths = 2)
pl.show()
#print error_rate(X,Y,w1,w2)
'''
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
