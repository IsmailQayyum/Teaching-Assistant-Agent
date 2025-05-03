

import numpy as np

def sigmoid(x):
    return 1/(1 +np.exp(-x))

def sigmoidDiff(x):
    return x *(1- x)

def lossBCE(Y, A2):
    return -np.mean(Y*np.log(A2)+(1- Y)* np.log(1- A2))

#initialise weights and bias
np.random.seed(42)
W1=np.random.randn(3, 2)
b1= np.random.randn(3, 1)
W2= np.random.randn(1, 3)
b2 = np.random.randn(1, 1)

# sample valus for input an targt output
X= np.array([[0, 1], [1, 0], [1, 1], [0, 0]]).T
y = np.array([[1, 0, 1, 0]])

learnRate =0.5
iterations= 10


for i in range(iterations):
    # forward propogation
    Z1 = np.dot(W1, X) + b1

    A1 = sigmoid(Z1)
    #print(A1,  Z1)
    Z2= np.dot(W2, A1)+ b2
    A2 = sigmoid(Z2)

    # loss bce
    loss = lossBCE(y, A2)
    #print(loss)
    # back propogation
    dZ2= A2 -y
    dW2 = np.dot(dZ2, A1.T) /X.shape[1]
    #print("dw2: {dW2}")
    db2 =np.sum(dZ2, axis=1, keepdims=True)/X.shape[1]
    #dot product
    dA1= np.dot(W2.T, dZ2)
    dZ1= dA1 * sigmoidDiff(A1)
    #here X.shape is number o samples
  #were dividing by x.shape to get mean
    dW1 = np.dot(dZ1, X.T) /X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True)/ X.shape[1]

    # updat weights n bias
    W1 -= learnRate* dW1

    b1-=learnRate * db1
    W2-=learnRate*dW2

    b2 -= learnRate *db2
    #print("new b2: {b2}")

    print(f"Iteration {i}, Loss: {loss:.4f}")

#final output
print("Final Output:", A2)