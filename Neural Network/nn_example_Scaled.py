import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split  #split boston data 
from sklearn.preprocessing import MinMaxScaler

# Load up some Data
X,Y = load_boston(return_X_y=True)



# Add a column of ones to the features
X = np.hstack((np.ones((X.shape[0],1)),X))

# Make sure the output is 2-dimensinal
Y = np.expand_dims(Y,1)

# Scale Data in {0,1}
XScaler = MinMaxScaler().fit(X)
X = XScaler.transform(X)
YScaler = MinMaxScaler().fit(Y)
Y = YScaler.transform(Y)

#Split into train and test
xTrain,xTest,yTrain,yTest = train_test_split(X,Y)

#Check distribution of Scaled Y
plt.hist(Y)


## Let's define a computational graph for a neural network with 2 hidden layers and Mean Squared Error Loss. Define an operation to minimize the loss by using a Gradient Descent Optimizer
# Define parameters 
w1 = tf.Variable(tf.random_normal([14,14]))
w2 = tf.Variable(tf.random_normal([14,14]))
w3 = tf.Variable(tf.random_normal([14,1]))

# Define Input node 
InX = tf.placeholder(dtype=tf.float32,shape=[None,14])
InY = tf.placeholder(dtype=tf.float32,shape=[None,1])

# Define Operations
L1 = tf.tanh(tf.matmul(InX,w1))     #Layer1  InX means Input X 
L2 = tf.tanh(tf.matmul(L1,w2))      #Layer2
Yhat = tf.matmul(L2,w3)            

# Define Error
E = tf.reduce_sum(tf.square(InY-Yhat))

# Define Operation to reduce error
op = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(E)  # train is Tensorflow Gradient Descent Optimizer

Errors = []
epochs = int(1e5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #sess is a messanger btw python land and c++ back land
    #Training Loop:
    for i in range(epochs):
        _,Error,W1,W2,W3=sess.run([op,E,w1,w2,w3],feed_dict={InX:xTrain,InY:yTrain})
        print("Iteration : {}, Error: {}".format(i+1,Error),end='\r')
        Errors.append(Error)

# Output = Iteration : 100000, Error: 29275.28906255


plt.plot(Errors)
plt.xlim([-10,int(1e4)])
plt.ylim([0,100])
plt.show()

