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
# momentum value decides how much of the old data do I wanna use for my momentum 
# Here, i'm using 90% of previous value 
op = tf.train.MomentumOptimizer(learning_rate=1e-5,use_nesterov=True,momentum=0.9).minimize(E) 

TrainErrors = []
TestErrors = []
epochs = int(1e4)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  #sess is a messanger btw python land and c++ back land
    #Training Loop:
    for i in range(epochs):
        _,Error,W1,W2,W3,PredTrain=sess.run([op,E,w1,w2,w3,Yhat],feed_dict={InX:xTrain,InY:yTrain})
        print("Iteration : {}, Error: {}".format(i+1,Error),end='\r')
        TrainErrors.append(Error/xTrain.shape[0]) #normalizing
        Error, PredTest = sess.run([E,Yhat],feed_dict={InX:xTest,InY:yTest})
        TestErrors.append(Error/xTest.shape[0])
    

plt.plot(TrainErrors,'-k') #black
plt.plot(TestErrors,'-r')
plt.legend(['Train Errors','Test Errors'])
plt.xlim([0,40])
plt.ylim([0,2])
plt.show()

#Residual plot for Test Data
plt.plot(yTest,yTest,'-r')
plt.scatter(yTest,PredTest)
plt.title('Test Residuals')
plt.show()

#Residual plot for Train Data
plt.plot(yTrain,yTrain,'-r')
plt.scatter(yTrain,PredTrain)
plt.title('Train Residuals')
plt.show()




