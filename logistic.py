'''
python 2.x
created on 02/23/2019 @ 06:54pm 
Creator @ anshulTalati 
subject : Logistic regression to train the model and classify the similiar input data 

'''
import math

# training data
D=[(170, 57, 32, 'W'), (192, 95, 28, 'M'), (150, 45, 30, 'W'), (170, 65, 29, 'M'), (175, 78, 35, 'M'), (185, 90, 32, 'M'), (170, 65, 28, 'W'), (155, 48, 31, 'W'), (160, 55, 30, 'W'), (182, 80, 30, 'M'), (175, 69, 28, 'W'), (180, 80, 27, 'M'), (160, 50, 31, 'W'), (175, 72, 30, 'M')]
m = len(D)
x = []
y = []

# Pre pocessing of the training data. 
for i in range(m):
    y.append(D[i][3])
    x.append(tuple( D[i][j] for j in range(3)))
A = y
y = [ w.replace('M', '1') if w == 'M'else w.replace('W', '0')  for w in y]
y = list(map(int, y))

# Sigmod function calculation for the hypothesis.
def sigMod(z):
    l = len(z)
    # sigmod =[]
    for i in range(1):
        sigmod = float(1.0/(1 + math.exp(-1*z[i])))
        # sigmod.append(func)
    return sigmod

# Hypothesis formulation for each Data Point 
def hypo(theta, x, m ):
    z=[]
    sumofhypo = theta[0]
    l= len(theta)
    for i in range(1, l):
        sumofhypo = sumofhypo + theta[i] * x[i-1]
    z.append(sumofhypo)
    return sigMod(z)

# Derivative function to calcualte the derivative for the gradient descent 
def derivative(x, y, theta,m, alpha, k  ):
    sumofE = 0 
    for i in range (m):
        xi = list(x[i])        
        hyp = hypo(theta, xi ,m )
        xi = [1] + xi
        xk =  xi[k]
        der = (hyp - y[i]) * xk 
        sumofE += der
    constant = float(alpha)/float(m)
    j = constant * sumofE
    return j

# To perform gradient descent optimisarion 
def gradient(x, y, theta, m, alpha):
    new_theta = []
    constant = alpha/m
    for j in range(len(theta)):
        CFDerivative = derivative(x , y, theta, m, alpha, j)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta


# To calculate the cost of regression so as to know if we are approaching the minima. 
def costOfRegression(x ,y, theta, m):
    summation = 0.0
    class1 = sum(-1.0 * y[i] * math.log( hypo( theta, x[i], m ) )  for i in range (m) if y[i] == 1)    
    class2 = sum(-1.0 * (1.0-y[i]) * math.log( 1 - hypo( theta, x[i], m ))  for i in range (m) if y[i] == 0) 
    summation = class1 + class2
    return summation 
    

# Perofrm the Logisitic regression on the data 
def logistic(X, Y, Alpha, Theta, m ):
    cost = cost = costOfRegression(X, Y , Theta, m )
    weight = Theta
    for i in range (3000):
        new_theta = gradient( X , Y, weight, m, Alpha)
        cost1 = costOfRegression(X, Y , new_theta, m )
        cost = cost1    
        weight = new_theta 
    return (prediction(weight, X))


# Dummy functin to help the comparision for the prediction function 
def compare(hyp):
    predy =[]
    if hyp >= 0.5:
        prediction = 'M'
    else:
        prediction = 'W'
    return prediction 

# To make prediction using the optimized theta value
def prediction(theta, x  ):
    predy=[]
    a= len(x)
    if a > 3:
        for i in range (a):
            hyp = hypo(theta, x[i],m)
            predy.append(compare(hyp))
    else:
        hyp = hypo(theta, x,m)
        predy = compare(hyp) 
    return predy, theta

# Initialization of Theta and alpha 
theta = [0.0, 0.0 , 0.0, 1.0 ]
alpha = 0.0001

# Showing the Training of the model 
print "Training the Logistic Regression Modeol for classification...........\n "
new = logistic(x, y, alpha, theta, m)
print "Training Data :" , A
print "Predicted Data:", new[0]

print '\n\n'

# Program Handler to take input data from the User.
while (True):
    input = raw_input("Please input a data point for the prediction as: height weight age OR -1 to Exit the Program \n")
    
    if input == '-1':
        break

    inputPoint = map(int, input.split())
    optimizedTheta = new[1]
    inputPoint =tuple(inputPoint)
    pred = prediction( optimizedTheta, inputPoint) 
    print "\nAccording to the model it is ", pred[0] 
    print "\n"