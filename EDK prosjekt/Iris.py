import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

X_train =[]                #Training input data
X_test = []                #Test input data
T_test =  []               #Test target data
T_train = []               #Train target data

True_label_test = []
True_label_train = []

classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
features = ['sepal length','sepal width','petal length','petal width']


nTrain = 30
nTest = 20
C= len(classes)                         #Number of classes 
D=len(features)                         #Number of features


alpha = 0.01                            #Step factor
num_iterations = 1000                   #Number of iterations
W = np.zeros([C,D])                     #CxD matrix
w_0 = np.zeros(C)                       #w_o

samples30 = True


###################################################################
###################################################################

#Sorting the data from class_1.csv, class_2.csv and class_3.csv into a training set and a test set
with open('class_1.csv', 'r') as file:
    reader = csv.reader(file)
    if samples30 == True:
        for rownumber, row in enumerate(reader):
            if rownumber < 30:
                X_train.append(row)
                T_train.append([1, 0, 0])
                True_label_train.append(0)
            else: 
                X_test.append(row)
                T_test.append([1, 0, 0])
                True_label_test.append(0)
    else:
        for rownumber, row in enumerate(reader):
            if rownumber < 20:
                X_test.append(row)
                T_test.append([1, 0, 0])
                True_label_test.append(0)
            else: 
                X_train.append(row)
                T_train.append([1, 0, 0])
                True_label_train.append(0)


with open('class_2.csv', 'r') as file:
    reader = csv.reader(file)
    if samples30 == True:
        for rownumber, row in enumerate(reader):
            if rownumber < 30:
                X_train.append(row)
                T_train.append([0, 1, 0])
                True_label_train.append(1)
            else: 
                X_test.append(row)
                T_test.append([0, 1, 0])
                True_label_test.append(1)
    else:
        for rownumber, row in enumerate(reader):
            if rownumber < 20:
                X_test.append(row)
                T_test.append([0, 1, 0])
                True_label_test.append(1)
            else: 
                X_train.append(row)
                T_train.append([0, 1, 0])
                True_label_train.append(1)


with open('class_3.csv', 'r') as file:
    reader = csv.reader(file)
    if samples30 == True:
        for rownumber, row in enumerate(reader):
            if rownumber < 30:
                X_train.append(row)
                T_train.append([0, 0, 1])
                True_label_train.append(2)
            else: 
                X_test.append(row)
                T_test.append([0, 0, 1])
                True_label_test.append(2)
    else:
        for rownumber, row in enumerate(reader):
            if rownumber < 20:
                X_test.append(row)
                T_test.append([0, 0, 1])
                True_label_test.append(2)
            else: 
                X_train.append(row)
                T_train.append([0, 0, 1])
                True_label_train.append(2)



X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)

T_train = np.array(T_train, dtype=float)
T_test = np.array(T_test, dtype=float)

###################################################################
###################################################################

#----------------  MSE - MINIMUM SQUARE ERROR  -------------------#
# Eq. 19 from the compendium 
def MSE(g_k, t_k):
    return np.matmul(np.matrix.transpose(g_k-t_k), g_k-t_k)


#----------------------  GRADIENT OF MSE  ----------------------#
# Eq. 22 from the compendium 
def gradMSE(g_k, t_k, x_k):
    x_k = np.array(x_k, dtype=float)
    firstE = g_k - t_k
    secondE = g_k*(1 - g_k)
    thirdE = np.transpose(x_k)
    #tempGradMSE =  np.dot(thirdE, firstE*secondE)
    tempGradMSE = np.outer( np.multiply(np.multiply(firstE, g_k), secondE), np.transpose(x_k) )
    return tempGradMSE


def gradMSETest(g, t, x):
    delta_MSE = g - t
    delta_g = np.multiply(g, 1 - g)
    delta_z = np.transpose(x)
    return np.dot(delta_z, delta_MSE * delta_g)


#----------------------   SIGMOID FUNCTION  ---------------------#
# Eq. 20 from the compendium 
def sigmoid(z):
    g_ik = 1.0/(1.0+np.exp(-z))
    return g_ik


#-------------------------  CONFUSION MATRIX  -------------------------#
def confusionMatrix(trueLabels, predictetLabels):
    comfMatrix = confusion_matrix(trueLabels, predictetLabels, labels = [0, 1, 2])
    return comfMatrix


#----------------------  PLOTTING CONFUSION MATRIX  ----------------------#
def plot_confusionMatrix(trueLabels, predictetLabels, title):
    disp = ConfusionMatrixDisplay.from_predictions(trueLabels, predictetLabels, display_labels=classes, cmap = plt.colormaps['Blues'])
    #disp.plot()
    disp.ax_.set_title(title)
    plt.show()


#-------------------------  ERROR RATE  ---------------------------#
def errorRate(predLabels, trueLabels):
    numOfHits = 0
    for i in range(len(predLabels)):
        if predLabels[i] == trueLabels[i]:
            numOfHits += 1
    accuracy = numOfHits/len(predLabels)
    error = 1 - accuracy
    return error

def plot_errorRate_iterations(x, t, a):
    errorRateArray = []
    iterationArray = []
    for i in range(100):
        W_1 = np.zeros([C,D]) 
        stepSize = 10*i +100
        W_i, g_i = training(W_1, x, t, stepSize, a)
        predTrain = predLabels(g_i)
        errorRate_train = errorRate(predTrain, True_label_train)
        errorRateArray.append(errorRate_train)
        iterationArray.append(stepSize - 1)
    
    plt.plot(iterationArray, errorRateArray)
    plt.title('Error rate for different numbers of iterations with alpha=' + str(a))
    plt.xlabel('Number of iterations')
    plt.ylabel('Error rate')
    plt.show()
    
def plot_errorRate_alpha(x, t, n):
    errorRateArray = []
    iterationArray = []
    alpha_values = [0.0001, 0.001, 0.01, 0.1, 1]
    for i in range(len(alpha_values)):
        W_1 = np.zeros([C,D]) 
        W_i, g_i = training(W_1, x, t, n, alpha_values[i])
        predTrain = predLabels(g_i)
        errorRate_train = errorRate(predTrain, True_label_train)
        errorRateArray.append(errorRate_train)
        iterationArray.append(alpha_values[i])
    
    plt.plot(iterationArray, errorRateArray)
    plt.title('Error rate for different values of alpha with n=' + str(n))
    plt.xlabel('Value of alpha')
    plt.ylabel('Error rate')
    plt.xscale('log')
    plt.show()

#---------------------------  HISTOGRAM  ----------------------#
#def historgram():


#----------------------  PREDICTED LABELS  ----------------------#
def predLabels(g):
    predictedLabels = [np.argmax(sample) for sample in g]
    return predictedLabels


#------------------------    TRAINING    ----------------------#

def training(W, X, T, n, a):
    for i in range(n):
        z = np.dot(X,np.transpose(W))
        g = sigmoid(z)
        temp_gradMSE = gradMSETest(g, T, X)
        W = W - a*np.transpose(temp_gradMSE)
    return W, g


#---------------------------  TESTING  ----------------------#

def test(W, X):
    z = np.dot(X, np.transpose(W))
    g = sigmoid(z)
    return g


###################################################################
###################################################################


W_train, g_train = training(W, X_train, T_train, num_iterations, alpha)
predictedLabelsTrain = predLabels(g_train)
confMatrixTrain = confusionMatrix(True_label_train, predictedLabelsTrain)
plot_confusionMatrix(True_label_train, predictedLabelsTrain, 'Confusion matrix for training set')
errorRate_train = errorRate(predictedLabelsTrain, True_label_train)
#plot_errorRate_iterations(X_train, T_train, alpha)
plot_errorRate_alpha(X_train, T_train, num_iterations)
print(errorRate_train)

g_test = test(W_train, X_test)
predictedLabelsTest = predLabels(g_test)
confMatrixTest = confusionMatrix(True_label_test, predictedLabelsTest)
plot_confusionMatrix(True_label_test, predictedLabelsTest, 'Confusion matrix for test set')