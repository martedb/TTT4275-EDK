import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###################################################################
#----------------  #Initializing variables#  ---------------------#
###################################################################



True_label_test = []
True_label_train = []

classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']                        #The classes    
features = ['sepal length','sepal width','petal length','petal width']              #The features
remove_features = []                                    #The features to be removed

#Dictonary for the features given the index they have in the datasets
class_features = {
    0 : 'sepal length',
    1 : 'sepal width',
    2 : 'petal length',
    3 : 'petal width'
}

nTrain = 30                             #Number of training samples
nTest = 20                              #Number of test samples
C= len(classes)                         #Number of classes 
D=len(features)                         #Number of features


X_train = np.zeros([3*nTrain, D])       #Initialising the X_train matrix, includes the train features
X_test = np.zeros([3*nTest, D])         #Initialising the X_test matrix, includes the test features
T_train = np.zeros([3*nTrain, C])       #Initialising the T_train matix, inxludes the labels corresponding to the X_train features
T_test = np.zeros([3*nTest, C])         #Initialising the T_test matrix, inxludes the labels corresponding to the X_test features

alpha = 0.01                            #Step factor
num_iterations = 1000                    #Number of iterations
#W = np.zeros([C,D])                    
#W = np.random.randn(C, D)
W = np.array(np.zeros([C,D]))           #CxD matrix 


samples30 = True                        #Is samples30 = True: the first 30 samples are used for training, if samples30=False the last 30 samples are used for training


###################################################################
#------------# Fetching the training and test samples #------------#
###################################################################

#Sorting the data from class_1.csv, class_2.csv and class_3.csv into a training set and a test set

with open('class_1.csv', 'r') as file:
    reader = csv.reader(file)
    if samples30 == True:                                       #Checks if the first 30 samples should be used for training
        for rownumber, row in enumerate(reader):
            if rownumber < 30:
                X_train[rownumber] = row
                T_train[rownumber] = [1, 0, 0]
                True_label_train.append(0)                       #True_label_train includes all the true labels corresponding to X_train
            else: 
                X_test[rownumber - 30] = row                     #The index [rownumber-30] ensures that the row gets placed at the start of the matrix, since rownumber is 30 for the first iteration of X_test
                T_test[rownumber - 30] = [1, 0, 0]      
                True_label_test.append(0)                        #True_label_test have all the true labels corresponding to X_test
    else:                                                        #If samples30 != True then the last 30 samples are used for training
        for rownumber, row in enumerate(reader):
            if rownumber < 20:
                X_test[rownumber] = row
                T_test[rownumber] = [1, 0, 0]
                True_label_test.append(0)
            else: 
                X_train[rownumber - 20] = row                   
                T_train[rownumber - 20] = [1, 0, 0]
                True_label_train.append(0)


with open('class_2.csv', 'r') as file:
    reader = csv.reader(file)
    if samples30 == True:
        for rownumber, row in enumerate(reader):
            if rownumber < 30:
                X_train[rownumber + nTrain] = row
                T_train[rownumber + nTrain] = [0, 1, 0]
                True_label_train.append(1)
            else: 
                X_test[rownumber - 30 + nTest] = row
                T_test[rownumber - 30 + nTest] = [0, 1, 0]
                True_label_test.append(1)
    else:
        for rownumber, row in enumerate(reader):
            if rownumber < 20:
                X_test[rownumber + nTest] = row
                T_test[rownumber + nTest] = [0, 1, 0]
                True_label_test.append(1)
            else: 
                X_train[rownumber - 20 + nTrain] = row
                T_train[rownumber - 20 + nTrain] = [0, 1, 0]
                True_label_train.append(1)



with open('class_3.csv', 'r') as file:
    reader = csv.reader(file)
    if samples30 == True:
        for rownumber, row in enumerate(reader):
            if rownumber < 30:
                X_train[rownumber + 2*nTrain] = row
                T_train[rownumber + 2*nTrain] = [0, 0, 1]
                True_label_train.append(2)
            else: 
                X_test[rownumber - 30 + 2*nTest] = row
                T_test[rownumber - 30 + 2*nTest] = [0, 0, 1]
                True_label_test.append(2)
    else:
        for rownumber, row in enumerate(reader):
            if rownumber < 20:
                X_test[rownumber + 2*nTest] = row
                T_test[rownumber + 2*nTest] = [0, 0, 1]
                True_label_test.append(2)
            else: 
                X_train[rownumber - 20 + 2*nTrain] = row
                T_train[rownumber - 20 + 2*nTrain] = [0, 0, 1]
                True_label_train.append(2)



'''
Checks to see if there are faetures that needs to be removed, if there are:
the strings in the remove_features array is checked against the class_features 
dictionary, and the matcing keys are added to the array matcing_keys. The columns
matcing the index of the matching_keys are removed from X_train and X_test. They 
are also removed from the features array and a column is removed from W
'''
matching_keys = []
for key, value in class_features.items():
    if value in remove_features:
        matching_keys.append(key)
X_train = np.delete(X_train, matching_keys, 1)
X_test = np.delete(X_test, matching_keys, 1)
features = np.delete(features, obj = matching_keys)
W = np.delete(W,matching_keys, 1)   



##################################################################
#------------# Functions for training and testing ##------------#
##################################################################

#----------------  MSE - MINIMUM SQUARE ERROR  -------------------#
# Eq. 19 from the compendium 
def MSE(g_k, t_k):
    return np.matmul(np.matrix.transpose(g_k-t_k), g_k-t_k)
#----------------------  GRADIENT OF MSE  ----------------------#
# Eq. 22 from the compendium 
def gradMSE(g, t, x):
    delta_MSE = g - t
    delta_g = np.multiply(g, 1 - g)
    delta_z = np.transpose(x)
    return np.dot(delta_z, delta_MSE * delta_g)


#----------------------   SIGMOID FUNCTION  ---------------------#
# Eq. 20 from the compendium 
def sigmoid(z):
    g_ik = 1/(1+np.exp(-z))
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

#------------------------- PLOTTING OF ERROR RATE  ---------------------------#

'''
Plot the error rate against different numbers of iterations. 
'''

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
    

'''
Plots the error rate against different values of alpha, the
alpha values are decided in alpha_values array
'''

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

'''
Plots the error rate for both different number of
iterations and alpha values
'''

def plot_errorRate_iterationsANDalpha(x, t):
    alpha_values = [0.0001, 0.001, 0.01]

    for i in range(len(alpha_values)):
        errorRateArray = []
        iterationArray = []
        for j in range(100):
            W_temp = np.zeros([C,D]) 
            stepSize = 100*j +100
            W_temp, g_temp = training(W_temp, x, t, stepSize, alpha_values[i-1])
            predTrain = predLabels(g_temp)
            errorRate_train = errorRate(predTrain, True_label_train)
            errorRateArray.append(errorRate_train)
            iterationArray.append(stepSize - 1)

        plt.plot(iterationArray, errorRateArray, label = 'alpha = ' + str(alpha_values[i-1]))
    plt.title('Error rate for different numbers of iterations and alpha values')
    plt.xlabel('Number of iterations')
    plt.ylabel('Error rate')
    plt.xscale('log')
    plt.xlim(left = 500)
    plt.legend()
    plt.show()
                



#------------------------  PLOTTING MSE   ------------------------#

def plot_MSEforN(x, t):
    alpha_values = [0.0001, 0.001, 0.01]

    for i in range(len(alpha_values)):
        mseArray = []
        iterationArray = []
        for j in range(100):
            W_temp = np.zeros([C,D]) 
            stepSize = 100*j +100
            W_temp, g_temp, mse_temp = trainingTest(W_temp, x, t, stepSize, alpha_values[i-1])
            
            mseArray.append(mse_temp)
            iterationArray.append(stepSize - 1)

        plt.plot(iterationArray, mseArray, label = 'alpha = ' + str(alpha_values[i-1]))
    plt.title('MSE for different numbers of iterations and alpha values')
    plt.xlabel('Number of iterations')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.xlim(left = 900)
    plt.ylim(0, 3)
    plt.legend()
    plt.show()


#------------------------  HISTOGRAMS   ------------------------#
'''
Plots histograms for all the features in all the classes, to be able
to find the features with the most overlap
'''
def plot_histograms(x, n):
    for j in range (len(classes)):
        rows = x[n*j: n + n*j, :]
        for i in range(len(features)):
            #plt.hist(rows[:,i], color='skyblue', bins = 10, edgecolor='black' )
            plt.hist(rows[:,i], edgecolor='black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Histogram for feature: ' + class_features[i] + ' and class: ' + classes[j])
            #plt.grid()
            plt.show()
 


#----------------------  PREDICTED LABELS  ----------------------#
def predLabels(g):
    predictedLabels = [np.argmax(sample) for sample in g]
    return predictedLabels

#------------------------    TRAINING    ----------------------#

def training(W, X, T, n, a):
    for i in range(n):
        z = np.matmul(X,np.transpose(W))
        g = sigmoid(z)
        temp_gradMSE = gradMSE(g, T, X)
        W = W - (a*(np.transpose(temp_gradMSE)))
    
    return W, g

'''
Only difference between training and trainingTest is
that trainingTest also calculates MSE and returns it
'''
def trainingTest(W, X, T, n, a):
    mse_t = 0
    for i in range(n):
        z = np.matmul(X,np.transpose(W))
        g = sigmoid(z)
        temp_gradMSE = gradMSE(g, T, X)
        W = W - (a*(np.transpose(temp_gradMSE)))
    for i in range(len(temp_gradMSE[0])):
        for j in range(len(temp_gradMSE[:, i])):
            mse_t += temp_gradMSE[i-1, j-1]
    return W, g, mse_t

#---------------------------  TESTING  ----------------------#

def test(W, X):
    z = np.dot(X, np.transpose(W))
    g = sigmoid(z)
    return g
    

#----------------------  SCATTER PLOT  ----------------------#
'''
When using scatter plot make shure that there are none features in the
remove_features array. Makes scatter plots of petal and sepal data,
so that it is possible to comment on linear separability. 
'''
def plot_scatter():
    rows_to_access1 = slice(0, 30)
    rows_to_access2 = slice(30, 60)
    rows_to_access3 = slice(60, 90)
    
    # Accessing the first column for each slice
    plt.scatter(X_train[rows_to_access1, 0], X_train[rows_to_access1, 1], label='Iris Setosa', color='blue')
    plt.scatter(X_train[rows_to_access2, 0], X_train[rows_to_access2, 1], label='Iris Versicolor', color='red')
    plt.scatter(X_train[rows_to_access3, 0], X_train[rows_to_access3, 1], label='Iris Virginica', color='purple')
    plt.title('Scatter plot for the features sepal width and sepal length')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Sepal width [cm]')
    plt.legend()
    plt.show()


    plt.scatter(X_train[rows_to_access1, 2], X_train[rows_to_access1, 3], label='Iris Setosa', color='blue')
    plt.scatter(X_train[rows_to_access2, 2], X_train[rows_to_access2, 3], label='Iris Versicolor', color='red')
    plt.scatter(X_train[rows_to_access3, 2], X_train[rows_to_access3, 3], label='Iris Virginica', color='purple')
    plt.title('Scatter plot for the features petal width and petal length')
    plt.xlabel('Petal length [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend()
    plt.show()


'''
Plots scatter plots for all the three classes, where the features in the classes
are plotted against eachother
'''

def plot_scatterClasses():
    rows_to_access1 = slice(0, 30)
    rows_to_access2 = slice(30, 60)
    rows_to_access3 = slice(60, 90)
    
    # Accessing the first column for each slice
    plt.scatter(X_train[rows_to_access1, 0], X_train[rows_to_access1, 1], label='Sepal', color='blue')
    plt.scatter(X_train[rows_to_access1, 2], X_train[rows_to_access1, 3], label='Petal', color='red')
    plt.title('Scatter plot for the features in Iris Setosa')
    plt.xlabel('Length [cm]')
    plt.ylabel('Width [cm]')
    plt.legend()
    plt.show()


    plt.scatter(X_train[rows_to_access2, 0], X_train[rows_to_access2, 1], label='Sepal', color='blue')
    plt.scatter(X_train[rows_to_access2, 2], X_train[rows_to_access2, 3], label='Petal', color='red')
    plt.title('Scatter plot for the features in Iris Versicolor')
    plt.xlabel('Length [cm]')
    plt.ylabel('Width [cm]')
    plt.legend()
    plt.show()

    plt.scatter(X_train[rows_to_access3, 0], X_train[rows_to_access3, 1], label='Sepal', color='blue')
    plt.scatter(X_train[rows_to_access3, 2], X_train[rows_to_access3, 3], label='Petal', color='red')
    plt.title('Scatter plot for the features in Iris Virginica')
    plt.xlabel('Length [cm]')
    plt.ylabel('Width [cm]')
    plt.legend()
    plt.show()
   



##########################################################
#------------------------# MAIN #------------------------#
##########################################################


def main(): 

    '''
    Training the classifier on the training set
    '''

    W_train, g_train, mse_train = trainingTest(W, X_train, T_train, num_iterations, alpha)
    predictedLabelsTrain = predLabels(g_train)
    confMatrixTrain = confusionMatrix(True_label_train, predictedLabelsTrain)


    errorRate_train = errorRate(predictedLabelsTrain, True_label_train)
    print('The error rate is ' + str(100* errorRate_train) + '%')
    

    '''
    Plots for the training:
    '''
    plot_confusionMatrix(True_label_train, predictedLabelsTrain, 'Confusion matrix for training set')
    #plot_histograms(X_train, nTrain)
    #plot_errorRate_iterationsANDalpha(X_train, T_train)
    #plot_errorRate_alpha(X_train, T_train, num_iterations)
    #plot_errorRate_iterations(X_train, T_train, alpha)
    #plot_MSEforN(X_train, T_train)
    #plot_scatterClasses()


    '''
    Testing the classifier on the test set
    '''

    g_test = test(W_train, X_test)
    predictedLabelsTest = predLabels(g_test)
    confMatrixTest = confusionMatrix(True_label_test, predictedLabelsTest)
    plot_confusionMatrix(True_label_test, predictedLabelsTest, 'Confusion matrix for test set')
    errorRate_test = errorRate(predictedLabelsTest, True_label_test)
    print('The error rate is ' + str(100* errorRate_test) + '%')
    
main()