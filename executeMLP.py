__author__ = 'latish'
import numpy as np
import matplotlib.pyplot as plt
import math
import random


input           = np.array([0.,0.])
deltaHidden     = np.array([0.,0.,0.,0.,0.])
delta           = np.array([0.,0.,0.,0.])
learningRate    = 0.1
wHidden         = np.ones((2,5))
wOut            = np.ones((5,4))
biasHidden      = np.array([0.,0.,0.,0.,0.])
biasOutput      = np.array([0.,0.,0.,0.])


wHidden, wOut, biasHidden, biasOutput

def sigmoid(x):
    return  1/(1+math.exp(-x))

def readWeights(fileName):
    testData        = np.genfromtxt(fileName, delimiter=',')
    symmetry        = np.array(testData)
    # print(symmetry.size)
    count = 0
    for i in range(2):
        for j in range(5):
            wHidden[i][j] = symmetry[count]
            count+=1
    print(wHidden,"\n",count)

    for i in range(5):
        for j in range(4):
            wOut[i][j] = symmetry[count]
            count+=1
    print(wOut)

    for i in range(biasHidden.size):
        biasHidden[i] = symmetry[count]
        count+=1

    for i in range(biasOutput.size):
        biasOutput[i] = symmetry[count]
        count+=1
    print(biasHidden,"\n out \n",count, biasOutput)

def testMLP(name):

    testData        = np.genfromtxt(name, delimiter=',')
    symmetry        = testData[:,0]
    eccentricity    = testData[:,1]
    target          = testData[:,2]
    input           = [0.,0.]
    correct         = 0
    global wHidden, wOut, biasHidden, biasOutput

    print(wHidden,"\n\n", wOut)
    print(target)

    for index in range(target.size):

        hidden      = np.array([0.,0.,0.,0.,0.])
        output      = np.array([0.,0.,0.,0.])
        input[0]    = symmetry[index]
        input[1]    = eccentricity[index]

        for x in range(2):
            for y in range(5):
                hidden[y]+= wHidden[x][y] * input[x]

        hidden += biasHidden

        for iH in range(hidden.size):
            hidden[iH] = sigmoid(hidden[iH])

        for xo in range(5):
            for yo in range(4):
                output[yo] += wOut[xo][yo] * hidden[xo]

        output += biasOutput

        for ix in range(output.size):
            output[ix] = sigmoid(output[ix])

        max = -1.0
        outputValue = -1.0

        for j in range(output.size):
            if output[j] >= max:
                outputValue = j+1.0
                max = output[j]

        if outputValue == target[index]:
            correct+=1

    print("percentage correctness",(correct/target.size)*100)

readWeights('10.csv')
# testMLP('test_data.csv')