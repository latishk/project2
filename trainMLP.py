__author__ = 'latish'
import numpy as np
import matplotlib.pyplot as plt
import math
import random

data            = np.genfromtxt('train_data.csv', delimiter=',')
symmetry        = data[:,0]
eccentricity    = data[:,1]
target          = data[:,2]
wHidden         = np.ones((2,5))
wOut            = np.ones((5,4))
out             = {
                    1. : [1,0,0,0],
                    2. : [0,1,0,0],
                    3. : [0,0,1,0],
                    4. : [0,0,0,1]
                  }
input           = np.array([0.,0.])
deltaHidden     = np.array([0.,0.,0.,0.,0.])
delta           = np.array([0.,0.,0.,0.])
numberOfEpochs  = 0
learningRate    = 0.1


def sigmoid(x):
    return  1/(1+math.exp(-x))

def updateWeight():

    global symmetry, eccentricity, target, wHidden, wOut, input, output, deltaHidden, delta, numberOfEpochs, learningRate, delta
    sumOfSquareDifferences = 0

    for i in range(target.size):

        hidden   = np.array([0.,0.,0.,0.,0.])
        output   = np.array([0.,0.,0.,0.])
        input[0] = symmetry[i]
        input[1] = eccentricity[i]

        for x in range(2):
            for y in range(5):
                hidden[y]+= wHidden[x][y] * input[x]
        # adding bais
        hidden+=1

        for h in range(hidden.size):
            hidden[h] = sigmoid(hidden[h])


        for x in range(5):
            for y in range(4):
                output[y]+= wOut[x][y] * hidden[x]

        # adding bais
        output+=1

        for o in range(output.size):
            output[o] = sigmoid(output[o])

        # print("output\n",output,"\n"," hidden",hidden)

        delta = np.array(np.array(out[target[i]]) - output)
        sumOfSquareDifferences+= sum(delta) ** 2


        for j in range(output.size):
            delta[j] *= (output[j] * (1 - output[j]))


        sigma = 0.

        for i in range(hidden.size):
            for j in range(delta.size):
                sigma += wOut[i][j]*delta[j]
            deltaHidden[i] = hidden[i]*(1-hidden[i]) * sigma
            sigma = 0.

        for i in range(5):
            for j in range(4):
                wOut[i][j] += learningRate * hidden[i] * delta[j]

        for i in range(2):
            for j in range(5):
                wHidden[i][j] += learningRate * input[i] * deltaHidden[j]


    return sumOfSquareDifferences



def testMLP():

    testData = np.genfromtxt('test_data.csv', delimiter=',')
    symmetry = testData[:,0]
    eccentricity = testData[:,1]
    target = testData[:,2]
    input  = [0.,0.]
    hidden = np.array([0.,0.,0.,0.,0.])
    output = np.array([0.,0.,0.,0.])
    global wHidden, wOut
    correct = 0
    print(target)

    for index in range(target.size):

        input[0] = symmetry[index]
        input[1] = eccentricity[index]

        for x in range(2):
            for y in range(5):
                hidden[y]+= wHidden[x][y] * input[x]
        # adding bais
        hidden+=1

        for iH in range(hidden.size):
            hidden[iH] = sigmoid(hidden[iH])

        # print("output\n", output)
        for xo in range(5):
            for yo in range(4):
                output[yo]+= wOut[xo][yo] * hidden[xo]

        # adding bais
        output+=1

        for ix in range(output.size):
            output[ix] = sigmoid(output[ix])
        print("here!!",output)

        max = -1
        outputValue = -1.
        for j in range(output.size):
            if output[j] >= max:
                outputValue = j+1.0
                print("output value is ",outputValue, "at index = ", index)
                max = output[j]

        if (outputValue == target[index]):
            # print(" output value",outputValue," target value", target[i])
            correct+=1
    print("percentage correctness",(correct/target.size))

def main():
    xaxis = []
    ssdList = []
    for i in range(10):
        ssdList.append(updateWeight())
        xaxis.append((i+1))
    plt.plot(xaxis,ssdList)
    plt.scatter(xaxis,ssdList)
    plt.show()



for x in range(2):
    for y in range(5):
            wHidden[x][y]= random.uniform(-1, 1)


for x in range(5):
    for y in range(4):
            wOut[x][y]= random.uniform(-1, 1)

main()
# testMLP()
