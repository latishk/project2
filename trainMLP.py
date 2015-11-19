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
                    1.: [1,0,0,0],
                    2.: [0,1,0,0],
                    3.: [0,0,1,0],
                    4.: [0,0,0,1]
                  }
input           = np.array([0.,0.])
deltaHidden     = np.array([0.,0.,0.,0.,0.])
delta           = np.array([0.,0.,0.,0.])
learningRate    = 0.1
biasHidden      = np.array([0.,0.,0.,0.,0.])
biasOutput      = np.array([0.,0.,0.,0.])


def sigmoid(x):
    return  1/(1+math.exp(-x))

def updateWeight():

    global symmetry, eccentricity, target, wHidden, wOut, input, output, deltaHidden, delta, numberOfEpochs, learningRate, delta
    global biasOutput, biasHidden
    deltaHidden     = np.array([0.,0.,0.,0.,0.])
    delta           = np.array([0.,0.,0.,0.])
    biasHidden      = np.array([0.,0.,0.,0.,0.])
    biasOutput      = np.array([0.,0.,0.,0.])


    sumOfSquareDifferences = 0

    for i in range(target.size):

        hidden   = np.array([0.,0.,0.,0.,0.])
        output   = np.array([0.,0.,0.,0.])
        input[0] = symmetry[i]
        input[1] = eccentricity[i]

        for x in range(2):
            for y in range(5):
                hidden[y] += wHidden[x][y] * input[x]

        hidden += biasHidden

        for h in range(hidden.size):
            hidden[h] = sigmoid(hidden[h])

        for x in range(5):
            for y in range(4):
                output[y] += wOut[x][y] * hidden[x]

        # adding bais
        output += biasOutput

        for o in range(output.size):
            output[o] = sigmoid(output[o])

        # print("output\n",output,"\n"," hidden",hidden)

        delta = np.array(np.array(out[target[i]]) - output)
        sumOfSquareDifferences += (sum(delta) ** 2)

        for j in range(output.size):
            delta[j] *= (output[j] * (1 - output[j]))

        sigma = 0.0
        for h in range(hidden.size):
            for j in range(delta.size):
                sigma += wOut[h][j]*delta[j]
            deltaHidden[h] = hidden[h]*(1 - hidden[h]) * sigma
            sigma = 0.

        for h in range(5):
            for j in range(4):
                wOut[h][j] += learningRate * hidden[h] * delta[j]

        for j in range(4):
            biasOutput[j] += learningRate * 1 * delta[j]

        for h in range(2):
            for j in range(5):
                wHidden[h][j] += learningRate * input[h] * deltaHidden[j]

        for j in range(5):
            biasHidden[j] += learningRate * 1 * deltaHidden[j]

    # print(" here!",output)
    return sumOfSquareDifferences



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


def reset():
    global wHidden, wOut, biasHidden, biasOutput

    for x in range(2):
        for y in range(5):
            wHidden[x][y] = random.uniform(-1, 1)

    for x in range(5):
        for y in range(4):
                wOut[x][y] = random.uniform(-1, 1)

    for w in biasHidden:
        w = np.random.uniform(-1, 1)

    for o in biasOutput:
        o = np.random.uniform(-1,1)


def writeTheWeightsToFile(fileName):
    global wHidden, wOut, biasHidden, biasOutput

    file = open(fileName,'w')
    line = ''
    for arr in wHidden:
        for weights in arr:
            line+= str(weights)  + ","

    for arr in wOut:
        for weights in arr:
            line+= str(weights)  + ","

    for weights in biasHidden:
        line+= str(weights)  + ","

    for iW in range(biasOutput.size - 1):
        line+= str(biasOutput[iW])  + ","

    line+=str(biasOutput[-1])
    file.write(line)
    line = ''
    file.close()

def main():
    epochs = [10000]
    xaxis = []
    ssdList = []

    for epoch in epochs:
        for i in range(epoch):
            # initializeWeights()
            ssdList.append(updateWeight())
            xaxis.append(i+1)
        writeTheWeightsToFile(""+str(epoch)+".csv")
        # testMLP('test_data.csv')
        reset()
        plt.plot(xaxis,ssdList)
        plt.scatter(xaxis,ssdList)
        plt.show()
        xaxis = []
        ssdList = []

e = 0
e += 1
main()
print(e,"\n")




