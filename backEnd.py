import random
import sys
import numpy as np
import pickle
import cPickle
import gzip
import cv2
import copy
from matplotlib import pyplot as plt
from sympy import *
from sympy.parsing.sympy_parser import *
import sympy
import os

#load test data from mnist data source, deprecated
def loadData():
    f=gzip.open("mnist.pkl.gz","rb")
    trainingData,validationData,testData=cPickle.load(f)
    f.close
    return (trainingData,validationData,testData)

#wrapper for loading the mnist data, deprecated
def loadDataWrapper():
    trainingData, validationData, testData = loadData()
    training_inputs = [np.reshape(x, (784, 1)) for x in trainingData[0]]
    training_results = [vectorizedResult(y) for y in trainingData[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in validationData[0]]
    validation_data = zip(validation_inputs, validationData[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in testData[0]]
    test_data = zip(test_inputs, testData[1])
    return (training_data, validation_data, test_data)

#vectorize the array fed in
def vectorizedResult(j):
        vector=np.zeros((10,1))
        vector[j]=1.0
        return vector

#calculates the results using the sigmoid transfer function
def sigmoid(z):
    sigmoid=1.0/(1.0+np.exp(-z))
    return sigmoid

sigmoidVector=np.vectorize(sigmoid)

#calculates sigmoid prime
def sigmoidPrime(z):
    sigmoidPrime=sigmoid(z)*(1-sigmoid(z))
    return sigmoidPrime

sigmoidPrimeVector=np.vectorize(sigmoidPrime)

class Network(object):

    #using cross entropy method to calculate cost during gradient descent
    @staticmethod
    def CrossEntropyCost(ouput,optimal):
        cost=np.nan_to_num(np.sum(-optimal*np.log(ouput)-
            (1-optimal)*np.log(1-output)))
        return cost

    #calculates the differnce between perfect cost and current cost
    @staticmethod
    def CrossEntropyDelta(output,optimal):
        return output-optimal

    #calculates difference
    @staticmethod
    def delta(a,y):
        return a-y    

    #initializes the neural net with provided specifications
    def __init__(self,sizes):
        self.layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(rows,1) for rows in self.sizes[1:]]
        for bias in self.biases: bias.astype(np.longdouble)
        self.weights=[np.random.randn(rows,cols)/np.sqrt(cols) for cols,rows in 
        zip(self.sizes[:-1],self.sizes[1:])]
        for weight in self.weights: weight.astype(np.longdouble)

    #feeds data forwards to neurons, eventually outputs a vector of probabilities
    def feedforward(self,output):
        for bias,weight in zip(self.biases,self.weights):
            output=sigmoidVector(np.dot(weight,output)+bias)
        return output

    #implements the stochastic gradient descent algorithm
    def stochasticGradientDescent(self,trainingData,epochs,sectionSize,rate,
        regularization=0.0,evaluationData=None,monitorEvaluationCost=False,
        monitorEvaluationAccuracy=False,monitorTrainingCost=False,
        monitorTrainingAccuracy=False):
        if evaluationData: dataNum=len(evaluationData)
        num=len(trainingData)
        evaluationCost,evaluationAccuracy=[],[]
        trainingCost,trainingAccuracy=[],[]
        for epoch in xrange(epochs):
            random.shuffle(trainingData)
            sections=([trainingData[start:start+sectionSize] 
                for start in xrange(0,num,sectionSize)])
            for section in sections:
                self.updateSection(section,rate,regularization,num)
            print "Epoch %d training complete"%epoch
            if monitorTrainingCost:
                cost=self.totalCost(trainingData,rate)
                trainingCost.append(cost)
                print "Cost on training data: %d"%cost
            if monitorTrainingAccuracy:
                accuracy=self.accuracy(trainingData,convert=True)
                trainingAccuracy.append(accuracy)
                print "Accuracy on training data: %d/%d"%(accuracy,num)
            if monitorEvaluationCost:
                cost=self.totalCost(evaluationData,rate,convert=True)
                evaluationCost.append(cost)
                print "Cost on evaluation data: %d"%cost
            if monitorEvaluationAccuracy:
                accuracy=self.accuracy(evaluationData)
                print "Accuracy on evaluation data: %d/%d"%(accuracy,dataNum)
        return evaluationCost,evaluationAccuracy,trainingCost,trainingAccuracy

    #updates the section batch that is about to be trained
    def updateSection(self,section,rate,regularization,num):
        biasGradients=[np.zeros(bias.shape) for bias in self.biases]
        weightGradients=[np.zeros(weight.shape) for weight in self.weights]
        for x,y in section:
            deltaBiasGradients,deltaWeightGradients=self.backProp(x,y)
            biasGradients=[biasGradient+deltaBiasGradient for biasGradient,
            deltaBiasGradient in zip(biasGradients,deltaBiasGradients)]
            weightGradients=[weightGradient+deltaWeightGradient for 
            weightGradient,
            deltaWeightGradient in zip(weightGradients,deltaWeightGradients)]
        self.weights=[(1-rate*(regularization/num))*weight-(rate/len(section)
            )*weightGradient for weight,weightGradient in zip(
            self.weights,weightGradients)]
        self.biases=[bias-(rate/len(section))*biasGradient for bias,biasGradient
        in zip(self.biases,biasGradients)]

    #implements the backpropagation algorithm
    def backProp(self,x,y):
        biasGradients=[np.zeros(bias.shape) for bias in self.biases]
        weightGradients=[np.zeros(weight.shape) for weight in self.weights]
        activation,zList=x,[]
        activations=[x]
        for bias,weight in zip(self.biases,self.weights):
            z=np.dot(weight,activation)+bias
            zList.append(z)
            activation=sigmoidVector(z)
            activations.append(activation)
        delta=Network.delta(activations[-1],y)
        biasGradients[-1]=delta
        weightGradients[-1]=np.dot(delta,activations[-2].transpose())
        for layer in xrange(2,self.layers):
            z=zList[-layer]
            delta=np.dot(self.weights[-layer+1].transpose(),
                delta)*sigmoidPrimeVector(z)
            biasGradients[-layer]=delta
            weightGradients[-layer]=np.dot(delta,
                activations[-layer-1].transpose())
        return (biasGradients,weightGradients)

    #calculates the accuracy of current data vs optimal data
    def accuracy(self,data,convert=False):
        if convert:
            results=[(np.argmax(self.feedforward(x)),np.argmax(y)) for (x,y)
             in data]
        else:
            results=[(np.argmax(self.feedforward(x)),y) for (x,y) in data]
        return sum(int(x==y) for (x,y) in results)

    #calculates current cost vs optimal cost
    def totalCost(self,data,rate,convert=False):
        cost=0.0
        for x,y in data:
            output=self.feedforward(x)
            if convert: y=vectorizedResult(y)
            cost+=0.5*(rate/len(data))*sum(np.linalg.norm(weight)**2 for weight 
                in self.weights)
        return cost

    #saves the entirety of the neural net into pickle file
    def save(self,filename):
        data={"sizes": self.sizes,
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases]}
        f=open(filename,"w")
        pickle.dump(data,f)
        f.close()

#loads the neural net from pickle file
def load(filename):
    f=open(filename,"r")
    data=pickle.load(f)
    f.close()
    net=Network(data["sizes"])
    net.weights=[np.array(weight) for weight in data["weights"]]
    net.biases=[np.array(bias) for bias in data["biases"]]
    return net

#used for sorting tuples via its second value
def getKey(item):
    return item[1]

#preprocess the given image, invert, blur, and increase contrast
def preProcess(cv2Image):
    image=cv2Image
    image=cv2.GaussianBlur(image,(5,5),0)
    image=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,11,2)
    image=cv2.bitwise_not(image)
    return image

#separates the different lines as well as creating the bounding boxes
def sepLines(rectListY):
    rectListY.sort()
    lineBreaks=[0]
    rectLines=[]
    doubles=[]
    for rectIndex in xrange(1,len(rectListY)):
        lowBound=sum([y+height for y,x,width,height in rectListY[lineBreaks[
            len(lineBreaks)-1]:rectIndex]])/len(rectListY[lineBreaks[len(
                lineBreaks)-1]:rectIndex])
        y,x,width,height=rectListY[rectIndex]
        yPrev,xPrev,widthPrev,heightPrev=rectListY[rectIndex-1]
        if y>lowBound:
            lineBreaks.append(rectIndex)
    for index in xrange(1,len(lineBreaks)):
        rectLines.append(rectListY[lineBreaks[index-1]:lineBreaks[index]])
    rectLines.append(rectListY[lineBreaks[len(lineBreaks)-1]:])
    for line in rectLines:
        line.sort(key=getKey)
        for index in xrange(len(line)-2,-1,-1):
            y1,x1,width1,height1=line[index]
            y2,x2,width2,height2=line[index+1]
            if (x1+x1+width1)/2>x2:
                line[index+1]=(min(y1,y2),min(x1,x2),max(width1,width2)+x2-x1,
                    max(y1+height1-y2,y2+height2-y1))
                line.pop(index)
    return rectLines,lineBreaks

#separate the image into little chunks containing the characters
def separate(image):
    temp=copy.deepcopy(image)
    temp,contours,hierarchy=cv2.findContours(temp,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    rectListY=[]
    for contour in contours:
        if cv2.contourArea(contour)>50:
            (x,y,width,height)=cv2.boundingRect(contour)
            rectListY.append((y,x,width,height))
    rectLines,lineBreaks=sepLines(rectListY)
    imageLines=copy.deepcopy(rectLines)
    for lineIndex in xrange(len(rectLines)):
        for rectIndex in xrange(len(rectLines[lineIndex])):
            y,x,width,height=rectLines[lineIndex][rectIndex]
            imageLines[lineIndex][rectIndex]=image[y:y+height,x:x+width]
    return (imageLines,lineBreaks,rectLines)

#further processes the chunks of images into regularized size as well
#as getting rid of noise
def postProcess(image):
    temp=copy.deepcopy(image)
    temp,contours,hierarchy=cv2.findContours(temp,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour)<30:
            cv2.drawContours(image,[contour],0,255,-1)
    image=cv2.GaussianBlur(image,(1,1),0)
    rows,cols=image.shape
    side=max(rows,cols)+30
    square=np.zeros((side,side),np.uint8)
    square[(side-rows)/2:rows+(side-rows)/2,(side-cols)/2:cols+(side-
        cols)/2]=image
    square=cv2.resize(square,(25,25))
    return square

#checks if there are any open parenthesis
def stillOpen(formula):
    count=0
    for sym in formula:
        if sym=="**(": count+=1
        elif sym==")": count-=1
    if count>0:
        return True
    else: return False

#finds superscript start and end lcoations
def findSuperUnder(formula,rectList):
    locations=[]
    for index in xrange(1,len(rectList)):
        y,x,width,height=rectList[index]
        yPrev,xPrev,widthPrev,heightPrev=rectList[index-1]
        if y+height<(yPrev+yPrev+heightPrev)/2 and y<yPrev:
            locations.append((index,"**("))
        elif y>(yPrev+yPrev+heightPrev)/2 or (y>yPrev and 
            y+height>yPrev+heightPrev+heightPrev/2):
            locations.append((index,")"))
    if len(locations)==0:
        return formula
    else:
        locations.sort(reverse=True)
        for index,symbol in locations:
            formula.insert(index,symbol)
        if stillOpen(formula):
            formula.append(")")
        return formula

#the OCR function proper that is given the image and the
#neural net and returns the recognized characters in lists    
def process(cv2Image,net):
    source=preProcess(cv2Image)
    imageLines,lineBreaks,rectLines=separate(source)
    indexList=list("0123456789abcd")
    indexList.append("exp")
    indexList.extend(list("fghijklmopqrstuvwxyz+-=()"))
    indexList.extend(["sqrt","/","Integral("," pi ",".","sum","n"])
    results=copy.deepcopy(imageLines)
    for lineIndex in xrange(len(imageLines)):
        lowBound=sum([y+height for y,x,width,height in rectLines[lineIndex]]
            )/len(rectLines[lineIndex])
        highBound=sum([y for y,x,width,height in rectLines[lineIndex]]
            )/len(rectLines[lineIndex])
        for imageIndex in xrange(len(imageLines[lineIndex])):
            image=postProcess(imageLines[lineIndex][imageIndex])
            imageVector=image.reshape(625,1)
            resultVector=net.feedforward(imageVector)
            y,x,width,height=rectLines[lineIndex][imageIndex]
            if y+height<=lowBound or y<highBound:
                resultVector[16]=0.
                resultVector[19]=0.
                resultVector[24]=0.
                resultVector[25]=0.
            if imageIndex>0 and results[lineIndex][imageIndex-1]=="d":
                resultVector[0]=0.
                resultVector[1]=0.
                resultVector[2]=0.
                resultVector[3]=0.
                resultVector[4]=0.
                resultVector[5]=0.
                resultVector[6]=0.
                resultVector[7]=0.
                resultVector[8]=0.
                resultVector[9]=0.
                resultVector[43]=0.
                resultVector[44]=0.
            if imageIndex>0 and results[lineIndex][imageIndex-1]=="integrate(":
                resultVector[41]=0
            resultVector[45]=0
            results[lineIndex][imageIndex]=indexList[np.argmax(resultVector)]
    for line in xrange(len(results)):
        results[line]=findSuperUnder(results[line],rectLines[line])
    #print results
    return results

#finds all instances of a character in a list or string
def findChar(s, char):
    return [i for i, letter in enumerate(s) if letter == char]

#special processing for inverse trig functions display
def inTrigPostProcess(latex):
    for index in sorted(findChar(latex,"a"),reverse=True):
        test=latex[index:]
        if test.find("atan")!=-1 or test.find("asin")!=-1 or test.find("acos"
            )!=-1 or test.find("acot")!=-1 or test.find("asec"
            )!=-1 or test.find("acsc")!=-1:
            latex=latex[:index]+latex[index+1:index+4]+"^{-1}"+latex[index+4:]
    return latex

#processing for trig functions
def trigProcess(line):
    for rIndex in xrange(len(line)-2,-1,-1):
        if line[rIndex]=="a" and line[rIndex+1]=="r" and line[rIndex+2]=="c":
            line.pop(rIndex+2)
            line.pop(rIndex+1)
    return line

#special processing for calculus functions
def CalcProcess(raw,answer):
    dLocation=sorted(findChar(answer,"d"),reverse=True)
    diff=False
    if "d" in answer:    
        for index in dLocation:
            answer=answer[:index+2]+[")"]+answer[index+2:]
            raw=raw[:index+2]+[")"]+raw[index+2:
            ] if "Integral(" in answer else raw[:index]+raw[index+2:]
        if "Integral(" in answer:
            answer=["integrate(" if ele=="Integral(" else ele for ele in answer]
            raw=["," if char=="d" else char for char in raw]
        else:
            answer=["diff("]+answer
            diff=True
        answer=["," if char=="d" else char for char in answer]
    return raw,answer,diff

#special processing for e
def naturalProcess(line):
    for index in xrange(1,len(line)):
        if line[index-1]=="exp":
            line[index]="("
    return line

#recieves the list of recognized characters and returns
#an image containing the expression printed in latex
def displaySolution(formulas):
    height=0.9
    done=False
    num=0
    for line in formulas:
        line=trigProcess(line)
        line=naturalProcess(line)
        raw=copy.deepcopy(line)
        answer=copy.deepcopy(line)
        raw,answer,diff=CalcProcess(raw,answer)
        raw="".join(raw)
        answer="".join(answer)
        transformation=(standard_transformations+(
            implicit_multiplication_application,))
        print raw
        raw=parse_expr(raw,evaluate=False,transformations=transformation)
        print answer
        answer=parse_expr(answer,transformations=transformation)
        print answer
        rawLatex=sympy.latex(raw)+"dx" if diff==True else sympy.latex(raw)
        answerLatex=sympy.latex(answer)
        rawLatex=inTrigPostProcess(rawLatex)
        answerLatex=inTrigPostProcess(answerLatex)
        result="$"+rawLatex+"="+answerLatex+"$"
        plt.text(0.05, height, r"%s"%result,fontsize=20)
        height-=0.1
    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    while os.path.exists("temp%d.png"%num):
        num+=1
    plt.savefig("temp%d.png"%num)
    plt.clf()
    return num

#function that generated the trainging data from scanned images of specific
#formatting
def trainingDat(filenames):
    recogNum=47
    sampleNum=44
    trainThumbnails=[]
    trainAnswers=[]
    testThhumbnails=[]
    testAnswers=[]
    for index in xrange(recogNum):
        for repeat in xrange(sampleNum):
            array=np.zeros(recogNum)
            array[index]=1.
            array=np.reshape(array,(recogNum,1))
            if repeat<40:
                trainAnswers.append(array)
            else:
                testAnswers.append(index)
    for filename in filenames:
        image=cv2.imread(filename,0)
        maxCol,maxRow=(1700,2200) if (filename!="Int-n.bmp" and 
            filename!="Int-n-2.bmp") else (900,2200)
        for col in xrange(0,maxCol,100):
            for row in xrange(0,maxRow,100):
                rectList=[]
                thumb=image[row:row+100,col:col+100]
                temp=copy.deepcopy(thumb)
                temp,contours,hierarchy=cv2.findContours(temp,cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour)>10:
                        (x,y,width,height)=cv2.boundingRect(contour)
                        rectList.append((y,x,width,height))
                if len(rectList)==0:
                    print col,row
                if len(rectList)>1:
                    rectList.sort()
                    (y1,x1,width1,height1)=rectList[0]
                    (y2,x2,width2,height2)=rectList[1]
                    (x,y,width,height)=(min(x1,x2),min(y1,y2),max(width1,width2
                        )+abs(x1-x2),max(height1,height2)+abs(y1-y2))
                else: (y,x,width,height)=rectList[0]
                thumb=thumb[y:y+height,x:x+width]
                thumb=postProcess(thumb)
                thumbnail=thumb.reshape(625,1)
                if row<2000:
                    trainThumbnails.append(thumbnail)
                else: testThhumbnails.append(thumbnail)
    trainSample=trainThumbnails[:recogNum*40]
    testSample=testThhumbnails[:recogNum*4]
    return (trainSample,testSample,trainAnswers,testAnswers)

#funciton that saves the generated training data
def saveData():
    (trainSample1,testSample1,trainAnswers1,testAnswers1)=trainingDat(
        ["0-8.bmp","8-g.bmp","h-q.bmp","q-y.bmp","z-Int.bmp","Int-n.bmp"])
    (trainSample2,testSample2,trainAnswers2,testAnswers2)=trainingDat(
        ["0-8-2.bmp","8-g-2.bmp","h-q-2.bmp","q-y-2.bmp","z-Int-2.bmp",
        "Int-n-2.bmp"])
    trainSample=trainSample1+trainSample2
    testSample=testSample1+testSample2
    trainAnswers=trainAnswers1+trainAnswers2
    testAnswers=testAnswers1+testAnswers2
    trainingData=zip(trainSample,trainAnswers)
    testData=zip(testSample,testAnswers)
    f=open("doubleDat","w")
    data={"training": trainingData,"test": testData}
    pickle.dump(data,f)
    f.close()

#loads the trainging data from file
def loadNewData():
    f=open("doubleDat","r")
    data=pickle.load(f)
    trainingData=data["training"]
    testData=data["test"]
    return trainingData,testData

#saving the recognized inputs to file as lists
def saveGrid():
    x,y=loadNewData()
    inputList=[]
    indexList=list("0123456789abcd")
    indexList.append("exp")
    indexList.extend(list("fghijklmopqrstuvwxyz+-=()"))
    indexList.extend(["sqrt","/","Integral("," pi ",".","sum","n"])
    for repeat in xrange(47):
        cv2.imshow("yay",x[repeat*40][0].reshape(25,25))
        cv2.waitKey(0)
        inputList.append(x[repeat*40][0].reshape(25,25))
    f=open("inputList","w")
    pickle.dump({"inputList":inputList,"indexList":indexList},f)
    f.close()

#loading the recognized inputs from file
def loadGrid():
    f=open("inputList","r")
    data=pickle.load(f)
    inputList=data["inputList"]
    indexList=data["indexList"]
    return inputList,indexList
