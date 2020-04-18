from __future__ import print_function
import sys
import re
from operator import add
import numpy as np 
from pyspark import SparkContext
from datetime import datetime

if __name__ == "__main__":

	sc = SparkContext(appName="Assignment5-task2")

	# Note down the current time for calculation of time
	startingTime = datetime.now()

	#########################################################################################
	#################### 1. Read the training dataset and preprocess it  ####################
	#########################################################################################

	# Read the training dataset 
	d_corpus = sc.textFile(sys.argv[1])

	# Each entry in validLines will be a line from the text file
	validDocLines = d_corpus.filter(lambda x : 'id' in x and 'url=' in x)

	# Now, we transform it into a set of (docID, text) pairs
	keyAndText = validDocLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 

	# leveraged the code from assignment 2
	# remove all non letter characters
	regex = re.compile('[^a-zA-Z]')
	keyAndWordsList = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

	# Get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
	# to ("word1", 1) ("word2", 1)...
	conslidatedWords = keyAndWordsList.flatMap(lambda x: x[1]).map(lambda x: (x,1))

	# Count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
	allCounts = conslidatedWords.reduceByKey(add)

	# Get the top 20,000 words in a local array in a sorted format based on frequency
	topWordsinDict = allCounts.top(20000, key = lambda x : x[1])

	# We'll create a RDD that has a set of (word, dictNum) pairs
	# start by creating an RDD that has the number 0 through 20000
	# 20000 is the number of words that will be in our dictionary
	top20000Words = sc.parallelize(range(20000))

	# Now, we transform (0), (1), (2), ... to ("MostCommonWord", 1)
	# ("NextMostCommon", 2), ...
	# the number will be the spot in the dictionary used to tell us
	# where the word is located
	dictionary = top20000Words.map (lambda x : (topWordsinDict[x][0], x))

	# The following function gets a list of dictionaryPos values,
	# and then creates a TF vector
	# corresponding to those values... for example,
	# if we get [3, 4, 1, 1, 2] we would in the
	# end have [0, 2/5, 1/5, 1/5, 1/5] because 0 appears zero times,
	# 1 appears twice, 2 appears once, etc.

	def buildArray(listOfIndices):
	    
	    returnVal = np.zeros(20000)
	    
	    for index in listOfIndices:
	        returnVal[index] = returnVal[index] + 1
	    
	    mysum = np.sum(returnVal)
	    
	    returnVal = np.divide(returnVal, mysum)
	    
	    return returnVal
	    
	# Next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...
	allWordsWithDocID = keyAndWordsList.flatMap(lambda x: ((j, x[0]) for j in x[1]))

	# Now join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
	allDictionaryWords = dictionary.join(allWordsWithDocID)

	# Now, we drop the actual word itself to get a set of (docID, dictionaryPos) pairs
	justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1],x[1][0]))

	# Now get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()

	# The following line this gets us a set of
	# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# and converts the dictionary positions to a bag-of-words numpy array...
	allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1])))

	# Now, create a version of allDocsAsNumpyArrays where, in the array,
	# every entry is either zero or one.
	# A zero means that the word does not occur,
	# and a one means that it does.
	zeroOrOne = allDocsAsNumpyArrays.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))

	# Function to generate labels for each document - Document with AU id --> 1, else --> 0
	def getLabel(x):
	  if x[:2] == 'AU':
	    return 1

	  else:
	    return -1

	# Generate Labeled Points containing - x[0] -> label  and  x[1] -> 20000 features
	yLabelAndXFeatures = zeroOrOne.map(lambda x: (getLabel(x[0]),x[1]))
	trainingData = yLabelAndXFeatures

	# Count number of rows
	n = float( trainingData.count())

	# Cache the RDD containing labels and features extracted out of training data
	trainingData.cache()
	# End of preprocessing of training data

	# 
	TrainingDatasetetPreprocessingCompletionTime = datetime.now()
	timeToReadTrainDataAndPreProcess = TrainingDatasetetPreprocessingCompletionTime - startingTime


	#######################################################################################
	#################### 3. Train the Logistic Regression Model  ##########################
	#######################################################################################

	# Initialize the different variables
	numberOfFeatures = 20000
	learningRate = 0.001
	coefficients = np.zeros(numberOfFeatures)
	gradients = np.zeros(numberOfFeatures)
	totalNumberOfIterations = 400
	currentIteration = 0
	listOfCost = []
	oldCost = float("inf")
	cRegularisationCoefficient = n/10000000000000000
	intercept = 0

	while (currentIteration < totalNumberOfIterations):
		# update cost
		cost = (float(1)/n)*trainingData.map(lambda x: (1, max(float(0), 1-x[0]*(np.dot(coefficients, x[1])-intercept)))).reduceByKey(np.add).collect()[0][1]
		cost += (float(1)/float(2)*n*cRegularisationCoefficient)*((np.linalg.norm(coefficients))**2)
		
		# Update Gradients
		gradients = (float(1)/n)*trainingData.map(lambda x: (1,(0 if (x[0]*(np.dot(coefficients, x[1])-intercept)) >= float(1) else -np.dot(x[0],x[1])))).reduceByKey(np.add).collect()[0][1]
		# gradients += (float(1)/n*cRegularisationCoefficient)*(coefficients)
		gradients += (float(2)/n*cRegularisationCoefficient)*(coefficients)

		# Update intercept
		interceptGradient = (float(1)/n)*trainingData.map(lambda x: (1,(0 if (x[0]*(np.dot(coefficients, x[1])-intercept)) >= float(1) else x[0]))).reduceByKey(np.add).collect()[0][1]
		# intercept = gradients = (float(1)/n)*trainingData.map(lambda x: (1,(0 if (x[0]*(np.dot(coefficients, x[1])-intercept)) >= float(1) else x[0]))).reduceByKey(np.add).collect()[0][1]

		# Update Parameters
		coefficients -= learningRate*gradients
		intercept -= learningRate*interceptGradient

		print('#'*10, 'Iteration', currentIteration + 1,'#'*10)
		print('Cost:', cost)
		print('Intercept:',intercept)
		print('Coefficients:', coefficients)
		print('Margin:',2/np.linalg.norm(coefficients),'\n')
		# BOLD DRIVER
		if (oldCost > cost):
			learningRate *= 1.05
		else:
			learningRate *= 0.5



		listOfCost.append(cost)

		######### Early stop
		if oldCost-cost < 0.00001:
			print('Training stopped at iteration', currentIteration + 1)
			break
		##############
		oldCost = cost

		currentIteration += 1

		# print(cost)


	# # Calculating the time for training the model
	trainingCompletionTime = datetime.now()
	trainingTime = trainingCompletionTime - TrainingDatasetetPreprocessingCompletionTime


	#######################################################################################
	#################### 2. Read the testing dataset and preprocess it ####################
	#######################################################################################

	# Read the dataset 
	testData = sc.textFile(sys.argv[2])

	# Each entry in validLines will be a line from the text file
	validDocLinesTest = testData.filter(lambda x : 'id' in x and 'url=' in x)

	# Now, we transform it into a set of (docID, text) pairs
	keyAndTextTest = validDocLinesTest.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6])) 

	# remove all non letter characters
	keyAndWordsListTest = keyAndTextTest.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

	# Get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
	# ("word1", docID), ("word2", docId), ...
	allWordsWithDocIDTest = keyAndWordsListTest.flatMap(lambda x: ((j, x[0]) for j in x[1]))

	# Join and link them, to get a set of ("word1", (dictionaryPos, docID)) pairs
	allDictionaryWordsTest = dictionary.join(allWordsWithDocIDTest)

	# Drop the actual word itself to get a set of (docID, dictionaryPos) pairs
	justDocAndPosTest = allDictionaryWordsTest.map(lambda x: (x[1][1],x[1][0]))

	# Get a set of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	allDictionaryWordsInEachDocTest = justDocAndPosTest.groupByKey()

	# The following line this gets us a set of
	# (docID,  [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
	# and converts the dictionary positions to a bag-of-words numpy array...
	allDocsAsNumpyArraysTest = allDictionaryWordsInEachDocTest.map(lambda x: (x[0], buildArray(x[1])))

	# Now, create a version of allDocsAsNumpyArrays where, in the array,
	# every entry is either zero or one.
	# A zero means that the word does not occur,
	# and a one means that it does.
	zeroOrOneTest = allDocsAsNumpyArraysTest.map(lambda x: (x[0],np.where(x[1] > 0, 1, 0)))

	# Create a RDD of testing data and derive features and labels ... x[0]-> label, x[1]-> features
	yLabelAndXFeatures = zeroOrOneTest.map(lambda x: (getLabel(x[0]),x[1]))
	testingData = yLabelAndXFeatures

	# Cache the RDD containing labels and features extracted out of testing data
	#testingData.cache()

	# Calculating the time for reading and preprocessing the testing data
	TestingDatasetetPreprocessingCompletionTime = datetime.now()
	timeToReadTestDataAndPreProcess = TestingDatasetetPreprocessingCompletionTime - trainingCompletionTime



	###################################################################################################
	################### 4. Predict the labels using the Logistic Regression Model  ####################
	###################################################################################################

	# Prediction Function using SVM
	def predictionSVM(x):
	  # ch;eck if changinf to > makes any difference
	  # return (np.where(np.dot(coefficients, x) - intercept) > 0 , 1,-1)
	  if (np.dot(coefficients, x) - intercept >= 0):
	    return 1
	  else:
	    return -1

	# Make the prediction using the function 'predictionLogisticRegresison'
	yLabelAndXFeaturesPrediction = testingData.map(lambda x: (x[0],x[1],predictionSVM(x[1]),np.dot(coefficients,x[1])))

	# Function to calculate True Positives
	def calculateTruePositives(x):
	  if (x[0] == 1 and x[2] == 1): # the article was Australian court case (x[0]) and the prediction was also Australian court case x[2]
	    return 1
	  else:
	    return 0

	# Function to calculate False Positives
	def calculateFalsePositives(x):
	  if (x[0] == -1 and x[2] == 1): # the article was not Australian court case (x[0]) but the prediction was Australian court case x[2]
	    return 1
	  else:
	    return 0

	# Function to calculate False Negatives
	def calculateFalseNegatives(x):
	  if (x[0] == 1 and x[2] == -1): # the article was Australian court case (x[0]) but the prediction was not Australian court case x[2]
	    return 1
	  else:
	    return 0

	# Function to calculate True Negatives
	def calculateTrueNegatives(x):
	  if (x[0] == -1 and x[2] == -1): # the article was not Australian court case (x[0]) and the prediction was not Australian court case x[2]
	    return 1
	  else:
	    return 0

	# Out of total positive labels predicted, how many correctly classified as positive, that is PPV
	def precision(x):
	  # Number of true positives/ (Number of true positives + Number of false positives) 
	  # return truePositive/(truePositive + falsePositive)
	  return x[1][0]/(float(x[1][0] + x[1][1]))

	# Out of actual positive labels, how many correctly classified as positive, that is, TPR
	def recall(x):
	  # Number of true positives/ (Number of true positives + Number of false Negatives) 
	  # return truePositive/(truePositive + falseNegative)
	  return x[1][0]/(float(x[1][0] +  x[1][2]))
	  
	  
	# Calculate 'True Positives', 'False Positives' and 'False Negatives'
	calcTP_FP_FN = yLabelAndXFeaturesPrediction.map(lambda x: (1, np.array([calculateTruePositives(x), calculateFalsePositives(x), calculateFalseNegatives(x),calculateTrueNegatives(x)]))).reduceByKey(np.add)

	print('')
	print ('#'*20)
	print('Number of True Positives:', calcTP_FP_FN.collect()[0][1][0])
	print('Number of False Positives:', calcTP_FP_FN.collect()[0][1][1])
	print('Number of False Negatives:', calcTP_FP_FN.collect()[0][1][2])
	print('Number of True Negatives:', calcTP_FP_FN.collect()[0][1][3])
	print('')

	# Calculate F1 score
	calculateF1score = calcTP_FP_FN.map(lambda x: (precision(x), recall(x))).map(lambda x: 2*x[0]*x[1] / (x[0] + x[1])).collect()[0]
	print('F1 score for classifier =',round(calculateF1score*100,2),'%')
	print('')

	# Calculating the testing time (making predictions using the model)
	testingCompletionTime = datetime.now()
	testingTime = testingCompletionTime - TestingDatasetetPreprocessingCompletionTime

	
	# calculate the total end to end time for program
	totalTimeSVM = timeToReadTrainDataAndPreProcess + timeToReadTestDataAndPreProcess + trainingTime + testingTime

	print("Time taken to read Testing and Training Data and preprocess them (h:mm:ss):", timeToReadTrainDataAndPreProcess + timeToReadTestDataAndPreProcess)
	print("Time taken to train using the Support Vector Machine (h:mm:ss):", trainingTime)
	print("Time taken to test using the Support Vector Machine (h:mm:ss):", testingTime)
	print("Total Time taken by Support Vector Machine (h:mm:ss):", totalTimeSVM)
	print ('#'*20)

	# List to store the results of task 2
	ansForTask2 = []

	ansForTask2.append(('F1 score for classifier =',round(calculateF1score*100,2),'%'))

	ansForTask2.append('')
	ansForTask2.append(("Time taken to read Testing and Training Data and preprocess them (days:seconds:microsecond):", timeToReadTrainDataAndPreProcess + timeToReadTestDataAndPreProcess))
	ansForTask2.append(("Time taken to train using the Support Vector Machine (days:seconds:microsecond):", trainingTime))
	ansForTask2.append(("Time taken to test using the Support Vector Machine (days:seconds:microsecond):", testingTime))
	ansForTask2.append(("Total Time taken by Support Vector Machine (days:seconds:microsecond):", totalTimeSVM))

	ansForTask2.append('')
	ansForTask2.append(('Number of True Positives', calcTP_FP_FN.collect()[0][1][0]))
	ansForTask2.append(('Number of False Positives', calcTP_FP_FN.collect()[0][1][1]))
	ansForTask2.append(('Number of False Negatives', calcTP_FP_FN.collect()[0][1][2]))
	ansForTask2.append(('Number of True Negatives', calcTP_FP_FN.collect()[0][1][3]))

	# Save the results of task1 in a text file
	sc.parallelize(ansForTask2).coalesce(1, shuffle = False).saveAsTextFile(sys.argv[3]) 


	sc.stop

