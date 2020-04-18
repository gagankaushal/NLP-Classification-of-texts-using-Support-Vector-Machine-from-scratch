# Solutions to Assignment 5 (MET CS777)

Submitted by: Gagan Kaushal (gk@bu.edu)


# Answers

## TASK 1
('F1 score for Logistic Regression classifier (trained using mllib) =', 99.6, '%')  
  
('Time taken to read Testing and Training Data and preprocess them (days:seconds:microseconds):', datetime.timedelta(0, 230, 238702))    
('Time taken to train the Logistic Regression model (days:seconds:microseconds):', datetime.timedelta(0, 5167, 335214))    
('Time taken to test the Logistic Regression model (days:seconds:microseconds):', datetime.timedelta(0, 175, 346919))    
('Total Time taken by Logistic Regression (trained using mllib) (days:seconds:microseconds):', datetime.timedelta(0, 5572, 920835))    
  
('Number of True Positives', 374)  
('Number of False Positives', 0)  
('Number of False Negatives', 3)  
('Number of True Negatives', 18347)  
  
The above total time taken by Logistic Regression corresponds to 1.5 hours  

# TASK 2
('F1 score for classifier =', 99.87, '%')  
  
('Time taken to read Testing and Training Data and preprocess them (days:seconds:microsecond):', datetime.timedelta(0, 1292, 854187))  
('Time taken to train using the Support Vector Machine (days:seconds:microsecond):', datetime.timedelta(0, 5148, 112811))  
('Time taken to test using the Support Vector Machine (days:seconds:microsecond):', datetime.timedelta(0, 158, 967617))  
('Total Time taken by Support Vector Machine (days:seconds:microsecond):', datetime.timedelta(0, 6599, 934615))  

('Number of True Positives', 377)  
('Number of False Positives', 1)  
('Number of False Negatives', 0)  
('Number of True Negatives', 18346)  

The above total time taken by SVM corresponds to 1.8 hours  

# TASK 3
('##########', ' LOGISTIC REGRESSION - RESULTS ', '##########')  
('F1 score for classifier =', 99.6, '%')  

('Time taken to read Testing and Training Data and preprocess them (trained using mllib)(days:seconds:microsecond:', datetime.timedelta(0, 2400, 746235))  
('Time taken to train the Logistic Regression model (trained using mllib)(days:seconds:microsecond:', datetime.timedelta(0, 1322, 621003))  
('Time taken to test the Logistic Regression model (trained using mllib)(days:seconds:microsecond:', datetime.timedelta(0, 156, 601166))  
('Total Time taken by Logistic Regression (trained using mllib)(days:seconds:microsecond:', datetime.timedelta(0, 3879, 968404))  

('Number of True Positives', 374)  
('Number of False Positives', 0)  
('Number of False Negatives', 3)  
('Number of True Negatives', 18347)  

('##########', ' SUPPORT VECTOR - RESULTS ', '##########')  
('F1 score for classifier =', 98.79, '%')  

('Time taken to read Testing and Training Data and preprocess them (days:seconds:microsecond):', datetime.timedelta(0, 2400, 746235))  
('Time taken to train the Support Vector Machine model(days:seconds:microsecond):', datetime.timedelta(0, 1415, 655597))  
('Time taken to test using the Support Vector Machine model (days:seconds:microsecond):', datetime.timedelta(0, 138, 874487))  
('Total Time taken by Support Vector Machine (days:seconds:microsecond):', datetime.timedelta(0, 3955, 276319))  

('Number of True Positives', 368)   
('Number of False Positives', 0)  
('Number of False Negatives', 9)  
('Number of True Negatives', 18347)  
  
The above total time taken by Logistic Regression corresponds to 1.1 hours  
The above total time taken by SVM corresponds to 1.2 hours  

# Applied one dimensionality selection technique - (Selection of features using specific probability distribution) 
## Description
Used 'np.random.choice()' function to select the 10k features out of 20k features.
However, I specified the probability for each of 20k features so that more preference is given to select those features that have higher regression coefficents.   
These probabilities for 20k features were leveraged as per the below algorithm:

1. Read the regression coefficients from 'Assignment_4_Final_Output_Task2_part-00000' that were generated after training the Logistic Regression (from scratch) in Assignment 4 - task 2.
2. Normalise the 20000 regression coefficients to the range [0,1]. This step is done because the probability lies in the range of [0,1]
3. Input the above probabilities for 20000 features in the 'np.random.choice' function.

Using the above probabilities, the function gives more preference and selects those 10000 features out of the 20000 features that are most important in classifying Australian Court Cases from Wikipedia pages (since those features will have higher regression coefficients)

Note: Above mentioned 'Assignment_4_Final_Output_Task2_part-00000' for large dataset and 'part-00000' for small dataset containing Regression Coefficients for large and small dataset respectively are available in the 'docs' folder.

## Reason for selection
I selected this approach because of following reasons:
1. Less Complex: Wanted a simple algorithm to select features as using complicated approaches like PCA (Principal Component Analysis) can be very expensive on very large datasets.
2. Generates comparable results in less time: By specifying the probabilities, I had the flexibility to give more preference to the features that have higher regression coefficients (features that contribute more in the classification task). So, we get faster performance from 10k features as compared to 20k features with almost comparable F1 score. This is best of both worlds.

## Yes, my approach is applicable on very large dataset since the features are selected randomly and doesn't involve very complicated mathematics to select the features. However, we would have to train the model once on the dataset to find the optimal regression coefficients. But an alternate approach could be to specify the probability of '1' only for the top 5 words.

# How to run  
Run the task 1, task 2 and task 3 as per the below templates by submitting the tasks to spark-submit. 


## Task 1 - template
```python

spark-submit <task_name> <Training_dataset> <Testing_dataset> <output_folder_for_results>

```

Task 1 - Small Dataset
```python

spark-submit main_task1.py SmallTrainingData.txt SmallTrainingData.txt Output_task1

```
	
Task 1 - Large Dataset
```python
-- pyspark file location
s3://gagankaushal/Assignment_5/main_task1.py

-- arguments
s3://metcs777/TrainingData.txt
s3://metcs777/TestingData.txt
s3://gagankaushal/Assignment_5/Output_task1

```

## Task 2 - template
```python

spark-submit <task_name> <Training_dataset> <Testing_dataset> <output_folder_for_results>

```

Task 2 - Small Dataset
```python

spark-submit main_task2.py SmallTrainingData.txt SmallTrainingData.txt Output_task2

```
	
Task 2 - Large Dataset
```python
-- pyspark file location
gs://gagankaushal.com/Assignment_5/Final_/main_task2.py

-- arguments
gs://metcs777/TrainingData.txt
gs://metcs777/TestingData.txt
gs://gagankaushal.com/Assignment_5/Final_/Output_task2

```



## Task 3 - template
```python

spark-submit <task_name> <Training_dataset> <path of output file generated by task 2 (Assignment 4) containing 'regression coefficients'> <Testing_dataset> <output_folder_for_results>

```

Task 3 - Small Dataset

```python

spark-submit main_task3.py SmallTrainingData.txt docs/part-00000 SmallTrainingData.txt Output_task3

```
Task 3 - Large Dataset
```python
-- pyspark file location
s3://gagankaushal/Assignment_5/main_task3.py

-- arguments
s3://metcs777/TrainingData.txt
s3://gagankaushal/Assignment_5/Assignment_4_Final_Output_Task2_part-00000
s3://metcs777/TestingData.txt
s3://gagankaushal/Assignment_5/Output_task3

```
Note: Above mentioned 'Assignment_4_Final_Output_Task2_part-00000' for large dataset and 'part-00000' for small dataset is available in the 'docs' folder

These files are basically used for specifying the probability for selecting 10000 features out of the 20000 features as per the below algorithm:
1. Read the regression coefficients from 'Assignment_4_Final_Output_Task2_part-00000'' that were generated after training the Logistic Regression (from scratch)
2. Normalise the 20000 regression coefficients to the range [0,1]. This step is done because the probability lies in the range of [0,1]
3. Input the above probabilities for 20000 features in the 'np.random.choice' function.

Using the above probabilities, the function gives more preference and selects those 10000 features out of the 20000 features that are most important in classifying Australian Court Cases from Wikipedia pages (since those features will have higher regression coefficients)




