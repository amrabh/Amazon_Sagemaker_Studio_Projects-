#!/usr/bin/env python
# coding: utf-8

# # PROJECT #1: Predict the employee salary based on the number of years of experience

# - The objective of this case study is to predict the employee salary based on the number of years of experience. 
# - In simple linear regression, we predict the value of one variable Y based on another variable X.
# - X is called the independent variable and Y is called the dependant variable.
# - Why simple? Because it examines relationship between two variables only.
# - Why linear? when the independent variable increases (or decreases), the dependent variable increases (or decreases) in a linear fashion.
# 

# # TASK #2: IMPORT LIBRARIES AND DATASETS

# In[2]:


#install seaborn library
#!pip install seaborn
#!pip install tensorflow
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# read the csv file 
salary_df = pd.read_csv('salary.csv')


# In[4]:


salary_df


# MINI CHALLENGE
#  - Use head and tail methods to print the first and last 7 rows of the dataframe
#  - Try to find the maximum salary value in the dataframe 

# In[7]:


# get the first 7 rows of the dataframe
salary_df.head(7)


# In[8]:


# get the last 7 rows of the dataframe
salary_df.tail(7)


# In[10]:


# find the maximum salary value in the dataframe
max(salary_df['Salary'])


# # TASK #3: PERFORM EXPLORATORY DATA ANALYSIS AND VISUALIZATION

# In[13]:


# check if there are any Null values
sns.heatmap(salary_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[14]:


# Check the dataframe info

salary_df.info()


# In[15]:


# Statistical summary of the dataframe

salary_df.describe()


# MINI CHALLENGE
#  - What are the number of years of experience corresponding to employees with minimum and maximim salaries?

# In[23]:


salary_df[salary_df['Salary']==max(salary_df['Salary'])]


# In[24]:


salary_df[salary_df['Salary']==min(salary_df['Salary'])]


# In[25]:


salary_df.hist(bins = 30, figsize = (20,10), color = 'r')


# In[26]:


# plot pairplot

sns.pairplot(salary_df)


# In[27]:


corr_matrix = salary_df.corr()
sns.heatmap(corr_matrix, annot = True)
plt.show()


# MINI CHALLENGE
# - Use regplot in Seaborn to obtain a straight line fit between "salary" and "years of experience"

# In[28]:


sns.regplot(salary_df['YearsExperience'],salary_df['Salary'])


# # TASK #4: CREATE TRAINING AND TESTING DATASET

# In[29]:


X = salary_df[['YearsExperience']]
y = salary_df[['Salary']]


# In[30]:


X


# In[31]:


y


# In[32]:


X.shape


# In[33]:


y.shape


# In[34]:


X = np.array(X).astype('float32')
y = np.array(y).astype('float32')


# In[35]:


# Only take the numerical variables and scale them
X 


# In[36]:


# split the data into test and train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# MINI CHALLENGE
#  - Try splitting the data into 75% for training and the rest for testing
#  - Verify that the split was successful by obtaining the shape of both X_train and X_test
#  - Did you notice any change in the order of the data? why?

# In[38]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.3)


# In[40]:


print(X_train1.shape)
print(X_test1.shape)
print(y_train1.shape)
print(y_test1.shape)


# # TASK #5: TRAIN A LINEAR REGRESSION MODEL IN SK-LEARN (NOTE THAT SAGEMAKER BUILT-IN ALGORITHMS ARE NOT USED HERE)

# In[41]:


# using linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

regresssion_model_sklearn = LinearRegression(fit_intercept = True)
regresssion_model_sklearn.fit(X_train, y_train)


# In[42]:


regresssion_model_sklearn_accuracy = regresssion_model_sklearn.score(X_test, y_test)
regresssion_model_sklearn_accuracy


# In[43]:


print('Linear Model Coefficient (m): ', regresssion_model_sklearn.coef_)
print('Linear Model Coefficient (b): ', regresssion_model_sklearn.intercept_)


# MINI CHALLENGE
# - Retrain the model while setting the fit_intercept = False, what do you notice?

# In[44]:


# using linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

regresssion_model_sklearn = LinearRegression(fit_intercept = True)
regresssion_model_sklearn.fit(X_train1, y_train1)


# In[45]:


regresssion_model_sklearn_accuracy = regresssion_model_sklearn.score(X_test1, y_test1)
regresssion_model_sklearn_accuracy


# In[46]:


print('Linear Model Coefficient (m): ', regresssion_model_sklearn.coef_)
print('Linear Model Coefficient (b): ', regresssion_model_sklearn.intercept_)


# In[51]:


# using linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

regresssion_model_sklearn = LinearRegression(fit_intercept = False)
regresssion_model_sklearn.fit(X_train, y_train)


# In[52]:


egresssion_model_sklearn_accuracy = regresssion_model_sklearn.score(X_test, y_test)
regresssion_model_sklearn_accuracy


# In[53]:


print('Linear Model Coefficient (m): ', regresssion_model_sklearn.coef_)
print('Linear Model Coefficient (b): ', regresssion_model_sklearn.intercept_)


# # TASK #6: EVALUATE TRAINED MODEL PERFORMANCE (NOTE THAT SAGEMAKER BUILT-IN ALGORITHMS ARE NOT USED HERE)

# In[21]:


y_predict = regresssion_model_sklearn.predict(X_test)


# In[22]:


y_predict


# In[23]:


plt.scatter(X_train, y_train, color = 'gray')
plt.plot(X_train, regresssion_model_sklearn.predict(X_train), color = 'red')
plt.ylabel('Salary')
plt.xlabel('Number of Years of Experience')
plt.title('Salary vs. Years of Experience')


# MINI CHALLENGE
#  - Use the trained model, obtain the salary corresponding to eployees who have years of experience = 5

# # TASK #7: TRAIN A LINEAR LEARNER MODEL USING SAGEMAKER

# In[54]:


# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
# Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2

import sagemaker
import boto3
from sagemaker import Session

# Let's create a Sagemaker session
sagemaker_session = sagemaker.Session()
bucket = Session().default_bucket()
# Let's define the S3 bucket and prefix that we want to use in this session
# bucket = 'sagemaker-practica' # bucket named 'sagemaker-practical' was created beforehand
prefix = 'linear_learner' # prefix is the subfolder within the bucket.

# Let's get the execution role for the notebook instance. 
# This is the IAM role that you created when you created your notebook instance. You pass the role to the training job.
# Note that AWS Identity and Access Management (IAM) role that Amazon SageMaker can assume to perform tasks on your behalf (for example, reading training results, called model artifacts, from the S3 bucket and writing training results to Amazon S3). 
role = sagemaker.get_execution_role()
print(role)


# In[55]:


X_train.shape


# In[56]:


y_train = y_train[:,0]


# In[57]:


y_train.shape


# In[58]:


import io # The io module allows for dealing with various types of I/O (text I/O, binary I/O and raw I/O). 
import numpy as np
import sagemaker.amazon.common as smac # sagemaker common libary

# Code below converts the data in numpy array format to RecordIO format
# This is the format required by Sagemaker Linear Learner 

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
smac.write_numpy_to_dense_tensor(buf, X_train, y_train)
buf.seek(0) 
# When you write to in-memory byte arrays, it increments 1 every time you write to it
# Let's reset that back to zero 


# In[59]:


import os

# Code to upload RecordIO data to S3
 
# Key refers to the name of the file    
key = 'linear-train-data'

# The following code uploads the data in record-io format to S3 bucket to be accessed later for training
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

# Let's print out the training data location in s3
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[60]:


X_test.shape


# In[61]:


y_test.shape


# In[62]:


# Make sure that the target label is a vector
y_test = y_test[:,0]


# In[63]:


# Code to upload RecordIO data to S3

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
smac.write_numpy_to_dense_tensor(buf, X_test, y_test)
buf.seek(0) 
# When you write to in-memory byte arrays, it increments 1 every time you write to it
# Let's reset that back to zero 


# In[64]:


# Key refers to the name of the file    
key = 'linear-test-data'

# The following code uploads the data in record-io format to S3 bucket to be accessed later for training
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'test', key)).upload_fileobj(buf)

# Let's print out the testing data location in s3
s3_test_data = 's3://{}/{}/test/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_test_data))


# In[65]:


# create an output placeholder in S3 bucket to store the linear learner output

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))


# In[66]:


# This code is used to get the training container of sagemaker built-in algorithms
# all we have to do is to specify the name of the algorithm, that we want to use

# Let's obtain a reference to the linearLearner container image
# Note that all regression models are named estimators
# You don't have to specify (hardcode) the region, get_image_uri will get the current region name using boto3.Session

from sagemaker.amazon.amazon_estimator import get_image_uri

container = get_image_uri(boto3.Session().region_name, 'linear-learner')


# In[67]:


# We have pass in the container, the type of instance that we would like to use for training 
# output path and sagemaker session into the Estimator. 
# We can also specify how many instances we would like to use for training
# sagemaker_session = sagemaker.Session()

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count = 1, 
                                       train_instance_type = 'ml.c4.xlarge',
                                       output_path = output_location,
                                       sagemaker_session = sagemaker_session)


# We can tune parameters like the number of features that we are passing in, type of predictor like 'regressor' or 'classifier', mini batch size, epochs
# Train 32 different versions of the model and will get the best out of them (built-in parameters optimization!)

linear.set_hyperparameters(feature_dim = 1,
                           predictor_type = 'regressor',
                           mini_batch_size = 5,
                           epochs = 5,
                           num_models = 32,
                           loss = 'absolute_loss')

# Now we are ready to pass in the training data from S3 to train the linear learner model

linear.fit({'train': s3_train_data})

# Let's see the progress using cloudwatch logs


# # TASK #8: DEPLOY AND TEST THE TRAINED LINEAR LEARNER MODEL 

# In[68]:


# Deploying the model to perform inference 

linear_regressor = linear.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m4.xlarge')


# In[69]:


from sagemaker.predictor import csv_serializer, json_deserializer

# Content type overrides the data that will be passed to the deployed model, since the deployed model expects data in text/csv format.

# Serializer accepts a single argument, the input data, and returns a sequence of bytes in the specified content type

# Deserializer accepts two arguments, the result data and the response content type, and return a sequence of bytes in the specified content type.

# Reference: https://sagemaker.readthedocs.io/en/stable/predictors.html

# linear_regressor.content_type = 'text/csv'
linear_regressor.serializer = csv_serializer
linear_regressor.deserializer = json_deserializer


# In[70]:


# making prediction on the test data

result = linear_regressor.predict(X_test)


# In[71]:


result # results are in Json format


# In[72]:


# Since the result is in json format, we access the scores by iterating through the scores in the predictions

predictions = np.array([r['score'] for r in result['predictions']])


# In[73]:


predictions


# In[74]:


predictions.shape


# In[75]:


# VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'gray')
plt.plot(X_test, predictions, color = 'red')
plt.xlabel('Years of Experience (Testing Dataset)')
plt.ylabel('salary')
plt.title('Salary vs. Years of Experience')


# In[76]:


# Delete the end-point

linear_regressor.delete_endpoint()


# # EXCELLENT JOB! NOW YOU'RE FAMILIAR WITH SAGEMAKER LINEAR LEARNER, YOU SHOULD BE PROUD OF YOUR NEWLY ACQUIRED SKILLS
