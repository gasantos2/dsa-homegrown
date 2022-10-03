# Databricks notebook source
# MAGIC %md
# MAGIC ## Week7 LinearRegression
# MAGIC 
# MAGIC In week 7, we've covered:
# MAGIC * Basic machine learning concepts and workflow
# MAGIC * Linear regression
# MAGIC   
# MAGIC   
# MAGIC In this notebook,  we will work on the Boston housing dataset and build a linear regression model to predict value of houses. 
# MAGIC 
# MAGIC The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/). The Boston housing data was collected in 1978 and each of the 506 entries represents aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# MAGIC - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# MAGIC - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# MAGIC - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# MAGIC - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.  
# MAGIC   
# MAGIC   
# MAGIC A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.  
# MAGIC   
# MAGIC   
# MAGIC Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a **TODO** statement and "____"
# MAGIC .
# MAGIC 
# MAGIC >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# COMMAND ----------

# MAGIC %md
# MAGIC Upload **Week7_LinearRegression_Homework.ipynb** and **housing.csv** to Google Drive.

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC ## 1. Import Libraries
# MAGIC Import pandas, numpy, matplotlib and seaborn. Then set %matplotlib inline.

# COMMAND ----------

# TODO  
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 2. Load Data
# MAGIC Read in the housing.csv file as a DataFrame called `data`.

# COMMAND ----------

# Upload housing.csv to your google Drive then read the data using pandas.

data = pd.read_csv('/dbfs/FileStore/shared_uploads/gasantos@redventures.com/housing.csv')

# Success
print("Boston housing dataset has {} data points with {} columns.".format(*data.shape))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 3. Explore Data

# COMMAND ----------

# MAGIC %md
# MAGIC Familiarizing ourself with the data through an explorative process is a fundamental practice to help us better understand and justify our results.

# COMMAND ----------

# TODO: Get the first n rows of data
data.head()

# COMMAND ----------

# TODO: Get a summary of the dataframe
data.info()

# COMMAND ----------

# TODO: Calculate descriptive statistics 
data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**.  
# MAGIC   
# MAGIC 
# MAGIC The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point.   
# MAGIC - `'RM'` is the average number of rooms among homes in the neighborhood.
# MAGIC - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# MAGIC - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.   
# MAGIC   
# MAGIC The **target variable**, `'MEDV'`, will be the variable we seek to predict.  

# COMMAND ----------

# TODO: Declare `features` and `target` variables
features = data[['RM','LSTAT','PTRATIO']]
target = data[['MEDV']]

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate Statistics
# MAGIC - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`.
# MAGIC - Store each calculation in their respective variable.
# MAGIC - Use `numpy` to perform the necessary calcualtions.

# COMMAND ----------

# TODO: Minimum price of the data
minimum_price = np.min(target)

# TODO: Maximum price of the data
maximum_price = np.max(target)

# TODO: Mean price of the data
mean_price = np.mean(target)

# TODO: Median price of the data
median_price = np.median(target)

# TODO: Standard deviation of prices of the data
std_price = np.std(target)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

# COMMAND ----------

# MAGIC %md
# MAGIC Observe features
# MAGIC 
# MAGIC For each of the three features **RM**, **LSTAT**, **PTRATIO**:
# MAGIC   * Do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? 
# MAGIC   * Show the data using appropriate plot and justify your answer for each.

# COMMAND ----------

features

# COMMAND ----------

# TODO  
for feature in features:  
  plt.scatter(features[f'{feature}'],target['MEDV'], label= f'{feature} x MEDV')
  plt.legend(loc='upper right')
  plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Answer**:    Seems like:
# MAGIC - PT RATIO does not have huge impact in the MEDV,
# MAGIC - LSTAT metric decreases with the increase of MEDV. 
# MAGIC - RM increases with the incresa of MEDV
# MAGIC 
# MAGIC We shoud investigate LSTAT and RM

# COMMAND ----------

# MAGIC %md
# MAGIC ----
# MAGIC 
# MAGIC ## 4. Develop Model

# COMMAND ----------

# MAGIC %md
# MAGIC Split data to get training and testing Data.

# COMMAND ----------

data_scaled = standard(data)
plt.scatter(data_scaled['RM'],data_scaled['MEDV'], label= 'RM x MEDV')
plt.scatter(data_scaled['LSTAT'],data_scaled['MEDV'], label= 'LSTAT x MEDV')
#plt.scatter(data_scaled['PTRATIO'],data_scaled['MEDV'], label= 'PTRATIO x MEDV')

# COMMAND ----------

# TODO

# importing necessaries libs
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep

# Creating X and y
X, y = features[['RM', 'LSTAT', 'PTRATIO']], target

# Split train and test (test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = prep.StandardScaler().fit(X_train)

X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Why should we perform feature scailing?

# COMMAND ----------

# MAGIC %md
# MAGIC **Answer**:    It is important to adjust the features to a common scale. 

# COMMAND ----------

# MAGIC %md
# MAGIC Create linear regression object

# COMMAND ----------

from sklearn.linear_model import LinearRegression 

# TODO 
#Create and Train the model
model = LinearRegression()

# COMMAND ----------

# MAGIC %md
# MAGIC Train/fit **lm** on the training data.

# COMMAND ----------

# TODO  
#Generate prediction
model.fit(X_train,y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Display a dataframe with the features and their corresponding coefficients in descending order. Think about how would you interprete the results here.

# COMMAND ----------

# TODO
df = pd.DataFrame(model.coef_,columns=['RM', 'LSTAT', 'PTRATIO']).transpose()

df

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Make Prediction
# MAGIC 
# MAGIC Now we have a trained model, let's make some predictions and evaluate the model's performance.

# COMMAND ----------

# MAGIC %md
# MAGIC Use **lm.predict( )** to predict house prices for **X_test** data set.

# COMMAND ----------

# TODO
print(model.score(X_test_s, y_test))


# COMMAND ----------

# MAGIC %md
# MAGIC Create a scatterplot of **y_test** versus the predicted values.

# COMMAND ----------

# TODO
predicted_values = model.predict(X_test_s)
plt.scatter(predicted_values,y_test)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Evaluate Model

# COMMAND ----------

# MAGIC %md
# MAGIC It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. 
# MAGIC 
# MAGIC Let's evaluate the model performance by calculating the Adjusted $R^2$.  

# COMMAND ----------

# MAGIC %md
# MAGIC What's the benefit of using Adjusted $R^2$ versus $R^2$?

# COMMAND ----------

# MAGIC %md
# MAGIC **Answer**:    The benefit using Ajusted R^2 vs R^2 is that the Adjusted takes into consideration number of parameters of the model, besides fitting the curve.

# COMMAND ----------

# MAGIC %md
# MAGIC Now calculate Adjusted $R^2$ for train set and test set. Hint: $R^2_{adj.}=1-(1-R^2)*\frac{n-1}{n-p-1}$

# COMMAND ----------

# TODO  
from sklearn.metrics import mean_squared_error as r

r_square = r(y_test, predicted_values)
print(r_square)

def adj_r_square(n, p, r_square):
    return 1 - (1 - r_square) * (n-1) / (n-p-1)


ad_r2 = adj_r_square(len(X_test), 3, r_square)
print(ad_r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## THE END, WELL DONE!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submission

# COMMAND ----------

# MAGIC %md
# MAGIC Download completed **Week7_LinearRegression_Homework.ipynb** from Google Colab and commit to your personal Github repo you shared with the faculty.
