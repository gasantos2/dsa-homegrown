# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Data Prep and Visualization in Python
# MAGIC 
# MAGIC In this project, we'll work through munging a data set and creating visualizations related to trends in the airline industry in the middle of the last century. You'll get started using [MatPlotLib](https://matplotlib.org/), a very powerful and popular plotting library in Python that is covered in this week's course materials.

# COMMAND ----------

# Install the pydataset package. This package gives us data sets to work with very easily
! pip install pydataset

# COMMAND ----------

# The convention for importing matplotlib with an alias is "plt". We'll also need pandas and numpy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Air Passengers Dataset
# MAGIC 
# MAGIC This dataset shows the number of passengers flying United States airlines by month from 1949-1960. Your job is to do various data munging operations on this dataset to clean it up and prepare it for several visualizations. You will then determine what code is needed to generate those visualizations.

# COMMAND ----------

from pydataset import data

passengers = data('AirPassengers')

# COMMAND ----------

# MAGIC %md
# MAGIC Ugh. When we examine the head of this datset, we can see that the years are in decimal form rather than month and year. We'll need to change that before we can do our analysis.
# MAGIC 
# MAGIC NOTE: The times are represented by twelfths. i.e. 1949.00000 = 149 0/12 (January). 1949.083333 = 1949 1/12 (February), and so on.

# COMMAND ----------

passengers.head(12)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The decimal years complicate the EDA work
# MAGIC 
# MAGIC We need to deal with this by making explicit month and year columns. It is common to have to reformat columns like this in a dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ## #1 Add a 'year' column to passengers that reflects the current year

# COMMAND ----------

from math import floor
from math import trunc

passengers['year'] = passengers['time'].apply(lambda x: floor(x))

# COMMAND ----------

# MAGIC %md
# MAGIC ## #2 Add a "month" column
# MAGIC 
# MAGIC Set this up in such a way that January is represented with a 1, February with a 2, etc.
# MAGIC 
# MAGIC *Hint: Create a column in `passengers` with a 2-digit decimal equivalent (after the dot). This column will repeat values every 12 rows. If we know what month each value in this column maps to, can we get our desired `month` column via some dataframe operation using this column?*

# COMMAND ----------

list_temp = []
for time in passengers['time']:
  list_temp.append(round(time - floor(time),2))
passengers['decimal_month'] = list_temp

# COMMAND ----------

## Creating a DataFreme auxilar to create month and month_name columns
df_aux = pd.DataFrame()
df_aux['month'] = list(range(1,13))
df_aux['month_name'] = ['January', 'Februray', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October','November', 'December']

mult = int(len(passengers)/len(df_aux))
df_aux = pd.concat([df_aux]*mult, ignore_index=True)#.reset_index()
df_aux.index = df_aux.index + 1

# COMMAND ----------

passengers = pd.concat([passengers,df_aux],axis=1)

# COMMAND ----------

passengers.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## #3 Generate the plot below of passengers vs. time using each monthly count

# COMMAND ----------

# MAGIC %md
# MAGIC <a href='https://drive.google.com/file/d/1PdaXbkCVzUXBnUP6c6cLP3nZ94ShSLg1/view?usp=embed_facebook&source=ctrlq.org'><img src='https://lh4.googleusercontent.com/7EHckqyjefS7rN8-gAtj2SgSyKfV3wlTnGKqCwzOf85F6NYlqYQbz7bDfWw=w2400' /></a>

# COMMAND ----------

# TODO
plt.plot(passengers['AirPassengers'])
plt.title("Number of Airline Passengers 1949-1960")
plt.ylabel("Hundreds of thousands")
plt.xlabel("Month")

# COMMAND ----------

# MAGIC %md
# MAGIC ## #4 Generate the plot below of passengers vs. time using an annual count

# COMMAND ----------

# MAGIC %md
# MAGIC <a href='https://drive.google.com/file/d/19WYHQR7sFgaeN5ZHlwx5x1-o-wxJ4weW/view?usp=sharing&amp;usp=embed_facebook&source=ctrlq.org'><img src='https://lh4.googleusercontent.com/2gbHNgm8UhbCEevaUBpMUSvVgk_6QuxMASqn9-wK1NdzrDXrcF-VIWK_o08=w2400' /></a>

# COMMAND ----------

# TODO
passengers_year = passengers.groupby('year').sum().reset_index()
plt.plot(passengers_year['AirPassengers'])
plt.title("Number of Airline Passengers 1949-1960")
plt.ylabel("Hundreds of thousands")
plt.xlabel("Year")

# COMMAND ----------

# MAGIC %md
# MAGIC ## #5 Generate the barplot below of passengers by year

# COMMAND ----------

# MAGIC %md
# MAGIC <a href='https://drive.google.com/file/d/1-4NF40zvVhwi6RWagJu98BaBuDNOXaEd/view?usp=sharing&amp;usp=embed_facebook&source=ctrlq.org'><img src='https://lh6.googleusercontent.com/IQRk35KApDIxYtHGH3WoczLnCvHCRdMNlHw64rgLWPYUostOoAn2hxp8lZA=w2400' /></a>

# COMMAND ----------

# TODO
plt.bar(passengers_year['year'] , passengers_year['AirPassengers'])
plt.title("Number of Airline Passengers 1949-1960")
plt.ylabel("Hundreds of thousands")
plt.xlabel("Year")

# COMMAND ----------

# MAGIC %md
# MAGIC ## #6 Generate the histogram below of monthly passengers
# MAGIC 
# MAGIC **Additional requirements:**
# MAGIC 
# MAGIC * Only include 1955 and beyond
# MAGIC * Use a binwidth of 50, a min of 200, and a max of 700
# MAGIC * Set the yticks to start at 0, end at 25 by interval of 5

# COMMAND ----------

# MAGIC %md
# MAGIC <a href='https://drive.google.com/file/d/1mEtvUbnh2LcDDc73LNr_qX984HzgyhiQ/view?usp=sharing&amp;usp=embed_facebook&source=ctrlq.org'><img src='https://lh6.googleusercontent.com/7I2FzRPSQPyoalFcwH3vTDeB9Gf80OUlaZOs1x9oRRYyQLlHXPU9H-NhSVQ=w2400' /></a>

# COMMAND ----------

# TODO
passengers_1955_beyond = passengers[passengers['year']>1955]

bins = np.arange(200, 700, 50)
yticks = np.arange(0, 25, 5) 
plt.hist(passengers_1955_beyond['AirPassengers'], bins = bins)
plt.yticks(yticks)
plt.title("Distribution of Monthly Airline Passengers 1955-1960")
plt.ylabel("Count")
plt.xlabel("Hundreds of thousands")

# COMMAND ----------

# MAGIC %md
# MAGIC ## #7 Generate the histogram below of monthly passengers
# MAGIC 
# MAGIC **Additional requirements:**
# MAGIC 
# MAGIC * Generate two groups to compare. Group 1 should be the years 1949-1950. Group 2 should be the years 1959-60.
# MAGIC * Binwidth of 50 from 100 to 700
# MAGIC * yticks from 0 to 24, spaced by 2
# MAGIC * Be sure to include a legend

# COMMAND ----------

# MAGIC %md
# MAGIC <a href='https://drive.google.com/file/d/1gqJbBVOPIurYikUIDpXoAF3gZx2p8lUA/view?usp=sharing&amp;usp=embed_facebook&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/Ok91nFY8Srjn1FpVwOil9ycH9y6isZejTqi7hifqaEA5E3tWpkwldWVLo3U=w2400' /></a>

# COMMAND ----------

# TODO
group1 = passengers[(passengers['year'] == 1949) | (passengers['year'] == 1950)]
group2 = passengers[(passengers['year'] == 1959) | (passengers['year'] == 1960)]

bins = np.arange(100, 700, 50)
yticks = np.arange(0, 24, 2) 

plt.hist(group1['AirPassengers'], bins, label='1949-50')
plt.hist(group2['AirPassengers'], bins, label='1959-60')
plt.legend(loc='upper right')
plt.yticks(yticks)
plt.title("Air passenger distributions, beginning and end of decade")
plt.ylabel("Count")
plt.xlabel("Hundreds of thousands")

# COMMAND ----------

# MAGIC %md
# MAGIC ## #8 Generate the time plot below
# MAGIC 
# MAGIC **Additional requirements:**
# MAGIC 
# MAGIC * Compare 1950, 1955, and 1960 by month

# COMMAND ----------

# MAGIC %md
# MAGIC <a href='https://drive.google.com/file/d/11nVH5EiYxxtJ48isS9VLtwLIjn0hALXV/view?usp=sharing&amp;usp=embed_facebook&source=ctrlq.org'><img src='https://lh3.googleusercontent.com/SKfWqBE324A__VS8V-TBqMQXHWE9OUjVoJyeyJME8uJzyfWS73aaCms7A3c=w2400' /></a>

# COMMAND ----------

# TODO
passengers_1950 = passengers[passengers['year'] == 1950] 
passengers_1955 = passengers[passengers['year'] == 1955]
passengers_1960 = passengers[passengers['year'] == 1960]

# COMMAND ----------

plt.plot(passengers_1950['month'],passengers_1950['AirPassengers'])
plt.plot(passengers_1955['month'],passengers_1955['AirPassengers'])
plt.plot(passengers_1960['month'],passengers_1960['AirPassengers'])
plt.legend(loc='upper right')
plt.title("Air passengers by month: beginning, mid, end of decade")
plt.ylabel("Hundreds of thousands")
plt.xlabel("Month")

# COMMAND ----------

# MAGIC %md
# MAGIC ## #9  Understand your data and tell a story
# MAGIC 
# MAGIC * Which of these plots would you create first to explore your data before building a model or performing an analysis? Why?
# MAGIC * If you could only use one of these plots to tell a story about air travel trends mid-centry, which would you use and why? What are some insights you could share?

# COMMAND ----------

# MAGIC %md
# MAGIC If I could only use one of these plots, probably I would use the #8 time plot. 
# MAGIC 
# MAGIC Some insights:
# MAGIC 
# MAGIC 1 . We can see that over the years, the number of Air passagenrs have incresed.
# MAGIC 
# MAGIC 2 . We can see kind of a trend to have a peak in the middle of the year

# COMMAND ----------

# MAGIC %md
# MAGIC # Part 2: Data Scaling and Normalization
# MAGIC Above shows some fundamental transformations that are useful techniques while working with data. There are other transformation techniques that are crucial to some data science algorithms. We will go over data scaling and normalization. This [Kaggle Tutorial](https://www.kaggle.com/code/alexisbcook/scaling-and-normalization/tutorial) is a good reference that discusses these methods at a high level. 
# MAGIC 
# MAGIC To recap, **data scaling** is a way to limit numerical values within a specified range. This is a necessary pre-preprocessing step to some algorithms, expecially neural networks. 
# MAGIC 
# MAGIC **Data normalization** is a way to change the distribution of numerical values. From this [Medium article](https://medium.com/analytics-vidhya/normal-distribution-and-machine-learning-ec9d3ca05070), "models like LDA, Gaussian Naive Bayes, Logistic Regression, Linear Regression, etc., are explicitly calculated from the assumption that the distribution is a bivariate or multivariate normal."

# COMMAND ----------

# Package imports

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

# COMMAND ----------

# Load dataset

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
dataset = read_csv(url, header=None)

# COMMAND ----------

# Take a look at the data 

dataset.head()

# COMMAND ----------

# Summarize the shape of the dataset (rows, columns)

dataset.shape

# COMMAND ----------

# Summarize each variable
# We can see that for each column, the min max is not consistent 

dataset.describe()

# COMMAND ----------

# Histograms of the columns

fig = plt.figure(figsize = (15,20))
ax = fig.gca()
dataset.hist(ax = ax)
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare dataset for Scaling
# MAGIC We will change the scale of features to be from zero to one. To do this we will use **min max scaling**, with its formulation found [here](https://en.wikipedia.org/wiki/Feature_scaling#:~:text=Also%20known%20as%20min%2Dmax,the%20nature%20of%20the%20data.). If you are interested in doing this yourself without a package, please do so. That will be provided in the solutions as well. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### #10 Remove the target feature so we only have the numerical columns. 
# MAGIC Make sure it is a pandas data frame and is named X

# COMMAND ----------

# TO DO
X = dataset.drop(columns=[60])

# COMMAND ----------

from sklearn.preprocessing import MinMaxScaler

# Use min max scalar 
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)

# Transform the test test
X_scaled = scaler.transform(X)

# COMMAND ----------

# The scaled data is returned as a nump array, change to pandas df

X_scaled = pd.DataFrame(X_scaled) 
X_scaled

# COMMAND ----------

# Lets take a look at the distribution of each column, it hasn't changed! 
# The only thing that has changed is the range of values, theyre all between zero and one

fig = plt.figure(figsize = (15,20))
ax = fig.gca()
X_scaled.hist(ax = ax)
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using data scaling to normalize our data set.
# MAGIC We will be using the box-cox method of normalizing our data. A reference to how this is derived can be found [here](https://www.statisticshowto.com/box-cox-transformation/), however, the details are NOT necessary to know. 

# COMMAND ----------

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='box-cox')
data = pt.fit_transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## The above errored :(
# MAGIC   This is because our data has zeros in it! We can solve for this by using the min max scalar, except for having a range from zero and one, we can make the range what we decide to make the data is strictly positive!

# COMMAND ----------

# MAGIC %md
# MAGIC ### #11 Re run the data scaling so instead of the range being from zero to one, change it so that it is strictly positive. (Zero is not positive) 

# COMMAND ----------

# TODO

scaler = MinMaxScaler(feature_range=(1,2))
scaler.fit(X)
X_scaled = scaler.transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ### #12 Fit the new transformer to the updated scaled data set 

# COMMAND ----------

# TODO

pt = PowerTransformer(method='box-cox')
X_normalized = pt.fit_transform(X_scaled)

# COMMAND ----------

# MAGIC %md
# MAGIC ### #13 Plot a histogram for each column and compare the results from the originally loaded data set

# COMMAND ----------

# TODO
X_normalized = pd.DataFrame(X_normalized)

# COMMAND ----------

X_normalized.hist()
plt.show()
