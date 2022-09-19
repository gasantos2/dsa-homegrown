# Databricks notebook source
# MAGIC %md
# MAGIC # Week3 Python Programming

# COMMAND ----------

# MAGIC %md
# MAGIC In week 3, we've covered:
# MAGIC * **DataFrames**

# COMMAND ----------

# MAGIC %md
# MAGIC The best way to consolidate the knowledge in your mind is by practicing.<br>Please complete the part marked with <span style="color:green">**# TODO**</span>.
# MAGIC 
# MAGIC [Google](www.google.com) and [Python DataFrame Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) are your good friends if you have any python questions.
# MAGIC 
# MAGIC Upload **Week3_PythonProgramming_Homework.ipynb** notebook to your Google Drive and open it with Google Colab.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and explore your data

# COMMAND ----------

# MAGIC %md
# MAGIC Import the necessary libraries.

# COMMAND ----------

# TODO
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC Import the data set by using `url` and assign it to a variable called `wine`. Hint: need to set the `sep=';'`
# MAGIC 
# MAGIC Check [Wine quality data description.](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

# COMMAND ----------

# TODO
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine = pd.read_csv(url, sep=';')

# COMMAND ----------

# MAGIC %md
# MAGIC Show the first 10 entries of `wine`.

# COMMAND ----------

# TODO
wine.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Show the last 10 entries of `wine`.

# COMMAND ----------

# TODO
wine.tail(10)

# COMMAND ----------

# MAGIC %md
# MAGIC What is the number of observations in the dataset?

# COMMAND ----------

# TODO
wine.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC What is the number of columns in the dataset?

# COMMAND ----------

# TODO
wine.shape[1]

# COMMAND ----------

# MAGIC %md
# MAGIC What is the data type of each column?

# COMMAND ----------

# TODO
wine.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Summarize statistics of all columns.

# COMMAND ----------

# TODO
wine.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Access the `quality` column.

# COMMAND ----------

# TODO
wine['quality']

# COMMAND ----------

# MAGIC %md
# MAGIC How many unique `quality` values are there?

# COMMAND ----------

# TODO
wine['quality'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC Which `quality` value has the most samples? Hint: `value_counts()`

# COMMAND ----------

# TODO
wine['quality'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Rename and filter your data

# COMMAND ----------

# MAGIC %md
# MAGIC Replace the space in the column name with `_` and show the first 10 entries. For example,  `fixed acidity` would become `fixed_acidity`.

# COMMAND ----------

# TODO
wine.columns = wine.columns.str.replace(' ','_')
wine.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Print a data frame with only two columns `residual_sugar` and `quality`.

# COMMAND ----------

# TODO
wine_residual_quality = wine[['residual_sugar','quality']]
wine_residual_quality

# COMMAND ----------

# MAGIC %md
# MAGIC Select only the observations with `residual_sugar` > 2 and `quality` < 8. 

# COMMAND ----------

# TODO
wine_filter = wine.loc[(wine['residual_sugar'] >2) & (wine['quality'] <8)]
wine_filter

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new column `residual_sugar_norm` (and display the updated dataframe) so that:
# MAGIC      
# MAGIC \begin{equation*}
# MAGIC residual\_sugar\_norm_i =  \frac{residual\_sugar_i - min(residual\_sugar)}{max(residual\_sugar) - min(residual\_sugar)}   \textrm{, where i = 0, 1, 2, ..., observation index}
# MAGIC \end{equation*}
# MAGIC 
# MAGIC (This technique is called "min-max scaling"; it scales the minimum to 0, the maximum to 1, and preserves all relative distances in between.)

# COMMAND ----------

# TODO
wine_filter['residual_sugar_norm'] = wine_filter['residual_sugar'] - min(wine_filter['residual_sugar']) / max(wine_filter['residual_sugar']) - min(wine_filter['residual_sugar'])
wine_filter

# COMMAND ----------

# MAGIC %md
# MAGIC Drop the new column `residual_sugar_norm` inplace.

# COMMAND ----------

# TODO
wine_filter.drop(columns=['residual_sugar_norm'], inplace=True)
wine_filter.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new column `quality_level` so that the top half quality wines have value as `'high'` and the bottom half quality wine have value as `'low'`.

# COMMAND ----------

# TODO
wine_filter['quality_level'] = ['high' if x > wine_filter['quality'].median() else 'low' for x in  wine_filter['quality']]
wine_filter.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Outlier Detection

# COMMAND ----------

# MAGIC %md
# MAGIC Access the `sulphates` column

# COMMAND ----------

# TODO
column_sulphates = wine['sulphates']
column_sulphates.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Look at the histogram of `sulphates`. Can you see any potential outliers?

# COMMAND ----------

from matplotlib import pyplot
%matplotlib inline
pyplot.hist(column_sulphates)
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO**
# MAGIC 
# MAGIC Number above 1.75 seems to be the outlier

# COMMAND ----------

# MAGIC %md
# MAGIC What is the mean of `sulphates`? The standard deviation?

# COMMAND ----------

# TODO
print(column_sulphates.mean())
print(column_sulphates.std())

# COMMAND ----------

# MAGIC %md
# MAGIC One method of identifying outliers (for normally distributed data) is to find points outside three standard deviations from the mean. Using this definition, what are the upper and lower bounds of "non-outlier" values of `sulphates`?

# COMMAND ----------

# TODO
upper_limit = column_sulphates.mean() + 3*column_sulphates.std()
lower_limit = column_sulphates.mean() - 3*column_sulphates.std()

print(f'Points outliers are outside the range {lower_limit} and {upper_limit}')

# COMMAND ----------

# MAGIC %md
# MAGIC How many points would be considered outliers?

# COMMAND ----------

# TODO
len(column_sulphates.loc[(column_sulphates<lower_limit) | (column_sulphates>upper_limit)])

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new dataframe from wine called removed_outliers that removes the rows containing the `sulphates` outliers you just found.

# COMMAND ----------

# TODO
removed_outliers = wine[(wine['sulphates'] < upper_limit) & (wine['sulphates'] > lower_limit)]
removed_outliers.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Another way to spot outliers is with the dots on a boxplot. Does this method seem to agree with the previous method?

# COMMAND ----------

pyplot.boxplot(sulphates)
pyplot.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Why do you think we might we want to be careful before removing these outliers?

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO**
# MAGIC 
# MAGIC If sulphates isn't normally distributed, we could flag too many outliers

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. GroupBy

# COMMAND ----------

# MAGIC %md
# MAGIC Which `quality_level` are with higher `fixed_acidity` on average? Hint: use `groupby`

# COMMAND ----------

# TODO
wine['quality_level'] = ['high' if x > wine['quality'].median() else 'low' for x in  wine['quality']]

wine.groupby('quality_level').mean()['fixed_acidity']


# COMMAND ----------

# MAGIC %md
# MAGIC For each of the numerical features, plot a histogram of the feature values under different categories of `quality_level`. Can you use this to tell which feature is more likely to indicate the `quality_level` of wine?
# MAGIC 
# MAGIC <i>Hint: Consider using seaborn's </i>`histplot()`<i>, with </i>`quality_level` <i>as the "hue" argument.</i>

# COMMAND ----------

# TODO
import seaborn as sns

def check_quality(column):    
    sns.histplot(data=wine, x=column, hue='quality_level')
    pyplot.title(column + ' vs quality_level')
    pyplot.xlabel(column)
    pyplot.ylabel('count')
    pyplot.show()

for column in wine.drop(columns=['quality_level']).columns:
  check_quality(column)

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Handling missing data

# COMMAND ----------

# MAGIC %md
# MAGIC Create a dataframe with `raw_data` and assign it to a variable called `df`.

# COMMAND ----------

# TODO
import numpy as np

raw_data = {'first_name': ['Jason', np.nan, 'Tina', 'Jake', 'Amy', 'Ellen'], 
        'last_name': ['Miller', np.nan, 'Ali', 'Milner', 'Cooze', 'Smith'], 
        'age': [42, np.nan, 36, 24, 73, 55], 
        'sex': ['m', np.nan, 'f', 'm', 'f', 'f'], 
        'region': ['Northeast', np.nan, 'Midwest', 'South', 'Northeast', 'Midwest'],
        'preTestScore': [4, np.nan, np.nan, 2, 3, np.nan],
        'postTestScore': [25, np.nan, np.nan, 62, 70, 50]}

df = pd.DataFrame(raw_data)
df

# COMMAND ----------

# MAGIC %md
# MAGIC Drop observations with missing values and assign it to a variable called `df_clean`.

# COMMAND ----------

# TODO
df_clean = df.dropna()
df_clean

# COMMAND ----------

# MAGIC %md
# MAGIC For each column in `df`, show what percentage of its values are null. Try to avoid having repeated blocks of code for each column.
# MAGIC 
# MAGIC (Regarding the second line, the goal is to get you thinking like a Pandas programmer. In practice you may have hundreds of variables in your dataframe, and hard coded statements using each column name will not scale.)
# MAGIC 
# MAGIC <i>Hint: Using </i>`df.isna()`<i> might be helpful here. How can you get column-wise percentages from the output of this?</i>

# COMMAND ----------

# TODO
100*(df.isna().sum(axis=0) / len(df))

# COMMAND ----------

# MAGIC %md
# MAGIC Use your result from above to drop columns from `df` with 40% or more of its values null, and store the result in a dataframe called `temp`.

# COMMAND ----------

# TODO
filter_column = 100*(df.isna().sum(axis=0) / len(df)) < 40
temp = df.loc[:,filter_column]
temp.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Fill all missing values in `df` with zeros.

# COMMAND ----------

# TODO
df_fill = df.fillna(0)
df_fill

# COMMAND ----------

# MAGIC %md
# MAGIC Use `groupby` on `df` to find the average post test score by region. How did `groupby` handle the missing values? If you were trying to find the averages by region, would you choose this method over filling missing values with 0 first?

# COMMAND ----------

# TODO
avg_post_test_score = df.groupby('region').mean()['postTestScore']
avg_post_test_score

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO**
# MAGIC 
# MAGIC When using average in groupby, it drops null values 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Joining Multiple Data Sources

# COMMAND ----------

# MAGIC %md
# MAGIC Let's practice joining data from different sources. First, let's prepare 2 dataframes. 
# MAGIC 
# MAGIC Convert `raw_data_a` and `raw_data_b` into Pandas dataframes, and display each. You'll notice that they have a common column between them (the `id` column). We'll use this to help join data from the 2 different sources.
# MAGIC 
# MAGIC **For the following exercises, treat dataframe a as the left table and dataframe b as the right table**

# COMMAND ----------

# TODO
raw_data_a = {
  'first_name': ['Jason', 'Tina', 'Jake', 'Amy', 'Ellen'], 
  'last_name': ['Miller', 'Ali', 'Milner', 'Cooze', 'Smith'], 
  'age': [42, 36, 24, 73, 55], 
  'region': ['Northeast', 'Midwest', 'South', 'Northeast', 'Midwest'],
  'id': [111, 102, 304, 213, 402]
}

df_a = pd.DataFrame(raw_data_a)
df_a

# COMMAND ----------

# TODO
raw_data_b = {
  'preTestScore': [4, 2, 3, 5],
  'postTestScore': [25, 70, 50, 55],
  'id': [111, 304, 213, 505]
}

df_b = pd.DataFrame(raw_data_b)
df_b

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's do a left join between dataframe a and dataframe b.
# MAGIC - This is where all the data from the left table is present, and we only retrieve relevant data from the right table
# MAGIC - Hint: use the `merge()` example [here](https://pandas.pydata.org/docs/user_guide/merging.html#brief-primer-on-merge-methods-relational-algebra) where the parameter is `how="left"`
# MAGIC - You'll want to join on the `id` column
# MAGIC - You should see:
# MAGIC     - 5 rows total
# MAGIC     - The columns from the left table fully populated (no nulls)
# MAGIC     - The columns from the right table only partially populated (3 non-null datapoints)

# COMMAND ----------

### TODO: Merge with left join
pd.merge(df_a,df_b, how='left',on ='id')

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try a right join between dataframe a and dataframe b.
# MAGIC 
# MAGIC - This is where all the data from the right table is present, and we only retrieve relevant data from the left table
# MAGIC - The only difference should by the value you provide for the `how` parameter
# MAGIC - You should see:
# MAGIC     - 4 rows total
# MAGIC     - The columns from the right table fully populated (no nulls)
# MAGIC     - The columns from the left table only partially populated (3 non-null datapoints)

# COMMAND ----------

### TODO: Merge with right join
pd.merge(df_a,df_b, how='right',on ='id')

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try an inner join between dataframe a and dataframe b.
# MAGIC 
# MAGIC - This is where we only include data from both left and right tables that have a shared id (NO NULL DATA)
# MAGIC - You should see:
# MAGIC     - 3 rows total
# MAGIC     - All columns fully populated (no nulls)

# COMMAND ----------

### TODO: Merge with inner join
pd.merge(df_a,df_b, how='inner',on ='id')

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, let's try an outer join between dataframe a and dataframe b.
# MAGIC 
# MAGIC - This is where we include all data from both left and right tables (even if they don't have a shared id).
# MAGIC - You should see:
# MAGIC     - 6 rows total
# MAGIC     - The columns from the left table partially populated (5 non-null datapoints)
# MAGIC     - The columns from the right table partially populated (4 non-null datapoints)

# COMMAND ----------

### TODO: Merge with outer join
pd.merge(df_a,df_b, how='outer',on ='id')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submission
# MAGIC 
# MAGIC Download completed **Week3_PythonProgramming_Homework.ipynb** from Google Colab and commit to your personal Github repo you shared with the faculty.
