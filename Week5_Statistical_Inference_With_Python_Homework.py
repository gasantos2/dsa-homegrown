# Databricks notebook source
# MAGIC %md
# MAGIC # Week5 Statistical Inference with Python

# COMMAND ----------

# MAGIC %md
# MAGIC In week 5, we've covered:
# MAGIC * **Probability Distributions**:
# MAGIC     * Binomial Distribution
# MAGIC     * Beta Distribution
# MAGIC * **Inference**:
# MAGIC     * Estimating a batting average from data
# MAGIC * **Variance**:
# MAGIC     * measuring uncertainty
# MAGIC     * Emprical Bayes estimation

# COMMAND ----------

# MAGIC %md
# MAGIC The best way to consolidate the knowledge in your mind is by practicing.<br>Please complete the part marked with <span style="color:green">**# TODO**</span>.
# MAGIC 
# MAGIC [Google](www.google.com) and [Python Documentation](https://docs.python.org/3/contents.html) are your good friends if you have any python questions.
# MAGIC 
# MAGIC Upload **Week5_Statistical_Inference_With_Python_Homework.ipynb** notebook to your Google Drive and open it with Google Colab

# COMMAND ----------

# MAGIC %md
# MAGIC ## Probability Distributions  
# MAGIC 
# MAGIC A probability distribution is a function that gives a probability to any event that might occur in an experiment. The simplest probability distribution is the binomial distribution, which can take on *n* outcomes. A 6-sided dice follows a binomial distribution with 6 possible outcomes, each side having some probability of landing up, and all the probabilities of all the sides sum to 1. An RV example of an event that follows the binomial distribution is the probability of someone converting. We can use the conversion rate, a value between 0 and 1, to describe the probability that any individual will convert. Note that the events will always be 0, didn't convert, or 1, but the conversion rate will take on a rational value between 0 and 1.
# MAGIC 
# MAGIC For more details of the math of the binomial distribution, watch this Khan Academy video: https://www.khanacademy.org/math/statistics-probability/random-variables-stats-library/binomial-random-variables/v/binomial-distribution  
# MAGIC 
# MAGIC Let's start by simulating some conversion events using python.

# COMMAND ----------

import numpy as np

number_of_events = 1
conversion_rate = 0.3
number_of_trials = 10

# COMMAND ----------

# MAGIC %md
# MAGIC Above, we have set up some parameters to simulate 10 customers, each with a probability = 3/10 of converting. Before running the below code, make a prediction for how many `1`s you expect to see. After you've written your prediction, run the cell several times and record the output in comments in the cell below.

# COMMAND ----------

# TODO: write your prediction here: 3

sample = np.random.binomial(n = number_of_events, p = conversion_rate, size = number_of_trials)
print(sample)
print("total conversions: ", sample.sum())

# COMMAND ----------

# TODO: Write down what you observed. How did your observations compare to your prediction?
## The number of predictions was higher than I predict, maybe because of the error, and low amount of trials.

# COMMAND ----------

# MAGIC %md
# MAGIC As the number of trials increases, we would expect the actual proportion of 1s to approach the "true" conversion rate,  which in this case is 0.3 by our design. This phenomenon is called the "Law of Large Numbers." In the code block below, create a short experiment that demonstrates this phenomenon.
# MAGIC 
# MAGIC Note that if you have a numpy array named `sample` you can get its proportion by using `sample.mean()`

# COMMAND ----------

# TODO: Write a short experiment that demonstrates the Law of Large Numbers.
number_of_trials_2 = 100000000
sample_2 = np.random.binomial(n = 1, p = 0.3, size = number_of_trials_2)
print("total conversions: ", sample_2.sum())
print(sample_2.sum()/number_of_trials_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference  
# MAGIC 
# MAGIC In the above, we were told the true conversion rate by some all-knowing oracle, and then we explored how conversions might play out over a set of *n* trials. Typically, we are tasked with the reverse problem. That is, we are given access to some observed data, and asked to infer what the true conversion rate is. This process is called "statistical inference" and it is a core concept to all of data science and machine learning. We will spend some time over the next weeks developing your python skills to explore statistical inference.  
# MAGIC 
# MAGIC We will be leveraging a well known data science communicator, David Robinson, and will read the first four articles of this series: http://varianceexplained.org/r/simulation-bayes-baseball/  
# MAGIC 
# MAGIC The coding language of the series is R, a statistical programming language, but you will be asked to recreate and explore some of the ideas here in python. Let's start by understanding a second probability distribution, the beta distribution. This probability distribution is closely related to the binomial distribution we explored above. In short, it can be used to describe what the probability of the true conversion rate is, given some data. Read more about it in the first article of the series: http://varianceexplained.org/statistics/beta_distribution_and_baseball/
# MAGIC 
# MAGIC ### Exercises  
# MAGIC 
# MAGIC 1. In the code block below, write a short simulation that samples batting averages from a Beta Distribution with parameters $\alpha = 81$ and $\beta = 219$ You will likely find this numpy method to be helpful: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html

# COMMAND ----------

# TODO
alpha = 81
beta = 219
np.random.beta(a = alpha, b = beta)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. What are the lowest and highest batting averages from your experiment? What is the median batting average from your experiment? How does the median batting average from your experiment relate to the parameters alpha=81 and beta=219?

# COMMAND ----------

# TODO
import matplotlib.pyplot as plt
from scipy.stats import beta

##Ploting a beta distribuition
fig, ax = plt.subplots(1, 1)
a, b = 81, 219
mean, var, skew, kurt = beta.stats(a, b, moments='mvsk')
x = np.linspace(beta.ppf(0.01, a, b),
                beta.ppf(0.99, a, b), 100)
ax.plot(x, beta.pdf(x, a, b),
       'r-', lw=5, alpha=0.6, label='beta pdf')

min_batting_avg = min(x)
max_batting_avg = max(x)
median_batting_avg = np.median(x)
mean = a/(a+b)

print(f"The lowest battin average is {min_batting_avg}, the highest is {max_batting_avg}, median is {median_batting_avg}")
print(f"The median relate to the mean of my experiment a/(a+b) = {mean}")

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Sample a single batting average from your experiment at random, and then use that batting average to simulate 300 at bats. Give a statistical summary of the 300 at bats.
# MAGIC 
# MAGIC     Hint: When you have a single probablity of success (p) and a sequence of 300 Bernoulli(p) trials, what distribution should you sample from?
# MAGIC 
# MAGIC     Hint: What is a statistical summary? What 2 values can I usually use to describe a probability distribution?

# COMMAND ----------

# TODO
import random
single_sample = random.choice(x)

# COMMAND ----------

number_of_events = 1
conversion_rate = single_sample
number_of_trials = 300

sample = np.random.binomial(n = number_of_events, p = conversion_rate, size = number_of_trials)

hits = sample.sum()
misses = len(sample) - sample.sum()
print("total hits: ", hits)
print("total misses: ", misses )

# COMMAND ----------

hits/(hits+misses)
## Two values that we can describe the probability is hits and misses (alpha and beta)

# COMMAND ----------

# MAGIC %md
# MAGIC 4. As in the article, lets start with the $Beta(81, 219)$ distribution as the starting point of our estimate for a player's batting average at the start of a season. Then we will simulate a 300 at-bat season where a player gets 100 hits. 
# MAGIC 
# MAGIC     We will use the 300 new data points to *update* our initial best guess based on the prior information $Beta(81, 219)$
# MAGIC 
# MAGIC     Finally, we will analyze our updated estimate for the batting average of the simulated player at the end of the season. Fill in the code below to complete the above.

# COMMAND ----------

# constants, the initial parameters of the beta distribution
ALPHA0 = 81
BETA0 = 219  
INITIAL_BETA = dict(alpha=ALPHA0, beta=BETA0)

def update_beta(alpha0, beta0, hits, at_bats):  
    '''
    Parameters:
        alpha0, int: the initial number of success
        beta0, int: the initial number of failures
        hits, int: the number of hits
        at_bats, int: the number of at bats
        
    Return:
        dictionary with two members alpha and beta, each representing the updated successes and failures respectively.
    '''
    # TODO FILL IN THE CODE
    alpha = ALPHA0 + hits
    beta = BETA0 + at_bats - hits
    
    return(dict(alpha=alpha,beta=beta))
    pass

# unit test. If you've written your function correctly, the following code should give the expected outputs
hits = 100
at_bats = 300
updated_beta = update_beta(alpha0 = ALPHA0, beta0 = BETA0, hits = hits, at_bats = at_bats)
print(updated_beta['alpha']) # should equal 181
print(updated_beta['beta']) # should equal 419

# COMMAND ----------

# MAGIC %md
# MAGIC Using code and text, give a brief statistical analysis of the updated probability distribution. If the manager asked you to give an estimate for the player's batting average at the end of the season, what answer would you give?

# COMMAND ----------

# TODO: Code
new_alpha, new_beta =updated_beta['alpha'], updated_beta['beta']
player_batting_average = new_alpha/(new_alpha+new_beta)
print('The new player´s batting average would be:',player_batting_average)

# COMMAND ----------

#Old average
alpha/(alpha+beta)

# COMMAND ----------

# MAGIC %md
# MAGIC TODO: Considering that we have new values of alpha and beta, we can estimate the new average after 300 bats, which is better than the first bat (0.27)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variance  
# MAGIC 
# MAGIC So far, we've done some work to understand probability distributions, but we haven't taken advantage of the most important part - variance. Variance is a measure of uncertainty, it's a way to quantify everything you might *not* know about your estimate. Read the second article to begin to understand how variance can be used to leverage uncertainty: http://varianceexplained.org/r/empirical_bayes_baseball/

# COMMAND ----------

import pandas as pd  

# prepare the dataset  
batting_url = 'https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Batting.csv'
pitching_url = 'https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Pitching.csv'
people_url = 'https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/People.csv'

batting = pd.read_csv(batting_url)
pitching = pd.read_csv(pitching_url)
master = pd.read_csv(people_url)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercises  
# MAGIC 
# MAGIC 5. Use pandas methods to recreate the `career` data set from the article. You may find the pandas docs useful: https://pandas.pydata.org/pandas-docs/version/0.25.3/#  
# MAGIC 
# MAGIC     Focus on the logical steps taken in the article text and R code and create the appropriate pandas syntax.

# COMMAND ----------

# TODO: recreate career dataframe

pitchers = pitching['playerID'].tolist()
batting = batting[batting['AB'] > 0]
batting = batting[~batting['playerID'].isin(pitchers)] 
batting_sum =batting.groupby(['playerID']).agg({'H':'sum','AB':'sum'}) 
batting_sum['Avg'] = batting_sum.loc[:,'H'] / batting_sum.loc[:,'AB'] 
career = pd.merge(batting_sum, master, how='inner', on='playerID')[["playerID","nameFirst", "nameLast", "H", "AB", "Avg"]]

career.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 6. display the 5 highest and 5 lowest batting averages from the career dataset.

# COMMAND ----------

## Lowest batting averages
career.sort_values('Avg').head(5)

# COMMAND ----------

## Highest batting averages
career.sort_values('Avg', ascending = False).head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Note: Later in the article - he recommends filtering down to batters with >500 at bats. Do this now to roughly match the alpha and beta values in the article.

# COMMAND ----------

# TODO
career= career[career["AB"]>=500]

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Calculate the mean and variance of the empirical batting averages in the dataset

# COMMAND ----------

# TODO  

empirical_mean = career['Avg'].mean()
empirical_variance = career['Avg'].var()

print(empirical_mean, empirical_variance)

# COMMAND ----------

# MAGIC %md
# MAGIC 8. Write functions to calculate alpha and beta using the method of moments and your estimated mean and variance above. https://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance  

# COMMAND ----------

# TODO  
def estBetaParams(mean, var):
  alpha = ((1-mean)/var-1 / mean) * mean**2
  beta = alpha * (1/mean - 1)
  return (alpha,beta)

# COMMAND ----------

# MAGIC %md
# MAGIC 9. Use your function to calculate an $\alpha_0$ and $\beta_0$ for the career data set, and then calculate an empirical bayes estimate of each players batting average. Display the top 5 and bottom 5 estimated batting averages.

# COMMAND ----------

# TODO
estBetaParams(empirical_mean, empirical_variance)

# COMMAND ----------

## Finding alpha and beta for the data set
alpha, beta = estBetaParams(empirical_mean, empirical_variance)

## Estimating batting average for each player
career['estimate_batting_avg'] = ((career.loc[:,'H'] + alpha) / (career.loc[:,'AB'] + alpha + beta))

# COMMAND ----------

## Displaying the top 5
career.sort_values('estimate_batting_avg', ascending=False).head(5)

# COMMAND ----------

## Displaying the bottom 5
career.sort_values('estimate_batting_avg').head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submission
# MAGIC 
# MAGIC Download completed **Week5_Statistical_Inference_With_Python_Homework.ipynb** from Google Colab and commit to your personal Github repo you shared with the faculty.