# Databricks notebook source
# MAGIC %md
# MAGIC ## Week2 Python Programming

# COMMAND ----------

# MAGIC %md
# MAGIC In week2, we've covered:
# MAGIC * Creating **variables** 
# MAGIC * Different **data types**:
# MAGIC     * int
# MAGIC     * float
# MAGIC     * boolean
# MAGIC     * string
# MAGIC * Import **standard library**:
# MAGIC     * math
# MAGIC     
# MAGIC     
# MAGIC * **Data structures**:
# MAGIC     * List
# MAGIC     * Tuples
# MAGIC     * Dictionary
# MAGIC * **Function**:
# MAGIC     * Define function without parameter
# MAGIC         * **def** func():
# MAGIC     * Define function with parameter
# MAGIC         * **def** func(param):
# MAGIC * **Control flow**:
# MAGIC     * if, elif, else
# MAGIC     * for
# MAGIC     * while
# MAGIC * **DataFrame**:
# MAGIC     * Exploring a dataset

# COMMAND ----------

# MAGIC %md
# MAGIC The best way to consolidate the knowledge in your mind is by practicing.<br>Please complete the part marked with <span style="color:green">**# TODO**</span>.
# MAGIC 
# MAGIC [Google](https://google.com) and [Python Documentation](https://docs.python.org/3/contents.html) are your good friends if you have any python questions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variables and types

# COMMAND ----------

# MAGIC %md
# MAGIC ### Symbol names

# COMMAND ----------

# MAGIC %md
# MAGIC Variable names in Python can contain alphanumerical characters `a-z`, `A-Z`, `0-9` and some special characters such as `_`. Normal variable names must start with a letter. 
# MAGIC 
# MAGIC By convention, variable names start with a lower-case letter, and Class names start with a capital letter. 
# MAGIC 
# MAGIC In addition, there are a number of Python keywords that cannot be used as variable names. These keywords are:
# MAGIC 
# MAGIC     and, as, assert, break, class, continue, def, del, elif, else, except, 
# MAGIC     exec, finally, for, from, global, if, import, in, is, lambda, not, or,
# MAGIC     pass, print, raise, return, try, while, with, yield

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assignment

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The assignment operator in Python is `=`. Python is a dynamically typed language, so we do not need to specify the type of a variable when we create one.

# COMMAND ----------

# MAGIC %md
# MAGIC Assign an `int` to a variable with name `x` and print its type.

# COMMAND ----------

# TODO
x = 1
print(type(x))

# COMMAND ----------

# MAGIC %md
# MAGIC Assign a new value with `string` type to the same variable `x` and print its type.

# COMMAND ----------

# TODO
x = 'string'
print(type(x))

# COMMAND ----------

# MAGIC %md
# MAGIC Print the value of a variable that has not been defined, see what error do you get:

# COMMAND ----------

# TODO
print(y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Operators and comparisons

# COMMAND ----------

# MAGIC %md
# MAGIC Most operators and comparisons in Python work as one would expect:
# MAGIC 
# MAGIC * Arithmetic operators `+`, `-`, `*`, `/`, `//` (integer division), '**' power

# COMMAND ----------

# MAGIC %md
# MAGIC Print the sum, difference, multiplication, and division between `1` and `2`.

# COMMAND ----------

# TODO  
sum12 = 1+2
dif12 = 1-2
mul12 = 1*2
div12 = 1/2

print(sum12)
print(dif12)
print(mul12)
print(div12)

# COMMAND ----------

# MAGIC %md
# MAGIC Carry out Integer division of two float numbers.

# COMMAND ----------

# TODO  
float1 = 1.0
float2 = 3.0

print(float1/float2)

# COMMAND ----------

# MAGIC %md
# MAGIC What is the value of `2` Power `15`?

# COMMAND ----------

# TODO
print(2**15)

# COMMAND ----------

# MAGIC %md
# MAGIC * The boolean operators are spelled out as the words `and`, `not`, `or`.  

# COMMAND ----------

# MAGIC %md
# MAGIC * Comparison operators `>`, `<`, `>=` (greater or equal), `<=` (less or equal), `==` equality, `is` identical.

# COMMAND ----------

# MAGIC %md
# MAGIC Compare two boolean `True` and `False`

# COMMAND ----------

# TODO
print(True == False)

# COMMAND ----------

# MAGIC %md
# MAGIC Use `False` to get `True`

# COMMAND ----------

# TODO
x = False
print(not x)

# COMMAND ----------

# MAGIC %md
# MAGIC Include both `False` and `True` in your code to get `True`

# COMMAND ----------

# TODO
print(False < True)

# COMMAND ----------

# MAGIC %md
# MAGIC Use two different ways: `==` and `is` to check if `l1` and `l2` is identical.

# COMMAND ----------

# TODO
l1 = [1, 2]
l2 = [1, 2.0]

## First method
print(l1 == l2)
## Second method
print(l1 is l2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Strings

# COMMAND ----------

# MAGIC %md
# MAGIC Strings are the variables type that is used for storing text messages.

# COMMAND ----------

# MAGIC %md
# MAGIC Create three string variables, and use one `print` to display all three variables.

# COMMAND ----------

# TODO
s1 = 'first'
s2 = 'second'
s3 = 'third'
print(s1,s2,s3)

# COMMAND ----------

# MAGIC %md
# MAGIC Check the length of "Hello world!".

# COMMAND ----------

# TODO
len("Hello world!")

# COMMAND ----------

# MAGIC %md
# MAGIC Get the index `0`character in a string.

# COMMAND ----------

# TODO
s ='string'
s[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Get the first 3 characters in a string.

# COMMAND ----------

# TODO
s ='string'
s[:3]

# COMMAND ----------

# MAGIC %md
# MAGIC Use a start index and a stop index to get the fifth character in a string.

# COMMAND ----------

# TODO
s = "Red Ventures"
s[:6][-1]

# COMMAND ----------

# MAGIC %md
# MAGIC Get the last character in a string.

# COMMAND ----------

# TODO
s = "Charlotte"
s[-1]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Escape characters
# MAGIC In Python strings, the backslash "\\" is a special character, also called the **"escape"** character. It is used in representing certain whitespace characters: "\t" is a tab, "\n" is a newline, and "\r" is a carriage return. Conversely, prefixing a special character with "\" turns it into an ordinary character.

# COMMAND ----------

# MAGIC %md
# MAGIC Print out the following strings:
# MAGIC <pre>
# MAGIC '    Everything is written in pencil.'
# MAGIC 
# MAGIC 'Everything is written
# MAGIC  in pencil.'
# MAGIC  
# MAGIC 'Everything \ is written \ in pencil.'
# MAGIC 
# MAGIC 'Red Ventures brands, just list a few:
# MAGIC    * Bankrate
# MAGIC    * Reviews.com
# MAGIC    * the Points Guy
# MAGIC    * creditcards.com'
# MAGIC </pre>

# COMMAND ----------

# TODO
print('\tEverything is written in pencil.')
print('Everything is written\n in pencil.')
print('Red Ventures brands, just list a few:\n \rBankrate')

# COMMAND ----------

# MAGIC %md
# MAGIC #### String methods
# MAGIC Here are some of the most common string methods. A method is like a function, but it runs "on" an object. If the variable s is a string, then the code s.lower() runs the lower() method on that string object and returns the result (this idea of a method running on an object is one of the basic ideas that make up Object Oriented Programming, OOP). Here are some of the most common string methods:
# MAGIC 
# MAGIC lower(), upper(), strip(), isdigit(), startswith(), endswith(), find(), replace(), split(), join()
# MAGIC 
# MAGIC [Documentation](https://docs.python.org/3/library/stdtypes.html#string-methods)

# COMMAND ----------

# MAGIC %md
# MAGIC Convert all letters to lower cases.

# COMMAND ----------

# TODO
s = 'RED FISH'
print(s.lower())

# COMMAND ----------

# MAGIC %md
# MAGIC Check if string starts with `r`

# COMMAND ----------

# TODO
s = 'Run Up Escalators'
print(s.startswith('r'))

# COMMAND ----------

# MAGIC %md
# MAGIC Find the index of the single quote.

# COMMAND ----------

# TODO
s = "You don't have to lie."
print(s.find("'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Type casting
# MAGIC The process of converting the value of one data type to another is called type casting or type conversion.

# COMMAND ----------

# MAGIC %md
# MAGIC Cast variable `z` to `int` and print its type.

# COMMAND ----------

# TODO
z = '4'
print(type(z))
z = int(z)
print(type(z))

# COMMAND ----------

# MAGIC %md
# MAGIC Are the types of 1 + 2 and 1.0 + 2 the same?

# COMMAND ----------

# TODO
print(type(1+2) == type(1.0+2))

# COMMAND ----------

# MAGIC %md
# MAGIC Add num_int and num_str, you should provide two outputs. One is a string and the other is an int.

# COMMAND ----------

# TODO
num_int = 123
num_str = "456"

# string 
print(str(num_int) + num_str)

# integer
print(num_int + int(num_str))

# COMMAND ----------

# MAGIC %md
# MAGIC What happens if you cast `-1` to a boolean?

# COMMAND ----------

# TODO
print(bool(-1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standard Library
# MAGIC Python’s standard library is very extensive, offering a wide range of facilities. The library contains built-in modules (written in C) that provide access to system functionality as well as modules written in Python that provide standardized solutions for many problems that occur in everyday programming. 
# MAGIC 
# MAGIC For example, the `math` module gives access to the underlying C library functions for floating point math.
# MAGIC 
# MAGIC [Documentation](https://docs.python.org/3/library/math.html)
# MAGIC <br>
# MAGIC <br>

# COMMAND ----------

# MAGIC %md
# MAGIC Import math module and calculate $e^4$.

# COMMAND ----------

# TODO
import math

print(math.exp(4))

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the value of $log_2(8)$.

# COMMAND ----------

# TODO
print(math.log2(8))

# COMMAND ----------

# MAGIC %md
# MAGIC Calculate the value of $\sqrt{100}$.

# COMMAND ----------

# TODO
print(math.sqrt(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## List
# MAGIC [List Documentation](https://www.w3schools.com/python/python_lists.asp)  

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Fill the ___ parts of code below

# COMMAND ----------

# TODO
# Let's create an empty list
my_list = [] 

# Let's add some values
my_list.append('Python')
my_list.append('is ok')
my_list.append('sometimes')

# Let's remove 'sometimes'
my_list.remove('sometimes')

# Let's change the second item
my_list[1] = 'is neat'

# COMMAND ----------

# Let's verify that it's correct
assert my_list == ['Python', 'is neat']

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Create a new list without modifying the original one

# COMMAND ----------

original = ['I', 'am', 'learning', 'hacking', 'in']

# COMMAND ----------

# TODO
# TODO
modified = original.copy()
modified[original.index('hacking')] = 'lists'
modified.append('Python')

# COMMAND ----------

assert original == ['I', 'am', 'learning', 'hacking', 'in']
assert modified == ['I', 'am', 'learning', 'lists', 'in', 'Python']

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Create a merged sorted list

# COMMAND ----------

list1 = [6, 12, 5]
list2 = [6.2, 0, 14, 1]
list3 = [0.9]

# COMMAND ----------

# TODO
my_list = sorted(list1 + list2 + list3, reverse=True)

# COMMAND ----------

print(my_list)
assert my_list == [14, 12, 6.2, 6, 5, 1, 0.9, 0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Loop through a list  
# MAGIC [For Loops Documentation](https://www.w3schools.com/python/python_for_loops.asp)  
# MAGIC [While Loops Documentation](https://www.w3schools.com/python/python_while_loops.asp)

# COMMAND ----------

# MAGIC %md
# MAGIC Loop through a list and check:
# MAGIC * If an item is a float, cast the float to int.
# MAGIC * If an item is an int, cast the int to float
# MAGIC * Hint: use isinstance()

# COMMAND ----------

loop_list1 = [1101, 704.971, 'Fort Mill']
# TODO 
# use for loop
for index, value in enumerate(loop_list1):
    if isinstance(value, float):
        loop_list1[index] = int(value)
    elif isinstance(value, int):
        loop_list1[index] = float(value)
    else:
        pass


# COMMAND ----------

print(loop_list1)
assert loop_list1 == [1101.0, 704, 'Fort Mill']

# COMMAND ----------

loop_list2 = [1101, 704.971, 'Fort Mill']
# TODO
# use while loop
i = 0
while i < len(loop_list2):
    item = loop_list2[i]
    print(item)
    
    if type(item) == float:
        loop_list2[i] = int(item)
    elif type(item) == int:
        loop_list2[i] = float(item)
    
    i = i + 1

# COMMAND ----------

print(loop_list2)
assert loop_list2 == [1101.0, 704, 'Fort Mill']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuples
# MAGIC [Tuples Documentation](https://www.w3schools.com/python/python_tuples.asp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Fill the ___ parts of code below

# COMMAND ----------

# TODO
# create a tuple with items of different data types
my_tuple = (6,'6',6.0)
print(my_tuple, type(my_tuple))

# Unpack my_tuple in several variables
item1, item2, item3 = my_tuple

# convert my_tuple to list and assign to variable my_list
my_list = list(my_tuple)
print(my_list, type(my_list))

# convert my_list to tuple and assign to variable my_new_tuple
my_new_tuple = tuple(my_list)
print(my_new_tuple, type(my_new_tuple))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Check the immutable property of tuple

# COMMAND ----------

# TODO
rv_tuple = 1101, 'Red', 'Ventures', 'Drive'
print(rv_tuple, type(rv_tuple))
# check id
print(id(rv_tuple))
# modify the variable
rv_tuple = rv_tuple + (29707, )
print(rv_tuple, type(rv_tuple))
# check id
print(id(rv_tuple))

# COMMAND ----------

# TODO
# Do the same exercises as above for a list and see the difference
# Remember list is mutable
rv_list = [1101, 'Red', 'Ventures', 'Drive']
print(rv_list, type(rv_list))
print(id(rv_list))
rv_list.append(29707)
print(rv_list, type(rv_list))
print(id(rv_list))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Define Functions  
# MAGIC 
# MAGIC [Functions Documentation](https://www.w3schools.com/python/python_functions.asp)
# MAGIC 
# MAGIC Write a function that accepts a string and return the number of uppercase letters and lowercase letters.

# COMMAND ----------

# TODO
def check_case(string):
  uppercase_number = sum([x.isupper() for x in string])
  lowercase_number = sum([x.islower() for x in string])
  
  return uppercase_number, lowercase_number

# COMMAND ----------

my_string = "This is Week3 Python Programming Homework."
assert check_case(my_string)[0] == 5
assert check_case(my_string)[1] == 30

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dictionary
# MAGIC [Dictionary Documentation](https://www.w3schools.com/python/python_dictionaries.asp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Fill the ___ parts of code below

# COMMAND ----------

# TODO
# create an empty dictionary
my_dict = {}
print(my_dict, type(my_dict))

# create a dictionary with values
my_dict = {
  'value1': 1,
  'value2': 2,
}
print(my_dict, type(my_dict))

# add key/value to my_dict
my_dict['value3']=3
print(my_dict)

# remove the key you added from my_dict
my_dict.pop('value3')
print(my_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Sort Dictionary
# MAGIC 
# MAGIC Print out key/value pair by the ascending order of keys.

# COMMAND ----------

# TODO
location_dict = {'Charlotte': 'NC', 'Austin': 'TX', 'New York': 'NY', 'Los Angeles': 'CA'}
keys = location_dict.keys()

for key in sorted(keys):
    print(f'Key:{key}\tValue:{location_dict[key]}')

# COMMAND ----------

# MAGIC %md
# MAGIC Display the location_dict by the descending order of values.

# COMMAND ----------

# TODO
location_dict = {'Charlotte': 'NC', 'Austin': 'TX', 'New York': 'NY', 'Los Angeles': 'CA'}
keys = list(location_dict.keys())
values = list(location_dict.values())

for value in sorted(values, reverse = True):
    print(f'Key:{keys[values.index(value)]}\tValue:{value}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set

# COMMAND ----------

# MAGIC %md
# MAGIC Though `set` is not covered in the video materials, it is also one of the major data structures in Python.
# MAGIC 
# MAGIC [Set Documentation](https://www.w3schools.com/python/python_sets.asp)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Fill the ____ parts of code below

# COMMAND ----------

# TODO
# create an set
my_set = set()
print(my_set, type(my_set), '\n')

# create a set with values
my_set = {"value1","1"}
print(my_set, type(my_set), '\n')

# add a new value to my_set
# the value should be a different
my_set.add("value2")
print(my_set, '\n')


# remove the value you added from my_set
my_set.remove("value2")
print(my_set, '\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Set Manipulation

# COMMAND ----------

# MAGIC %md
# MAGIC Use a list  with duplicated values and a tuple to create two sets respectively, and get a union set between those two.

# COMMAND ----------

# TODO
new_list = [1,2,3,1,2,3,4,5]
new_tup1e = (4,5,6,7,8)

result = set.union(set(new_list), set(new_tup1e))
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Create two sets `my_set1` and `my_set2`, and get values in `my_set1` but not in `my_set2`.

# COMMAND ----------

# TODO
my_set1 = {1,2,3,4,5,6,7}
my_set2 = {4,5}
my_set = my_set1 - my_set2 
print(my_set1)
print(my_set2)
print(my_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC [Dataframe Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

# COMMAND ----------

import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Download the CSV from the above url.
# This csv is separated by a ';' so we need to add that as a parameter to the read_csv method.
df = pd.read_csv(url, sep=';')

# COMMAND ----------

# MAGIC %md
# MAGIC Show the first 6 rows of the `df` dataframe

# COMMAND ----------

# TODO
df.head(6)

# COMMAND ----------

# MAGIC %md
# MAGIC Print out the summary statistics for each dataframe column

# COMMAND ----------

# TODO
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Submission
# MAGIC 
# MAGIC Download completed **Week2_Python_Homework.ipynb** from Google Colab and commit to your personal Github repo you shared with the faculty.
