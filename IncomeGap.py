# -*- coding: utf-8 -*-
"""
################################################################################################

Created on Mon Feb 24 16:53:25 2020
@author: Sobha

################################################################################################
"""

'''
Tasks
Complete the following tasks:

1. Calculate the median income of male employees and the median income of 
   female employee in the population.(look the set of all employees in the datasets
   as the population). (1 point)

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"provide\the\path")

data = pd.read_csv("multipleChoiceResponses.csv",skiprows=[1],usecols=['Q1','Q9'])

data.head()

data.columns = ['Gender','Compensation']

data.head()

data = data[~data['Compensation'].isin(['I do not wish to disclose my approximate yearly compensation'])]

data = data[~data['Compensation'].isna()]

pd.value_counts(data['Compensation'])

def salary_correction(string):
    ran = string.split(",")[0]
    try:
        midpoint = ((int(ran.split("-")[0])+int(ran.split("-")[1]))/2)*1000
    except:
        midpoint = int(ran)*1000
    return midpoint

data = data[~data['Gender'].isin(['Prefer not to say','Prefer to self-describe'])]

pd.value_counts(data['Gender'])

data['Compensation'] = data['Compensation'].astype(str).map(salary_correction).astype(int)

pd.value_counts(data['Compensation'])

data.shape

###Gender Ratio###
data.columns = ['Gender','Ratio']
((data.groupby('Gender').count()/data.shape[0])*100).reset_index()

data.columns = ['Gender','Compensation']
data.groupby('Gender').agg({'Compensation':'median'})

pop_median = dict((data.groupby('Gender').agg({'Compensation':'median'}))['Compensation'])

pop_stat = (pop_median['Male'] - pop_median['Female'])

print("Test Statistic [Male Compensation Med - Female Compensation Med] = {0}".format(pop_stat)+" USD")

################################################################################################

'''
2. Draw an overlaid graph to show the histograms of the incomes of female and male 
   employees in the population.
(You create one histogram for male, and another histogram for female, 
but the two histograms will be displayed
in the same graph with different colors). (1 point)
'''

male_f_table = data[data['Gender']=='Male']['Compensation']

female_f_table = data[data['Gender']=='Female']['Compensation']


plt.figure(figsize=(10,5))

plt.hist(male_f_table,label='Male',bins=10)

plt.hist(female_f_table,label='Female',color='r',bins=10)

plt.xlabel ('Total Compensation')

plt.ylabel ('Frequency')

plt.title ('OVERLAID HISTOGRAM OF FEMALE AND MALE EMPLOYEES INCOME')

plt.legend(loc='upper right')

plt.show()

# Percentage Distribution

#freq/size of the bin.

plt.figure(figsize=(10,5))

plt.hist(male_f_table,density =True,label='Male',bins=10)

plt.hist(female_f_table,density =True,label='Female', color='r', bins=10)

plt.xlabel ('Total Compensation')

plt.ylabel ('Percent Per Unit')

plt.title ('OVERLAID HISTOGRAM OF FEMALE AND MALE EMPLOYEES INCOME')

plt.legend(loc='upper right')

plt.show()


################################################################################################

'''

3. Use the random sampling, empirical distribution, sample comparison, bootstrap,
 hypothesis testing as well as A/B testing we discussed in the class to analyze the 
 income gap between female and male employees.

• Select a sample from the population. Make sure your sample include 500 employees 
  selected from the population, and consider how to ensure the sampling strategy
  is fair since the datasets include overwhelmed male employees than female employees 
  (1 point).

'''

data.shape

from random import sample 

def sample_data(data,size):
    indices = sample(range(0,data.shape[0]),size)
    sample_d = data.iloc[indices]
    return sample_d

sample_d = sample_data(data,500)

((sample_d.groupby('Gender').count()/500)*100).reset_index()

###############################################################################################

'''
• Define test statistic, null hypothesis and alternative hypothesis (1 point).

Null: 
In the population, the distributions of the compensation of Male and Female Employees are same.
(Compensation they are taking is the same)
(i.e the difference in Median between them is by chance)

Alternative: 
In the population, the Median between two segments are different. 
Male Compensation higher than Female Compensation. 

Test Statistic: Difference in medians (Median of Male) - (Median of Female).
                Large test statistic favor Alternative hypothesis.

'''

###############################################################################################

'''
• Draw the income histogram for the sample, calculate the median income of the sample, and draw 
  a red dot and a yellow dot of the female median income and male median income of the population,
  respectively, in the histogram (1 point).

'''

median_op = dict((sample_d.groupby('Gender').agg({'Compensation':'median'}))['Compensation'])

print(median_op)

plt.xlabel ('Total Compensation')

plt.ylabel ('Frequency')

plt.title ('INCOME HISTOGRAM OF THE SAMPLE')

plt.hist(sample_d['Compensation'],bins=20)

plt.scatter(median_op['Female'],0, color='r',s=50)

plt.scatter(median_op['Male'],0, color='y',s=50)


###############################################################################################

'''
• Draw the histogram of the test statistic of the sample, and draw a red dot to show the 
corresponding test statistic of the population (e.g. the difference of the median incomes 
between female and male employees) in the diagram (1 point).

'''

stats = []
ind = sample(list(range(data.shape[0])),15000)

for i in range(0,15000,500):
    sample_d  = data.iloc[ind[i:i+500]]
    median_op = dict((sample_d.groupby('Gender').agg({'Compensation':'median'}))['Compensation'])
    test_stat = median_op['Male'] - median_op['Female']
    stats.append(test_stat)

plt.xlabel ('Total Compensation')

plt.ylabel ('Frequency')

plt.title ('TEST STATISTIC HISTOGRAM OF THE SAMPLE')

plt.hist(stats)

plt.scatter(pop_stat,0, color='r',s=50)


###############################################################################################

'################################## Testing Null Hypothesis ###################################' 

def shuffle_data(data):
    indices_l = sample(range(0,data.shape[0]),data.shape[0])
    data['Gender'] = data['Gender'].iloc[indices_l].values
    return data

data.head()

shuffle_data = shuffle_data(data)

shuffle_data.head()

################################################################################################

stats_null = []
ind = sample(list(range(shuffle_data.shape[0])),15000)

for i in range(0,15000,500):
    sample_d    = shuffle_data.iloc[ind[i:i+500]]
    median_op_s = dict((sample_d.groupby('Gender').agg({'Compensation':'median'}))['Compensation'])
    test_stat   = median_op_s['Male'] - median_op_s['Female']
    stats_null.append(test_stat)

plt.hist(stats_null)

plt.scatter(pop_stat,0, color='r',s=50)


' Favourable to Null Hypothesis - We reject Alt. Hypothesis '
' We conclude that this difference in Compensation between Male and Female is just by chance  '

p_value = sum(np.array(stats_null)>=pop_stat)/500

print(p_value)

################################################################################################
'''

• Write a procedure to use bootstrap to produce at least 5000 samples (1 point).

• Draw the histogram of the test statistic of the bootstrap samples (1 point).

• Define confidence interval and P-value to validate the hypothesis you defined (2 points).

'''


def generate_bootstrap(data,size,samples):
    bootstrap = []
    for i in range(0,samples):
        bootstrap.append(data['Compensation'].sample(size,replace=True).values)
    return bootstrap

bootstrap = generate_bootstrap(data,10,5000)
bootstrap 

def bootstrap_median(data,size,samples): 
  b_stats = [] 
  for i in np.arange(samples):
      sample = data.sample(size,replace=True)
      median_op_s = dict((sample.groupby('Gender').agg({'Compensation':'median'}))['Compensation'])
      test_stat   = median_op_s['Male'] - median_op_s['Female']
      b_stats.append(test_stat) 
  return b_stats

b_stats = bootstrap_median(data,500,5000)

b_stats

plt.xlabel ('Total Compensation')

plt.ylabel ('Frequency')

plt.title ('TEST STATISTIC HISTOGRAM OF THE SAMPLE USING BOOTSTRAP')

plt.hist(b_stats)

plt.scatter(pop_stat,0,color='red',s=20)

'# P-value and Confidence Interval Calculation '

p_value = sum(np.array(b_stats)>=pop_stat)/5000

print(p_value)

print("Confidence Intervals")
np.percentile(b_stats, 50)

np.percentile(b_stats, 2.5)

np.percentile(b_stats, 97.5)

'''
4. Submit Python code, the writing for explaining the data cleaning procedure, defining the 
test statistic, hypoth-esises, random sampling, bootstrap, confidential intervals, P-vales, 
as well as interpretation of your results,and all outputs described above.
'''
###submitted report
