SciPy


Sample Mean and Population Mean

The individual measurements on Monday, Tuesday, and Wednesday are called samples. A sample is a subset of the entire population. 
The mean of each sample is the sample mean and it is an estimate of the population mean.




For a population, the mean is a constant value no matter how many times its recalculated. 
But with a set of samples, the mean will depend on exactly what samples we happened to choose. 
From a sample mean, we can then extrapolate the mean of the population as a whole. 
There are many reasons we might use sampling, such as:

We do not have data for the whole population.
We have the whole population data, but it is so large that it is infeasible to analyze.
We can provide meaningful answers to questions faster with sampling.

When we have a numerical dataset and want to know the average value, we calculate the mean. 
For a population, the mean is a constant value no matter how many times it is recalculated. 
But with a set of samples, the mean will depend on exactly what samples we happened to choose. 
From a sample mean, we can then extrapolate the mean of the population as a whole.



import numpy as np

population = np.random.normal(loc=65, scale=3.5, size=300)
population_mean = np.mean(population)

print "Population Mean: {}".format(population_mean)

sample_1 = np.random.choice(population, size=30, replace=False)
sample_2 = np.random.choice(population, size=30, replace=False)
sample_3 = np.random.choice(population, size=30, replace=False)
sample_4 = np.random.choice(population, size=30, replace=False)
sample_5 = np.random.choice(population, size=30, replace=False)

sample_1_mean = np.mean(sample_1)
print "Sample 1 Mean: {}".format(sample_1_mean)
sample_2_mean = np.mean(sample_2)
sample_3_mean = np.mean(sample_3)
sample_4_mean = np.mean(sample_4)
sample_5_mean = np.mean(sample_5)

print "Sample 2 Mean: {}".format(sample_2_mean)
print "Sample 3 Mean: {}".format(sample_3_mean)
print "Sample 4 Mean: {}".format(sample_4_mean)
print "Sample 5 Mean: {}".format(sample_5_mean)








If our sample selection is poor then we will have a sample mean seriously skewed from our population mean.

Central Limit Theorem
if we have a large enough sample size, all of our sample means will be sufficiently close to the population mean





a = np.random.binomial(10, 0.30, size=10000)
# Let's generate 10,000 "experiments"
# N = 10 shots
# P = 0.30 (30% he'll get a free throw)

print(np.mean(a == 4))


np.random.normal(loc, scale, size)
a = np.random.normal(0, 1, size=100000)

loc: the mean for the normal distribution
scale: the standard deviation of the distribution
size: the number of random numbers to generate






import numpy as np

# Create population and find population mean
population = np.random.normal(loc=65, scale=100, size=3000)
population_mean = np.mean(population)

# Select increasingly larger samples
extra_small_sample = population[:10]
small_sample = population[:50]
medium_sample = population[:100]
large_sample = population[:500]
extra_large_sample = population[:1000]

# Calculate the mean of those samples
extra_small_sample_mean = np.mean(extra_small_sample)
small_sample_mean = np.mean(small_sample)
medium_sample_mean = np.mean(medium_sample)
large_sample_mean = np.mean(large_sample)
extra_large_sample_mean = np.mean(extra_large_sample)

# Print them all out!
print "Extra Small Sample Mean: {}".format(extra_small_sample_mean)
print "Small Sample Mean: {}".format(small_sample_mean)
print "Medium Sample Mean: {}".format(medium_sample_mean)
print "Large Sample Mean: {}".format(large_sample_mean)
print "Extra Large Sample Mean: {}".format(extra_large_sample_mean)

print "\nPopulation Mean: {}".format(population_mean)






Hypothesis Tests

We invite 100 men and 100 women to this class. After one week, 34 women sign up, and 39 men sign up. 
More men than women signed up, but is this a "real" difference?

"If we gave the same invitation to every person in the world, would more men still sign up?"

A more formal version is: "What is the probability that the two population means are the same and that the difference we observed in the sample means is just chance?"

These statements are all ways of expressing a null hypothesis. 

A null hypothesis is a statement that the observed difference is the result of chance.



NULL HYPOTHESIS:
The null hypothesis states that any difference observed within sample means is coincidental.



Type I Or Type II


Type I error
is finding a correlation between things that are not related. 
This error is sometimes called a "false positive" and occurs when the null hypothesis is rejected even though it is true.

For example, lets say you conduct an A/B test for an online store and conclude that interface B is significantly better than interface A at directing traffic to a checkout page. 
You have rejected the null hypothesis that there is no difference between the two interfaces. 
If, in reality, your results were due to the groups you happened to pick, and there is actually no significant difference between interface A and interface B in the greater population, you have been the victim of a false positive.


Type II error 
is failing to find a correlation between things that are actually related. 
This error is referred to as a "false negative" and occurs when the null hypothesis is accepted even though it is false.

For example, with the A/B test situation, lets say that after the test, you concluded that there was no significant difference between interface A and interface B. 
If there actually is a difference in the population as a whole, your test has resulted in a false negative.




import numpy as np

def intersect(list1, list2):
  return [sample for sample in list1 if sample in list2]

# the true positives and negatives:
actual_positive = [2, 5, 6, 7, 8, 10, 18, 21, 24, 25, 29, 30, 32, 33, 38, 39, 42, 44, 45, 47]
actual_negative = [1, 3, 4, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 26, 27, 28, 31, 34, 35, 36, 37, 40, 41, 43, 46, 48, 49]

# the positives and negatives we determine by running the experiment:
experimental_positive = [2, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 32, 35, 36, 38, 39, 40, 45, 46, 49]
experimental_negative = [1, 3, 6, 12, 14, 23, 25, 29, 30, 31, 33, 34, 37, 41, 42, 43, 44, 47, 48]

#define type_i_errors and type_ii_errors here
type_i_errors = intersect(experimental_positive, actual_negative)
type_ii_errors = intersect(experimental_negative, actual_positive)




P-Values

A p-value is the probability that the null hypothesis is true.
A higher p-value is more likely to give a false positive 
so if we want to be very sure that the result is not due to just chance, we will select a very small p-value



def accept_null_hypothesis(p_value):
  """
  Returns the truthiness of the null_hypothesis

  Takes a p-value as its input and assumes p < 0.05 is significant
  """
  if p_value > 0.05:
    return True
  else:
    return False

hypothesis_tests = [0.1, 0.009, 0.051, 0.012, 0.37, 0.6, 0.11, 0.025, 0.0499, 0.0001]

for p_value in hypothesis_tests:
    accept_null_hypothesis(p_value)










HYPOTHESIS TESTING
Types of Hypothesis Test



Some situations involve correlating numerical data:

1. a professor expects an exam average to be roughly 75%, and wants to know if the actual scores line up with this expectation. Was the test actually too easy or too hard?
2. a manager of a chain of stores wants to know if certain locations have different revenues on different days of the week. Are the revenue differences a result of natural fluctuations or a significant difference between the stores' sales patterns?
3. a PM for a website wants to compare the time spent on different versions of a homepage. Does one version make users stay on the page significantly longer?

Types of Hypothesis Test for numerical data:

One Sample T-Tests
Two Sample T-Tests
ANOVA
Tukey Tests




Others involve categorical data:

1. a pollster wants to know if men and women have significantly different yogurt flavor preferences. Does a result where men more often answer "chocolate" as their favorite reflect a significant difference in the population?
2. do different age groups have significantly different emotional reactions to different ads?



Types of Hypothesis Test for categorical data:

Binomial Tests
Chi Square
After this lesson, you will have a wide range of tools in your arsenal to find meaningful correlations in data.







One Sample T-Testing

A univariate T-test compares a sample mean to a hypothetical population mean. 
It answers the question "What is the probability that the sample came from a distribution with the desired mean?"

Null hypothesis, which is a prediction that there is no significant difference. 
The null hypothesis that this test examines can be phrased as such: "The set of samples belongs to a population with the target mean".




SciPy has a function called ttest_1samp, which performs a 1 Sample T-Test for you.

tstat, pval = ttest_1samp(example_distribution, expected_mean)
print pval



from scipy.stats import ttest_1samp
import numpy as np

correct_results = 0 # Start the counter at 0

daily_visitors = np.genfromtxt("daily_visitors.csv", delimiter=",")
correct_results = 0
for i in range(1000): # 1000 experiments
   #your ttest here:
    tstat, pval = ttest_1samp(daily_visitors[i], 30)
    if pval <0.05:
      correct_results += 1
  
print "We correctly recognized that the distribution was different in " + str(correct_results) + " out of 1000 experiments."
print "We correctly recognized that the distribution was different in " + str(correct_results) + " out of 1000 experiments."
print correct_results



We correctly recognized that the distribution was different in 499 out of 1000 experiments.
We correctly recognized that the distribution was different in 499 out of 1000 experiments.
499 #If we get a pval < 0.05, we can conclude that it is unlikely that our sample has a true mean of 30. Thus, the hypothesis test has correctly rejected the null hypothesis, and we call that a correct result.









Two Sample T-Test


Suppose that last week, the average amount of time spent per visitor to a website was 25 minutes. 
This week, the average amount of time spent per visitor to a website was 28 minutes. 
Did the average time spent per visitor change? Or is this part of natural fluctuations?


A 2 Sample T-Test compares two sets of data, which are both approximately normally distributed.

The null hypothesis, in this case, is that the two distributions have the same mean.




from scipy.stats import ttest_ind
import numpy as np

week1 = np.genfromtxt("week1.csv",  delimiter=",")
week2 = np.genfromtxt("week2.csv",  delimiter=",")

week1_mean = np.mean(week1)
week2_mean = np.mean(week2)


week1_std = np.std(week1)
week2_std = np.std(week2)


tstatstic, pval = ttest_ind(week1, week2)

print pval
>>> 0.000676767690007






from scipy.stats import ttest_ind
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

a_mean = np.mean(a)
b_mean = np.mean(b)
c_mean = np.mean(c)

a_std = np.std(a)
b_std = np.std(b)
c_std = np.std(c)

a_b_pval = ttest_ind(a, b).pvalue
a_c_pval = ttest_ind(a, c).pvalue
b_c_pval = ttest_ind(b, c).pvalue

error_prob = (1-(0.95**3))   # the probability of error in a variable

'''
For a p-value of 0.05, if the null hypothesis is true then the probability of obtaining a significant result is 1 – 0.05 = 0.95. 
When we run another t-test, the probability of still getting a correct result is 0.95 * 0.95, or 0.9025. 
That means our probability of making an error is now close to 10%! This error probability only gets bigger with the more t-tests we do.
'''





ANOVA


ANOVA (Analysis of Variance) tests the null hypothesis that all of the datasets have the same mean. 
If we reject the null hypothesis with ANOVA, we are saying that at least one of the sets has a different mean; 
however, it does not tell us which datasets are different.


fstat, pval = f_oneway(scores_mathematicians, scores_writers, scores_psychologists)

The null hypothesis, in this case, is that all three populations have the same mean score on this videogame. 
If we reject this null hypothesis (if we get a p-value less than 0.05), we can say that we are reasonably confident that a pair of datasets is significantly different. 
After using only ANOVA, we cannot make any conclusions on which two populations have a significant difference.



from scipy.stats import ttest_ind
from scipy.stats import f_oneway
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b_new.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

stat, pval = f_oneway(a, b, c)
print pval







Assumptions of Numerical Hypothesis Tests

1. The samples should each be normally distributed...ish
2. The population standard deviations of the groups should be equal #ratio of STD approximately 1
3. The samples must be independent


import codecademylib
import numpy as np
import matplotlib.pyplot as plt

dist_1 = np.genfromtxt("1.csv",  delimiter=",")
dist_2 = np.genfromtxt("2.csv",  delimiter=",")
dist_3 = np.genfromtxt("3.csv",  delimiter=",")
dist_4 = np.genfromtxt("4.csv",  delimiter=",")

#plot your histogram here
#plt.hist(dist_1)
#plt.hist(dist_2)
#plt.hist(dist_3)
plt.hist(dist_4)
plt.show()

not_normal = 4

ratio = np.std(dist_2) / np.std(dist_3)
print ratio






from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np


movie_scores = np.concatenate([drama_scores, comedy_scores, documentary_scores])
labels = ['drama'] * len(drama_scores) + ['comedy'] * len(comedy_scores) + ['documentary'] * len(documentary_scores)

tukey_results = pairwise_tukeyhsd(movie_scores, labels, 0.05)





from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np

a = np.genfromtxt("store_a.csv",  delimiter=",")
b = np.genfromtxt("store_b.csv",  delimiter=",")
c = np.genfromtxt("store_c.csv",  delimiter=",")

stat, pval = f_oneway(a, b, c)
print pval

# Using our data from ANOVA, we create v and l
v = np.concatenate([a, b, c])
labels = ['a'] * len(a) + ['b'] * len(b) + ['c'] * len(c)

tukey_results = pairwise_tukeyhsd(v, labels, 0.05)
print tukey_results






Binomial Test


A Binomial Test compares a categorical dataset to some expectation.

Examples include:

Comparing the actual percent of emails that were opened to the quarterly goals
Comparing the actual percentage of respondents who gave a certain survey response to the expected survey response
Comparing the actual number of heads from 1000 coin flips of a weighted coin to the expected number of heads

null_hypothesis

H0: there is no difference between the observed behavior and the expected behavior.


binom_test



binom_test requires three inputs:
1. the number of observed successes
2. the number of total trials
3. an expected probability of success

For example, with 1000 coin flips of a fair coin, we would expect a "success rate" (the rate of getting heads), to be 0.5, and the number of trials to be 1000. 
Lets imagine we get 525 heads. Is the coin weighted? This function call would look like:

pval = binom_test(525, n=1000, p=0.5)





from scipy.stats import binom_test

pval = binom_test(510,n=10000,p=0.06)
pval2 = binom_test(590,n=10000,p=0.06)


print(pval, pval2)









Chi Square Test


With three discrete categories of data per dataset, we can no longer use a Binomial Test.


Chi Square test. It is useful in situations like:

An A/B test where half of users were shown a green submit button and the other half were shown a purple submit button. Was one group more likely to click the submit button?
Men and women were both given a survey asking "Which of the following three products is your favorite?" Did the men and women have significantly different preferences?






The input to chi2_contingency is a contingency table where:

The columns are each a different condition, such as men vs. women or Interface A vs. Interface B
The rows represent different outcomes, like "Survey Response A" vs. "Survey Response B" or "Clicked a Link" vs. "Didn't Click"



Null hypothesis

H0: there is no significant difference between the datasets. 

We reject that hypothesis, and state that there is a significant difference between two of the datasets if we get a p-value less than 0.05.





from scipy.stats import chi2_contingency

# Contingency table
#         harvester |  leaf cutter
# ----+------------------+------------
# 1st gr | 30       |  10
# 2nd gr | 35       |  5
# 3rd gr | 28       |  12

X = [[30, 10],
     [35, 5],
     [28, 12]]
chi2, pval, dof, expected = chi2_contingency(X)
print pval







import familiar
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

vein_pack_lifespans=familiar.lifespans(package='vein')

artery_pack_lifespans=familiar.lifespans(package='artery')

pval = ttest_1samp(vein_pack_lifespans, 71)

print format(pval, '0.10f')

package_comparison_results = ttest_ind(vein_pack_lifespans, artery_pack_lifespans)
print(package_comparison_results.pvalue)
if pval.pvalue<0.05:
  print("The Vein Pack Is Proven To Make You Live Longer!")
else:
  print("The Vein Pack Is Probably Good For You Somehow!")
  
  
iron_contingency_table = familiar.iron_counts_for_package()

iron_pvalue = chi2_contingency(iron_contingency_table)

print(iron_pvalue)











import numpy as np
import fetchmaker
from scipy.stats import binom_test
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

rottweiler_tl=fetchmaker.get_tail_length('rottweiler')

# print(rottweiler_tl)

whippet_rescue = fetchmaker.get_is_rescue('whippet')

num_whippet_rescues = np.count_nonzero(whippet_rescue)
num_whippets = np.size(whippet_rescue)

pval = binom_test(6, n=num_whippets, p=0.08)

print pval  >>> 0.581178010624 # as the p value is greated than 0.05 we connot reject the null hypothesis that there is a significat difference in the whippet adoption rate.


w = fetchmaker.get_weight('whippet')
t = fetchmaker.get_weight('terrier')
p = fetchmaker.get_weight('pitbull')

print f_oneway(w, t, p).pvalue

values = np.concatenate([w,t,p])
lables = ['whippet'] * len(w) + ['terrier'] * len(t) + ['pitbull'] * len(p)

tukey_results = pairwise_tukeyhsd(values, labels, 0.05)

print tukey_results # if the result is True we can reject the null hiphothesis 


poodle_colors = fetchmaker.get_color('poodle')
shihtzu_color = fetchmaker.get_color('shihtzu')

color_table = [
[np.count_nonzero(poodle_colors == 'black'), np.count_nonzero(shihtzu_color == 'black')],
[np.count_nonzero(poodle_colors == 'brown'), np.count_nonzero(shihtzu_color == 'brown')],
[np.count_nonzero(poodle_colors == 'gold'), np.count_nonzero(shihtzu_color == 'gold')],
[np.count_nonzero(poodle_colors == 'grey'), np.count_nonzero(shihtzu_color == 'grey')],
[np.count_nonzero(poodle_colors == 'white'), np.count_nonzero(shihtzu_color == 'white')]
]

>>> [
     [17, 10],  # 17 poodle with color black and 10 shihtzu with color black.
     [13, 36], 
     [8, 6], 
     [52, 41], 
     [10, 7]
     ]



_, color_pval, _, _ = chi2_contingency(color_table)

print color_pval
>>> 0.00530240829324   # as the pvalue <0.05 we can say that there is a significan difference between poodle and shitzu colors.














