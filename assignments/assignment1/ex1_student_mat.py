
# coding: utf-8

import thinkstats2
import thinkplot

from scipy import stats
import numpy as np
import pandas as pd

famList = list()
romanList = list()
data = open("student-mat.csv", "r")
# read famsize and romantic data into list
for line in data:
  spLine = line.split(",")
  famList.append(spLine[4])
  romanList.append(spLine[22])
# remove first element 
famList = famList[1:]
romanList = romanList[1:]

# calculate percentage of family size have three or less family members
famLE3 = famList.count("LE3")/float(len(famList));
print('family has three or less members percentage=', "{:.2f}".format(famLE3))
# calculate student in relationship percentage
romanticY = romanList.count("yes")/float(len(romanList))
print('student in relationship percentage=', "{:.2f}".format(romanticY))

famSizeHist = thinkstats2.Hist(famList, label='famsize')
romanList = thinkstats2.Hist(romanList, label='romantic')

# plot familiy size histogram
thinkplot.Hist(famSizeHist)
thinkplot.Show(xlabel='Value', ylabel='Frequency', title='Family Size Fig')
# plot romantic interest histogram
thinkplot.Hist(romanList)
thinkplot.Show(xlabel='Value', ylabel='Frequency', title='Romantic Interest Fig')

# Use One Sample T Test to valuate whether this data set is a good sample or not.
# Our null hypothesis is that: true_mu = 0
famList = map(lambda x: 1 if x == 'GT3' else 0, famList)
romanList = map(lambda x: 1 if x == 'yes' else 0, romanList)
true_mu = 0

print('family size: t-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_1samp(famList, true_mu))
print('romantic: t-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_1samp(romanList, true_mu))
print('t-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_ind(famList, romanList, equal_var=False))

# Probability Mass Functions.
famPmf = thinkstats2.Pmf(famList)
romanPmf = thinkstats2.Pmf(romanList)

famHist = thinkstats2.Hist(famPmf, label='famsize')
romanHist = thinkstats2.Hist(romanPmf, label='romantic')

# Plot family size Pmf
thinkplot.Hist(famHist)
thinkplot.Show(xlabel='familySize', ylabel='probability', title='Family Size PMF Fig')

# Plot romantic Pmf
thinkplot.Hist(romanHist)
thinkplot.Show(xlabel='romantic', ylabel='probability', title='Family Size PMF Fig')