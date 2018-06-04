
# coding: utf-8

import thinkstats2
import thinkplot

famList = list()
romanList = list()
data = open("/Users/xwan/developers/ML/python_exercises/assignments/assignment1/student-mat.csv", "r")
# read famsize and romantic data into list
for line in data:
  spLine = line.split(",")
  famList.append(spLine[4])
  romanList.append(spLine[22])
print(famList[0], romanList[0])
# remove first element 
famList = famList[1:]
romanList = romanList[1:]


# Probability Mass Functions.
famPmf = thinkstats2.Pmf(famList)
romanPmf = thinkstats2.Pmf(romanList)
print(famPmf)
print(romanPmf)

famHist = thinkstats2.Hist(famPmf, label='famsize')
romanHist = thinkstats2.Hist(romanPmf, label='romantic')

# Plot family size Pmf
thinkplot.Hist(famHist)
thinkplot.Show(xlabel='familySize', ylabel='probability', title='Family Size PMF Fig')

# Plot romantic Pmf
thinkplot.Hist(romanHist)
thinkplot.Show(xlabel='romantic', ylabel='probability', title='Family Size PMF Fig')
