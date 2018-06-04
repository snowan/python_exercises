import thinkstats2
import thinkplot
import random

# 1. random number generator by random
# Generate 100 random number from 1 to 10
rand1List = []
for i in range (1, 10):
  rand1List.append(round(random.random() * 10))
# print(randList)

# 2. random number generator by math
import math

def drBRandom(lastNum):
  return math.cos(lastNum)

rand2List = []
ln = 0.01
for x in range(1, 10):
  ln = drBRandom(ln)
  rand2List.append(round(ln * 10))
# print(rand2List)

# make list into Pmf
rand1Pmf = thinkstats2.Pmf(rand1List)
rand2Pmf = thinkstats2.Pmf(rand2List)
# print(rand2Pmf)

# Ploting random number PMF
thinkplot.Hist(rand1Pmf)
thinkplot.Show(xlabel='random number', ylabel='Frequency', title='Random number1 fig')

thinkplot.Hist(rand2Pmf)
thinkplot.Show(xlabel='random number', ylabel='Frequency', title='Random number2 fig')

