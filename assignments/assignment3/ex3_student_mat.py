
# coding: utf-8

import thinkstats2
import thinkplot

from scipy import stats
import numpy as np
import pandas as pd
import re

# famList = list()
# romanList = list()
meduList = list()
absenceList = list()
famrelList = list()
G3List = list()
data = open("/Users/xwan/developers/ML/python_exercises/assignments/assignment3/student-mat.csv", "r")
# read famsize and romantic data into list
for line in data:
  spLine = line.split(",")
  meduList.append(spLine[6])
  famrelList.append(spLine[23])
  absenceList.append(spLine[29])
  G3List.append(spLine[32])
G3List[:] = [k[:-1] for k in G3List]


# remove first element
meduList = meduList[1:]
famrelList = famrelList[1:]
absenceList = absenceList[1:]
G3List = G3List[1:]
# print('medu---', meduList)
# print('famrel---', famrelList)
# print('absence---', absenceList)
# print('g3 --- ', G3List)

meduList = thinkstats2.Hist(meduList, label='Mother educational level')
absenceList = thinkstats2.Hist(absenceList, label='Number of absence days ')
famrelList = thinkstats2.Hist(famrelList, label='Quality of family relationship')
G3List = thinkstats2.Hist(G3List, 'final grade')


# plot Mother's eductaional level histogram
thinkplot.Hist(meduList)
thinkplot.Show(xlabel='Value', ylabel='Frequency', title='Mother educational level Fig')
# plot absence histogram
thinkplot.Hist(absenceList)
thinkplot.Show(xlabel='Value', ylabel='Frequency', title='Absence Fig')
# plot quality of family relationship histogram
thinkplot.Hist(famrelList)
thinkplot.Show(xlabel='Value', ylabel='Frequency', title='Quality of Family Relationship Fig')
# plot final grade histogram
thinkplot.Hist(G3List)
thinkplot.Show(xlabel='Value', ylabel='Frequency', title='Final Grade Fig')
