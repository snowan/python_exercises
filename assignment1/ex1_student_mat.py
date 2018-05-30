
# coding: utf-8

import thinkstats2
import thinkplot

famList = list()
romanList = list()
data = open("student-mat.csv", "r")
# read famsize and romantic data into list
for line in data:
  spLine = line.split(",")
  famList.append(spLine[4])
  romanList.append(spLine[22])
print(famList[0], romanList[0])
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
