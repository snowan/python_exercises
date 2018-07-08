
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

data = open("/Users/xwan/Developer/python_exercises/datasets/student-mat.csv", "r")
famrel = list()
freetime = list()
goout = list()
dalc = list()
walc = list()
health = list()
absences = list()
scoreg1 = list()
# read file
for line in data:
  spLine = line.split(",")
  famrel.append(spLine[23])
  freetime.append(spLine[24])
  goout.append(spLine[25])
  dalc.append(spLine[26])
  walc.append(spLine[27])
  health.append(spLine[28])
  absences.append(spLine[29])
  scoreg1.append(spLine[30])



# remove first element 
famrel = famrel[1:]
freetime = freetime[1:]
goout = goout[1:]
dalc = dalc[1:]
walc = walc[1:]
health = health[1:]
absences = absences[1:]
scoreg1 = scoreg1[1:]

# famrel and G1 correlation
print ("famrel and G1 correlation: ", np.corrcoef(famrel, scoreg1))
# scatter plot
plt.scatter(scoreg1, famrel)
plt.title('Famrel VS G1')
plt.ylabel('famrel')
plt.xlabel('G1')
plt.show()

# freetime and G1 correlation
print ("freetime and G1 correlation: ", np.corrcoef(freetime, scoreg1))
plt.scatter(scoreg1, freetime)
plt.title('Freetime VS G1')
plt.ylabel('freetime')
plt.xlabel('G1')
plt.show()

# goout and G1 correlation
print ("goout and G1 correlation: ", np.corrcoef(goout, scoreg1))
plt.scatter(scoreg1, goout)
plt.title('goout VS G1')
plt.ylabel('goout')
plt.xlabel('G1')
plt.show()

# dalc and G1 correlation
print ("Dalc and G1 correlation: ", np.corrcoef(dalc, scoreg1))
plt.scatter(scoreg1, dalc)
plt.title('Dalc VS G1')
plt.ylabel('Dalc')
plt.xlabel('G1')
plt.show()

# walc and G1 correlation
print ("Walc and G1 correlation: ", np.corrcoef(walc, scoreg1))
plt.scatter(scoreg1, walc)
plt.title('Walc VS G1')
plt.ylabel('Walc')
plt.xlabel('G1')
plt.show()

# health and G1 correlation
print ("health and G1 correlation: ", np.corrcoef(health, scoreg1))
plt.scatter(scoreg1, health)
plt.title('health VS G1')
plt.xlabel('G1')
plt.ylabel('health')
plt.show()

# absences and G1 correlation
print ("absences and G1 correlation: ", np.corrcoef(absences, scoreg1))
plt.scatter(scoreg1, absences)
plt.title('absences VS G1')
plt.xlabel('G1')
plt.ylabel('absences')
plt.show()




#  famrel, freetime, goout, Dalc, Walc, health and absences

# list = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
#         'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
#         'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
#         'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3\n']
# print(list.index('famrel'))

