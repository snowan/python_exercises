import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = open("/Users/xwan/developer/python_exercises/datasets/student-mat.csv", "r")

age = list()
health = list()
freetime = list()
scoreG1 = list()
# read famsize and romantic data into list
for line in data:
  spLine = line.split(",")
  age.append(spLine[2])
  health.append(spLine[28])
  freetime.append(spLine[24])
  scoreG1.append(spLine[30])
# remove first element 
age = age[1:]
health = health[1:]
freetime = freetime[1:]
scoreG1 = scoreG1[1:]

# print(health)

# scatter plot age and G1
plt.scatter(age, scoreG1, c='r', label='data')
plt.plot(age, scoreG1)
plt.title('Age VS G1 Scatter plot')
plt.ylabel('Age')
plt.xlabel('G1')
# plt.show()

# scatter plot freetime and G1
plt.scatter(freetime,scoreG1)
plt.title('Freetime VS G1 Scatter plot')
plt.ylabel('Freetime')
plt.xlabel('G1')
plt.legend()
# plt.show()

# scatter plot health and G1
plt.scatter(health,scoreG1)
plt.title('Health VS G1 Scatter plot')
plt.ylabel('Health')
plt.xlabel('G1')
plt.legend()
# plt.show()


x = [1,5,7,323,7,45,3,0]
y = [92,3,5,2,1,65,7,34]

plt.scatter(x, y)
# plt.show()


student_data = pd.read_csv('/Users/xwan/developer/python_exercises/datasets/student-mat.csv')
st_age = student_data['age']
st_freetime = student_data['freetime']
st_health = student_data['health']
score_G1_age = student_data['G1']
score_G1_health = student_data['G1']
score_G1_freetime = student_data['G1']
score_G1_health1 = student_data['G1']

plt.scatter(st_health, score_G1_health1)
plt.xlabel('Health')
plt.ylabel('Score G1')
plt.title('Scatter plot Health VS G1')
plt.show()

plt.scatter(st_health, score_G1_health)
plt.xlabel('Health')
plt.ylabel('Score G1')
plt.title('Scatter plot Health VS G1')
plt.show()

plt.scatter(st_age, score_G1_age)
plt.xlabel('Age')
plt.ylabel('Score G1')
plt.title('Scatter plot Age VS G1')
plt.show()


plt.scatter(st_freetime, score_G1_freetime)
plt.xlabel('Freetime')
plt.ylabel('Score G1')
plt.title('Scatter plot Freetime VS G1')
plt.show()
