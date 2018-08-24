
# coding: utf-8

# In[1]:

import tensorflow as tf
import keras.


# In[19]:

# Preparing training data (inputs-outputs)
training_inputs = tf.placeholder(shape=[None, 3], dtype=tf.float32)
# Desired outpus for each input
training_outputs = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# In[20]:

# Preparing neural network parameters (weights and bias) using TensorFlow Variables
weights = tf.Variable(initial_value=[[.3], [.1], [.8]], dtype=tf.float32)  
bias = tf.Variable(initial_value=[[1]], dtype=tf.float32)  


# In[21]:

# Preparing inputs of the activation function
af_input = tf.matmul(training_inputs, weights) + bias


# In[22]:

# Activation function of the output layer neuron  
predictions = tf.nn.sigmoid(af_input)  


# In[23]:

# Measuring the prediction error of the network after being trained  
prediction_error = tf.reduce_sum(training_outputs - predictions)  


# In[24]:

# Minimizing the prediction error using gradient descent optimizer  
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(prediction_error)  


# In[25]:

# Creating a TensorFlow Session
sess = tf.Session()


# In[26]:

# Initializing the TensorFlow Variables (weights and bias)
sess.run(tf.global_variables_initializer())


# In[27]:

# Training data inputs
training_inputs_data = [[255, 0, 0],  
                        [248, 80, 68],  
                        [0, 0, 255],  
                        [67, 15, 210]]  


# In[28]:

# Training data desired outputs  
training_outputs_data = [[1],  
                         [1],  
                         [0],  
                         [0]]  


# In[29]:

# Training loop of the neural network  
for step in range(10000):  
    sess.run(fetches=[train_op], feed_dict={
                                   training_inputs: training_inputs_data,  
                                   training_outputs: training_outputs_data})  
  


# In[30]:

# Class scores of some testing data  
print("Expected Scores 1 : ", sess.run(fetches=predictions, feed_dict={training_inputs: [[248, 80, 68],                                                                   [0, 0, 255]]}))  

print("Expected Scores 2: ", sess.run(fetches=predictions, feed_dict={training_inputs: [[255, 100, 50],                                                                   [0, 0, 255]]}))  
 


# In[31]:

# Closing the TensorFlow Session to free resources  
sess.close()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[3]:

# Creating a NumPy array holding the input data  
numpy_inputs = [[5, 2, 13],  
                [7, 9, 0]] 


# In[ ]:




# In[6]:

# Converting the NumPy array to a TensorFlow Tensor 
# convert_to_tensor() doc: https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor  
training_inputs = tf.convert_to_tensor(value=numpy_inputs, dtype=tf.int8)  


# In[7]:

# Creating a TenforFlow Session
sess = tf.Session()


# In[8]:

# Running the sessoin for evaluating the previously created Tensor
print("Output is : ", sess.run(fetches=training_inputs))


# In[9]:

# Closing the TenforFlow Session
sess.close()


# In[ ]:




# In[ ]:




# In[37]:

# Create a placeholder with data type int8 and shape 2x3.  
training_inputs = tf.placeholder(dtype=tf.int8, shape=(2, 3))  

# Creating a TenforFlow Session
sess = tf.Session()

# Running the session for evaluating assigning a value to the placeholder  
print("Output is : ", sess.run(fetches=training_inputs,  
                               feed_dict={training_inputs: [[5, 2, 13],  
                                                   [7, 9, 0]]}))

print("Output is : ", sess.run(fetches=training_inputs,  
                               feed_dict={training_inputs: [[1, 2, 3],  
                                                            [4, 5, 6]]}))  
print("Output is : ", sess.run(fetches=training_inputs,  
                               feed_dict={training_inputs: [[12, 13, 14],  
                                                            [15, 16, 17]]}))


# In[36]:




# In[ ]:



