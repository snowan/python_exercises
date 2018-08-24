
# coding: utf-8

# In[6]:


# Import all the required modules
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np


# ## Load MNIST dataset into train and test datasets

# In[81]:


# Load dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_test_val = y_test


# In[82]:


# check train and test datasets
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)
print(y_test_val.shape)


# ## Visualize MNIST digits from training dataset

# In[91]:


# visualising some training dataset
for i in range(6):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap = 'gray', interpolation = 'none')
    plt.title('Digit: {}'.format(y_train[i]))
    plt.tight_layout()


# ## Reshape dataset

# In[99]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x train shape:', x_train.shape)


# In[100]:


print(np.unique(y_train, return_counts=True))


# In[101]:


#set number of categories
category = 10


# In[102]:


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, category)
y_test = keras.utils.to_categorical(y_test, category)
y_train[0]


# ## Building a CNN model, convolitonal filter, maxpooling, dropout, flatten, dense, and second level dense and dropout, compile

# In[103]:


input_shape = (28, 28, 1)
# building model
model = Sequential()

#convolutional layer with rectified linear unit activation
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

#32 convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))

#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

#randomly turn neurons on and off to improve convergence
model.add(Dropout(0.25))

#flatten since too many dimensions, we only want a classification output
model.add(Flatten())

#fully connected to get all relevant data
model.add(Dense(128, activation='relu'))

#one more dropout
model.add(Dropout(0.5))

#output a softmax to squash the matrix into output probabilities
model.add(Dense(category, activation='softmax'))

#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# In[104]:


batch = 128
epoch = 5
#model training
model_log = model.fit(x_train, y_train,
          batch_size=batch,
          epochs=epoch,
          verbose=1,
          validation_data=(x_test, y_test))


# ## Evaluate model, plot model accuracy and model loss

# In[34]:


# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# #### Test model accuracy and model loss, we can see that model accuracy is 99%,  very good

# In[96]:


# plotting model accuracy and model loss

plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()


# ## Model summary

# In[93]:


model.summary()


# ## Predict model, show images

# In[67]:


# Predict images
preds = model.predict(x_test)
print(preds)


# In[68]:


preds = np.argmax(np.round(preds), axis=1)


# In[94]:


preds.shape, y_test_val.shape


# ## Plot correct prediction handwritten digits

# In[95]:


# print(preds)
# print(y_test)
correct = np.where(preds == y_test_val)[0]

print ('Found correct labels: ', len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Digit {}".format(preds[correct], y_test[correct]))
    plt.tight_layout()


# ## Plot incorrect prediction handwritten digits

# In[90]:


incorrect = np.where(preds != y_test_val)[0]

print ('Found correct labels: ', len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Digit {}".format(preds[incorrect], y_test[incorrect]))
    plt.tight_layout()

