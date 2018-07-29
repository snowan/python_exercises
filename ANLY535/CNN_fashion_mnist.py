
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import to_categorical


# In[6]:


(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.fashion_mnist.load_data()


# In[7]:


print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)


# In[8]:


# Find the unique numbers from the train labels
classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


# In[9]:


plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))


# In[10]:


#As a first step, convert each 28 x 28 image of the train and test set into a matrix of size 28 x 28 x 1 which is fed into the network.
train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape


# In[11]:


# The data right now is in an int8 format, so before you feed it into the network you need to convert its type to float32,
# and you also have to rescale the pixel values in range 0 - 1 inclusive. So let's do that!
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.


# In[16]:


# Now you need to convert the class labels into a one-hot encoding vector.
# let's convert the training and testing labels into one-hot encoding vectors:

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# In[17]:


from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)


# In[18]:


# For one last time let's check the shape of training and validation set.
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


# In[27]:


# Model the data
# First, let's import all the necessary modules required to train the model.
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# In[28]:


# You will use a batch size of 64 using a higher batch size of 128 or 256 is also preferable it all depends on 
# the memory. It contributes massively to determining the learning parameters and affects the prediction accuracy. 
# You will train the network for 20 epochs.

batch_size = 64
epochs = 20
num_classes = 10


# In[29]:


# Next, you'll add the max-pooling layer with MaxPooling2D() and so on. 
# The last layer is a Dense layer that has a softmax activation function with 10 units, 
# which is needed for this multi-class classification problem.
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[30]:


# Compile the Model
# After the model is created, you compile it using the Adam optimizer, one of the most popular optimization algorithms.
# You can read more about this optimizer here. Additionally, you specify the loss type which is categorical cross 
# entropy which is used for multi-class classification, you can also use binary cross-entropy as the loss function. 
# Lastly, you specify the metrics as accuracy which you want to analyze while the model is training.
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[31]:


# Let's visualize the layers that you created in the above step by using the summary function. 
# This will show some parameters (weights and biases) in each layer and also the total parameters in your model.

fashion_model.summary()


# In[32]:


# Train the Model
# It's finally time to train the model with Keras' fit() function! The model trains for 20 epochs. 
#The fit() function will return a history object; By storying the result of this function in fashion_train, 
#you can use it later to plot the accuracy and loss function plots between training and validation which will help you
# to analyze your model's performance visually.

fashion_train = fashion_model.fit(train_X, train_label, 
                                  batch_size=batch_size,
                                  epochs=epochs,verbose=1,
                                  validation_data=(valid_X, valid_label))


# In[33]:


# Model Evaluation on the Test Set
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[34]:


# However, you saw that the model looked like it was overfitting. Are these results really all that good?

# Let's put your model evaluation into perspective and plot the accuracy and loss plots 
# between training and validation data:

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[36]:


# Adding Dropout into the Network
# So let's create, compile and train the network again but this time with dropout. 
# And run it for 20 epochs with a batch size of 64.
batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))


# In[37]:


fashion_model.summary()


# In[38]:


fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[39]:


fashion_train_dropout = fashion_model.fit(train_X, 
                                          train_label, 
                                          batch_size=batch_size,
                                          epochs=epochs,verbose=1,
                                          validation_data=(valid_X, valid_label))


# In[40]:


# Let's save the model so that you can directly load it and not have to train it again for 20 epochs. 
# This way, you can load the model later on if you need it and modify the architecture; Alternatively, 
# you can start the training process on this saved model. It is always a good idea to save the model -and 
# even the model's weights!- because it saves you time. Note that you can also save the model after every 
# epoch so that, if some issue occurs that stops the training at an epoch, you will not have to start the 
# training from the beginning.

fashion_model.save("fashion_model_dropout.h5py")


# In[41]:


# Model Evaluation on the Test Set
# Finally, let's also evaluate your new model and see how it performs!
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)


# In[42]:


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[43]:


# Wow! Looks like adding Dropout in our model worked, even though the test accuracy did not improve significantly 
# but the test loss decreased compared to the previous results.

# Now, let's plot the accuracy and loss plots between training and validation data for the one last time.

accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[ ]:


# Finally, you can see that the validation loss and validation accuracy both are in sync with the training loss 
# and training accuracy. Even though the validation loss and accuracy line are not linear, but it shows that 
# your model is not overfitting: the validation loss is decreasing and not increasing, and there is not much gap 
# between training and validation accuracy.

# Therefore, you can say that your model's generalization capability became much better since the loss
# on both test set and validation set was only slightly more compared to the training loss.



# In[44]:


# Predict Labels
predicted_classes = fashion_model.predict(test_X)


# In[45]:


# Since the predictions you get are floating point values, it will not be feasible to compare the predicted labels with true 
# test labels. So, you will round off the output which will convert the float values into an integer. Further, 
# you will use np.argmax() to select the index number which has a higher value in a row.

# For example, let's assume a prediction for one test image to be 0 1 0 0 0 0 0 0 0 0, the output for this should be a class label 1.

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)


# In[46]:


predicted_classes.shape, test_Y.shape


# In[47]:


correct = np.where(predicted_classes==test_Y)[0]
print "Found %d correct labels" % len(correct)
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()


# In[48]:


incorrect = np.where(predicted_classes!=test_Y)[0]
print "Found %d incorrect labels" % len(incorrect)
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()


# In[ ]:


# By looking at a few images, you cannot be sure as to why your model is not able to classify the above images correctly,
# but it seems like a variety of the similar patterns present on multiple classes affect the performance of
# the classifier although CNN is a robust architecture. For example, images 5 and 6 both belong to different classes 
# but look kind of similar maybe a jacket or perhaps a long sleeve shirt.




# In[49]:


# Classification Report
# Classification report will help us in identifying the misclassified classes in more detail.
# You will be able to observe for which class the model performed bad out of the given ten classes.

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))


# In[ ]:


# You can see that the classifier is underperforming for class 6 regarding both precision and recall. 
# For class 0 and class 2, the classifier is lacking precision. Also, for class 4, 
# the classifier is slightly lacking both precision and recall.


