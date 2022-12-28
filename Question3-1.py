#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd


# In[3]:


mnist= tf.keras.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()


# In[4]:


print (train_X.shape)
print (train_y.shape)
print (test_X.shape)
print (test_y.shape)


# In[5]:


nr = 4
nc = 4

n1 = nr*nc
images = train_X[:n1]
labels = train_y[:n1]

fig, axe = plt.subplots(nr, nc, figsize=(2*nc,2*nr))
for i in range(nr*nc):
    x = axe[i//nr, i%nc]
    x.imshow(images[i], cmap='gray')
    x.set_title('Labels: {}'.format(labels[i]))
plt.tight_layout()    
plt.show()
label_dict ={
0:'T-shirt',
1: 'jeans/Trouser',
2: 'Pullover',
3: 'Dress',
4: 'Coat',
5: 'Sandal/Heels',
6: 'Shirt',
7: 'Sneaker',
8: 'Bag',
9: 'Shoes'}


# In[6]:


print (train_X.shape)
print (train_X[0])


# In[7]:


train_X = train_X/255.0
test_X = test_X/255.0


# In[8]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dense(200, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()


# In[9]:


history = model.fit(train_X, train_y, epochs=100,validation_split = 0.2, batch_size=64)


# In[10]:


test_loss, test_accuracy = model.evaluate(test_X, test_y)
print("Test accuracy: {}".format(test_accuracy))


# In[11]:


train_loss, train_accuracy = model.evaluate(train_X, train_y)
print("Train accuracy: {}".format(train_accuracy))


# In[12]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist['sparse_categorical_accuracy'])


# In[34]:


filters, biases = model.layers[2].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 6, 1
for n in range(n_filters):
    f = filters[0:,:n]
    for z in range(3):
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[0:n], cmap='gray')
        ix += 1
plt.show()


# In[ ]:




