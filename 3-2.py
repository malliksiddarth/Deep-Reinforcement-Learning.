#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input


# In[2]:


mnist= tf.keras.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()


# In[3]:


print (train_X.shape)
print (train_y.shape)
print (test_X.shape)
print (test_y.shape)


# In[4]:


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


# In[5]:


print (train_X.shape)
print (train_X[0])


# In[6]:


train_X = train_X/255.0
test_X = test_X/255.0


# In[20]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()


# In[21]:


history = model.fit(train_X, train_y, epochs=50,validation_split = 0.2, batch_size=64)


# In[22]:


test_loss, test_accuracy = model.evaluate(test_X, test_y)
print("Test accuracy: {}".format(test_accuracy))


# In[23]:


train_loss, train_accuracy = model.evaluate(train_X, train_y)
print("Train accuracy: {}".format(train_accuracy))


# In[24]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist['sparse_categorical_accuracy'])


# In[27]:


ixs = [2, 5, 9]  # Three convolution layers

#outputs = [model.layers[3].output for i in ixs]

model1 = model(inputs=model.inputs)

images = ['0.png', '8.png']

for image in images:

    img_arr = img_to_array(load_img(image, target_size=(224, 224)))
    np.reshape(img_arr,(28,28,3))
    
    img_exp_d = expand_dims(img_arr, axis=0)
    
    img_preprocess = preprocess_input(img_exp_d)
    
    feature_maps = model.predict(img_preprocess)

    for fmap in feature_maps:
        index = 0
        while index < 64:
            ax = pyplot.subplot(8, 8, index + 1)
            ax.set_xticks([])
            ax.set_yticks([])

            pyplot.imshow(fmap[0, :, :, index], cmap='gray')
            index += 1
        # show the figure
        pyplot.show()


# In[ ]:




