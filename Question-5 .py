#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy 
from math import log as lg 
import math
from numpy.core.fromnumeric import shape as sh
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers 


# From the text book the given equations are:
#    A(t) = b+Wh(t−1) +Ux(t)--->    (10.8)
#    
#    
#    
#    H(t) = tanh(a(t))--->           (10.9)
#    
#    
#    O(t) = c + V h(t) --->          (10.10)
#    
#    
#    
#    Yˆ  (t)  = softmax(o )  --->    (10.11)

# In[6]:


#the given x values from the quetion
x = numpy.array([[1,0],[0.50,.25],[0,1]])
#given values for each 
b = numpy.array([-1,1])
c = numpy.array([0.5,-0.5])
w = numpy.array([[1,-1],[0,2]])
u = numpy.array([[-1,0],[1,-2]])
v = numpy.array([[-2,1],[-1,0]])
def softmax(x):
    f_x = numpy.exp(x) / numpy.sum(numpy.exp(x))
    return f_x
def tanh(t):
  return numpy.tanh(a(t))
def a(t):
  if t < 1:
    return numpy.array([0,0])
  return b + (w.dot(tanh(t-1))) + (u.dot(x[t-1]))
def o(t):
  return c + (v.dot(tanh(t)))
def y_not(t):
  return softmax(o(t))


# * Custom loss function

# In[7]:


def loss(y):
  return (y[1]-0.5)**2 - lg(y[0])


# In[8]:


model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,1)))
model.add(tf.keras.layers.SimpleRNN(2))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()


# (1) Write a program that computes the outputs and loss using the given sequence and parameter values. give y(t) i.e. y1(t) and y2(t) for t=1:3 and customized loss

# In[9]:


for k in range(1,4):
  y = y_not(k)
  print("Output=",y)
  print("Loss=",loss(y))
  print("Time T=",k)


# (2) Estimate the gradient of the loss function with respect to b1 and b2 using the central difference method 
# using e=0.00005

# In[10]:


def tanh(t,b):
  return numpy.tanh(a(t,b))
def a(t,b):
  if t < 1:
    return numpy.array([0,0])
  return b + (w.dot(tanh(t-1,b))) + (u.dot(x[t-1]))
def o(t,b):
  return c + (w.dot(tanh(t,b)))
def y_not(t,b):
  return softmax(o(t,b))


# In[11]:


B1=numpy.copy(b)
B1[0]=b[0]+2


# In[12]:


print(b)
print(B1)


# In[13]:


def Difference(z,n,s):
  B1 = numpy.copy(b)
  B1=B1.astype('float64')
  B1[n]=b[n]+z
  B2 = numpy.copy(b)
  B2=B2.astype('float64')
  B2[n] = b[n]-z
  Y1=y_not(s,B1)
  Y2=y_not(s,B2)
  l1 = loss(Y1)
  l2 = loss(Y2)
  return (l1-l2)/(2*z)
z=0.00005
print("Gradient for B1 at Time, T=3 is ",Difference(z,0,3))
print("Gradient for B2 at Time, T=3 is ",Difference(z,1,3))


# (3) Compute the gradient of the loss function with respect to b1 and b2 by unfolding the network through time. You need to show the intermediate results.

# In[16]:


def softmax(x):
    fcap_x = numpy.exp(x) / numpy.sum(numpy.exp(x))
    return fcap_x
def tanh(s,B):
  result = numpy.tanh(a(s,B))
  print("Time, T=",s,"\t h(t)=",result)
  return result
def a(s,B):
  if s < 1:
    return numpy.array([0,0])
  result = B + (w.dot(tanh(s-1,B))) + (u.dot(x[s-1]))
  print("Time, T=",s,"\t a(t)=",result)
  return result
def o(s,B):
  result = c + (v.dot(tanh(s,B)))
  print("Time, T=",s,"\t o(t)=",result)
  return result
def y_cap(s,B):
  result = softmax(o(tant,b))
  print("Time, T=",s,"\t y_not(t)=",result)
  return result


# In[17]:


print("Gradient for B1 at Time, T=3 is ",Difference(z,0,3))
print("Gradient for B2 at Time, T=3 is ",Difference(z,1,3))


# (4) Fixing other parameters, perform one step of gradient descent optimization on b1 and b2 using a learning rate of 0.002.

# In[19]:


def softmax(x):
    f_capx = numpy.exp(x) / numpy.sum(numpy.exp(x))
    return f_capx
def tanh(s,B):
  return numpy.tanh(a(s,b))
def a(s,B):
  if s < 1:
    return numpy.array([0,0])
  return b + (w.dot(tanh(s-1,b))) + (u.dot(x[s-1]))
def o(s,B):
  return c + (v.dot(tanh(s,b)))
def y_cap(s,B):
  return softmax(o(s,B))
ln_rate = 0.002
GB1 = Difference(z,0,3)
GB2 = Difference(z,1,3)
b=b.astype('float64')
b[0]=b[0]-(ln_rate * GB1)
b[1]=b[1]-(ln_rate * GB2)


# In[20]:


print(" The values for B are (before Gradient Decent)",b)
print("The values achived after Gradient Decent",b)


# (5) Use your program to compute the loss using the new values for b (with other parameters as given) for the original sequence.

# In[21]:


Y = y_cap(3,b)


# In[22]:


print(loss(y))


# In[ ]:




