# coding: utf-8

# ## ANN learning to distingush two kinds of photos
# ## ------------------------------------------------------
# ## Code started from a quick classifier by Ilija Vukotic
# ## ======================================================


import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical


# #### load the data

TIME_CUT_INDEX = 2
QE_INDEX = 0



# signal


filename = "/projectnb/snoplus/sphere_data/npy/feature_map_collections.sph_out_Te130_15k_1702211533.6000.7000.npy"
signal_data = np.load(filename)[TIME_CUT_INDEX][QE_INDEX]
print signal_data

print('signal (images, y, x): ', signal_data.shape)

signal_images=signal_data.shape[0]

for i in range(1):
  #continue
  ev = signal_data[i]
  input_shape=ev.shape
  print('image size:', input_shape)
  plt.imshow(ev)
  plt.colorbar()
  plt.show()



# background


filename = "/projectnb/snoplus/sphere_data/npy/feature_map_collections.sph_out_1el_2p53_MeV_15k.5000.6000.npy"
background_data = np.load(filename)[TIME_CUT_INDEX][QE_INDEX]

print('background (images, y, x): ', background_data.shape)

background_images=background_data.shape[0]

for i in range(1):
  #continue
  ev = background_data[i]
  input_shape=ev.shape
  print('image size:', input_shape)
  plt.imshow(ev)
  plt.colorbar()
  plt.show()




labels = np.array([1] * signal_images + [0] * background_images)
data=np.concatenate((signal_data, background_data))
data = data/20.0
print (data.shape)
print type(data.shape)
data = data.reshape(data.shape+(1,))


# ### split into training and test samples


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print('images:', trainX.shape, testX.shape)
print('labels:', trainY.shape, testY.shape)

#trainY = to_categorical(trainY, num_classes=2)
#testY = to_categorical(testY, num_classes=2)


# ### functions

# In[73]:


def createModel():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(100,50,1))) #h=10, w=20
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2))) #h = 5, w = 10
    
    model.add(Conv2D(48, (4, 4), activation='relu')) #h=5-2+1=4, w = 10-3+1=8
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2))) #h=2, w=4
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) #h=2, w=4
    model.add(MaxPooling2D(pool_size=(2, 2))) #h=1, w=2
    model.add(Dropout(0.25))
 #
    model.add(Conv2D(80, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    # #model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # #model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(80, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
     
    return model


# In[74]:


my_network=createModel()


# In[83]:


batch_size = 100
epochs = 50
my_network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
 
history = my_network.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(testX, testY))
 
my_network.evaluate(testX, testY)


# In[84]:


# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


# In[85]:


predY = my_network.predict_proba(testX)
#print
print('\npredY.shape = ',predY.shape)
print(predY[0:10])
print(testY[0:10])
auc = roc_auc_score(testY, predY)
plt.show()
print('\nauc:', auc)
#fpr, tpr, thr =roc_curve(np.argmax(testY, axis=1), np.argmax(predY, axis=1))
fpr, tpr, thr =roc_curve(testY, predY)
plt.plot(fpr, tpr, label = 'auc = ' + str(auc) )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
print('False positive rate:',fpr[1], '\nTrue positive rate:',tpr[1])



my_network.summary()


print(testY.shape)
testY = testY.reshape(500,1)
print(predY.shape)
print(testY.T)
A = np.hstack((testY, predY))
print(A[0:10,:])
sig_out = A[A[:,0]>0]
print(sig_out.shape)
print(sig_out[0:10,:])
bkg_out = A[A[:,0]==0]
print(bkg_out.shape)
print(bkg_out[0:10,:])
sig = sig_out[:,1]
bkg = bkg_out[:,1]
##sig_hist, bin_edges = np.histogram(sig_out[:,1],bins=10, range=(0,1))
##plt.hist(sig_hist,bins=10, range=(0,1), histtype='step',ls='solid', alpha = 1.0, lw=3, color= 'r')
plt.hist(sig, bins=20, range=(0,1), histtype='step',ls='solid', alpha = 1.0, lw=3, color= 'r')
plt.hist(bkg, bins=20, range=(0,1), histtype='step',ls='dashed', alpha = 1.0, lw=3, color= 'b')
Ntot_sig = len(sig)
Ntot_bkg = len(bkg)
print('Ntot_sig = ',Ntot_sig,'   Ntot_bkg = ',Ntot_bkg)
roc_c = np.zeros((20,2))
for i in range(0,20):
    n_sig = len(sig[np.where(sig>0.05*i)])
    print('cut = ',0.05*i, '   n_sig = ', n_sig, '   Signal efficiency = ', n_sig/Ntot_sig)

    n_bkg = len(bkg[np.where(bkg>0.05*i)])
    if(n_bkg == 0): 
        r_bkg = 1000
    else:
        r_bkg = Ntot_bkg/n_bkg
    print('cut = ',0.05*i, '   n_bkg = ', n_bkg, '   Background rejection = ', r_bkg)
    print('')
    roc_c[i][1] = n_sig/Ntot_sig
    roc_c[i][0] = n_bkg/Ntot_bkg
fig2 = plt.figure()
plt.scatter(roc_c[:,0],roc_c[:,1])
    #for i in range(0,100):  
#    roc_c[i][1] = n_sig/Ntot_sig
#    roc_c[i][0] = n_bkg/Ntot_bkg
#plt.hist2d(roc_c[:,0],roc_c[:,1])
