#%%
import tensorflow as tf
import numpy as np

from data_prepare import *
from Network_structure import *
from loss_function import *
from train_method import *
from save_method import *
import os
#sys.path.append('../')
from Novel_CNN import *

# EEGdenoiseNet V2
# Author: Haoming Zhang 
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
#####################################################自定义 user-defined ########################################################
#%%
epochs = 50    # training epoch
batch_size  = 40    # training batch size
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'Novel_CNN'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN 
noise_type = 'EMG'

result_location = r'/home/azureuser/cloudfiles/code/Users/lu.l.wang/EEGdenoiseNet/saved models/'     #  Where to export network results   ############ change it to your own location #########
foldername = '50_40_10_NCNN_EMG'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='0'
save_train = True
save_vali = True
save_test = True

################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(lr=0.00005, rho=0.9)
adam=tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(lr=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = rmsp

if noise_type == 'EOG':
  datanum = 512
elif noise_type == 'EMG':
  datanum = 1024

#%%
# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''

#%%
#################################################### 数据输入 Import data #####################################################

file_location = '/home/azureuser/cloudfiles/code/Users/lu.l.wang/EEGdenoiseNet/data/'                    ############ change it to your own location #########
if noise_type == 'EOG':
  EEG_all = np.load( file_location + 'EEG_all_epochs.npy')                              
  noise_all = np.load( file_location + 'EOG_all_epochs.npy') 
elif noise_type == 'EMG':
  EEG_all = np.load( file_location + 'EEG_all_epochs_512hz.npy')                              
  noise_all = np.load( file_location + 'EMG_all_epochs_512hz.npy') 

#%%
############################################################# Running #############################################################
#for i in range(10):
i = 1     # We run each NN for 10 times to increase  the  statistical  power  of  our  results
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = noise_type)

#%%
if denoise_network == 'fcNN':
  model = fcNN(datanum)

elif denoise_network == 'Simple_CNN':
  model = simple_CNN(datanum)

elif denoise_network == 'Complex_CNN':
  model = Complex_CNN(datanum)

elif denoise_network == 'RNN_lstm':
  model = RNN_lstm(datanum)

elif denoise_network == 'Novel_CNN':
  model = Novel_CNN(datanum)


else: 
  print('NN name arror')


saved_model, history = train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, 
                      epochs, batch_size,optimizer, denoise_network, 
                      result_location, foldername , train_num = str(i))

#%%
#denoised_test, test_mse = test_step(saved_model, noiseEEG_test, EEG_test)

#%%
# save signal
save_eeg(saved_model, result_location, foldername, save_train, save_vali, save_test, 
                    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, 
                    str(i), denoise_network, datanum)
if not os.path.exists(result_location +'/'+  foldername + '/' +  str(i) + '/' +'nn_output'):
  os.makedirs(result_location +'/'+  foldername + '/' +  str(i) + '/'+ 'nn_output'   )
np.save(result_location +'/'+ foldername + '/'+ str(i)  +'/'+ "nn_output" + '/'+ 'loss_history.npy', history)

#%%
#save model
if saved_model is not None:
  path = os.path.join(result_location, foldername, str(i), "denoise_model")
  tf.keras.models.save_model(saved_model, path)