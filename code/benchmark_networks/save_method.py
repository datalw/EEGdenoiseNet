import numpy as np
from data_prepare import *
from Network_structure import *
from loss_function import *
import os  
from train_method import *

def save_eeg(saved_model, result_location, foldername, save_train, save_vali, save_test, 
            noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, train_num, denoise_network, datanum):

    if save_train == True:
        # generate every signal in training set
        Denoiseoutput_train, train_mse = test_step(saved_model, noiseEEG_train, EEG_train, denoise_network, datanum)    

        if not os.path.exists(result_location +'/'+  foldername + '/' +  train_num + '/' +'nn_output'):
            os.makedirs(result_location +'/'+  foldername + '/' +  train_num + '/'+ 'nn_output'   )
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' + '/' + 'noiseinput_train.npy', noiseEEG_train)
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' + '/' +  'Denoiseoutput_train.npy', Denoiseoutput_train)               #######################   change the adress!
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' + '/' +  'EEG_train.npy', EEG_train)

    if save_vali == True:
        # generate every signal in test set
        Denoiseoutput_val, val_mse = test_step(saved_model, noiseEEG_val, EEG_val, denoise_network, datanum)        
            
        if not os.path.exists(result_location +'/'+  foldername + '/' +  train_num + '/'+ 'nn_output'):
            os.makedirs(result_location +'/'+  foldername + '/' +  train_num + '/'+ 'nn_output')    
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' +'/' + 'noiseinput_val.npy', noiseEEG_val)
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' +'/' +  'Denoiseoutput_val.npy', Denoiseoutput_val)                      #######################   change the adress!
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' +'/' + 'EEG_val.npy', EEG_val)
        
    if save_test == True:
        # generate every signal in test set

        Denoiseoutput_test, test_mse = test_step(saved_model, noiseEEG_test, EEG_test, denoise_network, datanum)


        if not os.path.exists(result_location +'/'+  foldername + '/' +  train_num + '/'+ 'nn_output'):
            os.makedirs(result_location +'/'+  foldername + '/' +  train_num + '/' + 'nn_output')    
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' +'/' + 'noiseinput_test.npy', noiseEEG_test)
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' +'/' +  'Denoiseoutput_test.npy', Denoiseoutput_test)                      #######################   change the adress!
        np.save(result_location +'/'+  foldername + '/' + train_num + '/' + 'nn_output' +'/' + 'EEG_test.npy', EEG_test)