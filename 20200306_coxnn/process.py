from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Convolution2D
from keras.regularizers import l2,l1
from keras import optimizers, layers, regularizers
from keras.optimizers import SGD,Adam,RMSprop
from tensorflow.compat.v1 import InteractiveSession
import keras.backend as K

import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.preprocessing import StandardScaler
from scipy import stats
import tensorflow as tf
from keras.engine.topology import Layer
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from scipy import stats
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from numpy.random import seed
import nnet_survival
#calibration
import matplotlib.pyplot as plt
import matplotlib

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

#Data process1
import os
from functools import reduce
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


class PHOTOMICS():
    def __init__(self, omics, PH, clinical):
        self.omics = omics
        self.PH = PH
        self.clinical = clinical
    
    def start_sess(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    
    def architecture(self,n_intervals):
        #mrna_input
        input_1 = Input(shape = (122,122,1))
        mrna_conv_1   = Convolution2D(256, (3, 3), kernel_initializer='glorot_normal')(input_1)
        mrna_bn_1     = BatchNormalization()(mrna_conv_1)
        mrna_act_1    = Activation('relu')(mrna_bn_1)
        mrna_pool_1   = MaxPooling2D(pool_size = (2,2))(mrna_act_1)

        mrna_conv_2   = Convolution2D(256, (3, 3), kernel_initializer='glorot_normal')(mrna_pool_1)
        mrna_bn_2     = BatchNormalization()(mrna_conv_2)
        mrna_act_2    = Activation('relu')(mrna_bn_2)
        mrna_pool_2   = MaxPooling2D(pool_size = (2,2))(mrna_act_2)

        flat_1 = Flatten()(mrna_pool_2)

        #meth_input
        input_2 = Input(shape = (122,122,1))
        meth_conv_1   = Convolution2D(256, (3, 3), kernel_initializer='glorot_normal')(input_2)
        meth_bn_1     = BatchNormalization()(meth_conv_1)
        meth_act_1    = Activation('relu')(meth_bn_1)
        meth_pool_1   = MaxPooling2D(pool_size = (2,2))(meth_act_1)

        meth_conv_2   = Convolution2D(256, (3, 3), kernel_initializer='glorot_normal')(meth_pool_1)
        meth_bn_2     = BatchNormalization()(meth_conv_2)
        meth_act_2    = Activation('relu')(meth_bn_2)
        meth_pool_2   = MaxPooling2D(pool_size = (2,2))(meth_act_2)

        flat_2 = Flatten()(meth_pool_2)

        #mirna_input
        input_3 = Input(shape = (42,42,1))
        mirna_conv_1   = Convolution2D(256, (3, 3), kernel_initializer='glorot_normal')(input_3)
        mirna_bn_1     = BatchNormalization()(mirna_conv_1)
        mirna_act_1    = Activation('relu')(mirna_bn_1)
        mirna_pool_1   = MaxPooling2D(pool_size = (2,2))(mirna_act_1)

        mirna_conv_2   = Convolution2D(256, (3, 3), kernel_initializer='glorot_normal')(mirna_pool_1)
        mirna_bn_2     = BatchNormalization()(mirna_conv_2)
        mirna_act_2    = Activation('relu')(mirna_bn_2)
        mirna_pool_2   = MaxPooling2D(pool_size = (2,2))(mirna_act_2)

        flat_3 = Flatten()(mirna_pool_2)

        #clinical_input
        input_4 = Input(shape=(22, ), name='clinical')
        dense = Dense(1, activation='relu', kernel_initializer='glorot_normal')(input_4)
        #flat4 = Flatten()(dense)

        if self.omics == 'mrna':
            if self.clinical:
                concat = Concatenate()([flat_1, dense])
            else:
                concat = flat_1

            dense_1 = Dense(512, activation = 'relu',kernel_initializer='glorot_normal')(concat)
            dense_1_dropout = Dropout(0.5)(dense_1)
            dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
            dense_2_dropout = Dropout(0.1)(dense_2)     

            if self.PH:
                dense_3 = Dense(1, use_bias=0, kernel_initializer='zeros')(dense_2_dropout)
                output  = nnet_survival.PropHazards(n_intervals)(dense_3)
            else:
                output = Dense(n_intervals, activation='sigmoid', kernel_initializer='he_normal')(dense_2_dropout)

            if self.clinical:
                model = Model(inputs=[input_1, input_4], outputs=[output])
            else:
                model = Model(inputs=[input_1], outputs=[output])
        
        if self.omics == 'meth':
            if self.clinical:
                concat = Concatenate()([flat_2, dense])
            else:
                concat = flat_2

            dense_1 = Dense(512, activation = 'relu',kernel_initializer='glorot_normal')(concat)
            dense_1_dropout = Dropout(0.5)(dense_1)
            dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
            dense_2_dropout = Dropout(0.1)(dense_2)
            
            if self.PH:
                dense_3 = Dense(1, use_bias=0, kernel_initializer='zeros')(dense_2_dropout)
                output  = nnet_survival.PropHazards(n_intervals)(dense_3)
            else:
                output = Dense(n_intervals, activation='sigmoid', kernel_initializer='he_normal')(dense_2_dropout)
            
            if self.clinical:
                model = Model(inputs=[input_2, input_4], outputs=[output])
            else:
                model = Model(inputs=[input_2], outputs=[output])
        
        if self.omics == 'mirna':
            if self.clinical:
                concat = Concatenate()([flat_3, dense])
            else:
                concat = flat_3

            dense_1 = Dense(512, activation = 'relu',kernel_initializer='glorot_normal')(concat)
            dense_1_dropout = Dropout(0.5)(dense_1)
            dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
            dense_2_dropout = Dropout(0.1)(dense_2)
                 
            if self.PH:
                dense_3 = Dense(1, use_bias=0, kernel_initializer='zeros')(dense_2_dropout)
                output  = nnet_survival.PropHazards(n_intervals)(dense_3)
            else:
                output = Dense(n_intervals, activation='sigmoid', kernel_initializer='he_normal')(dense_2_dropout)

            if self.clinical:
                model = Model(inputs=[input_3,input_4], outputs=[output])
            else:
                model = Model(inputs=[input_3], outputs=[output])

        if self.omics == 'mrna_meth':
            if self.clinical:
                concat = Concatenate()([flat_1, flat_2, dense])
            else:
                concat = Concatenate()([flat_1,flat_2])

            dense_1 = Dense(512, activation = 'relu',kernel_initializer='glorot_normal')(concat)
            dense_1_dropout = Dropout(0.5)(dense_1)
            dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
            dense_2_dropout = Dropout(0.1)(dense_2)    
            
            if self.PH:
                dense_3 = Dense(1, use_bias=0, kernel_initializer='zeros')(dense_2_dropout)
                output  = nnet_survival.PropHazards(n_intervals)(dense_3)
            else:
                output = Dense(n_intervals, activation='sigmoid', kernel_initializer='he_normal')(dense_2_dropout)
            
            if self.clinical:
                model = Model(inputs=[input_1,input_2,input_4], outputs=[output])
            else:
                model = Model(inputs=[input_1, input_2], outputs=[output])

        if self.omics == 'mrna_mirna':
            if self.clinical:
                concat = Concatenate()([flat_1, flat_3, dense])
            else:
                concat = Concatenate()([flat_1,flat_3])

            dense_1 = Dense(512, activation = 'relu',kernel_initializer='glorot_normal')(concat)
            dense_1_dropout = Dropout(0.5)(dense_1)
            dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
            dense_2_dropout = Dropout(0.1)(dense_2)
            
            if self.PH:
                dense_3 = Dense(1, use_bias=0, kernel_initializer='zeros')(dense_2_dropout)
                output  = nnet_survival.PropHazards(n_intervals)(dense_3)
            else:
                output = Dense(n_intervals, activation='sigmoid', kernel_initializer='he_normal')(dense_2_dropout)

            if self.clinical:
                model = Model(inputs=[input_1,input_3,input_4], outputs=[output])
            else:
                model = Model(inputs=[input_1, input_3], outputs=[output])

        if self.omics == 'mrna_meth_mirna':
            if self.clinical:
                concat = Concatenate()([flat_1, flat_2, flat_3, dense])
            else:
                concat = Concatenate()([flat_1, flat_2, flat_3])

            dense_1 = Dense(512, activation = 'relu',kernel_initializer='glorot_normal')(concat)
            dense_1_dropout = Dropout(0.5)(dense_1)
            dense_2 = Dense(128, activation = 'relu',kernel_initializer='glorot_normal')(dense_1_dropout)
            dense_2_dropout = Dropout(0.1)(dense_2)
            
            if self.PH:
                dense_3 = Dense(1, use_bias=0, kernel_initializer='zeros')(dense_2_dropout)
                output  = nnet_survival.PropHazards(n_intervals)(dense_3)
            else:
                output = Dense(n_intervals, activation='sigmoid', kernel_initializer='he_normal')(dense_2)

            if self.clinical:
                model = Model(inputs=[input_1,input_2,input_3,input_4], outputs=[output])
            else:
                model = Model(inputs=[input_1,input_2,input_3], outputs=[output])

        return model

    # Reset Keras Session
    def reset_keras(self):
        print("Restarting Keras Session...")
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()
        try:
            del model
        except:
            pass
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        print('Done!')

    #Process Data
    #Select common patients based on the number and type of omics under observation
    #Choices of omics: [mrna, meth, mirna, mrna_meth, mrna_mirna, mrna_meth_mirna]

    def input_process1(self, path_omics1, path_omics2, path_omics3):
        print('Data processing-I...')
        #training_list = os.listdir('data/' + path)
        training_list1 = os.listdir('data/' + path_omics1)
        training_list2 = os.listdir('data/' + path_omics2)
        training_list3 = os.listdir('data/' + path_omics3)

        if self.omics=='mrna':      #for only mRNA data
            training_list=training_list1
            shape = (len(training_list), 122, 122, 1)
            shape_mirna = (len(training_list), 42, 42, 1)        

            dataset1 = np.ndarray(shape=shape,dtype=np.float32)
            dataset2 = np.ndarray(shape=shape,dtype=np.float32)
            dataset3 = np.ndarray(shape=shape_mirna,dtype=np.float32)
            i=0
            for item in training_list:
                img1 = load_img("data/" + path_omics1 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                # Convert to Numpy Array
                x1 = img_to_array(img1) 
                dataset1[i] = x1
                i += 1
                if i % 100 == 0:
                    print("%d images to array" % i)
            print("All mrna images done!")
        
        elif self.omics=='meth':      #for only Methylation data
            training_list=training_list2
            shape = (len(training_list), 122, 122, 1)
            shape_mirna = (len(training_list), 42, 42, 1)        

            dataset1 = np.ndarray(shape=shape,dtype=np.float32)
            dataset2 = np.ndarray(shape=shape,dtype=np.float32)
            dataset3 = np.ndarray(shape=shape_mirna,dtype=np.float32)
            i=0
            for item in training_list:
                img2 = load_img("data/" + path_omics2 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                # Convert to Numpy Array
                x2 = img_to_array(img2) 
                dataset2[i] = x2
                i += 1
                if i % 100 == 0:
                    print("%d images to array" % i)
            print("All meth images done!")  

        elif self.omics=='mirna':     #for only miRNA data
            training_list=training_list3
            shape = (len(training_list), 122, 122, 1)
            shape_mirna = (len(training_list), 42, 42, 1)        

            dataset1 = np.ndarray(shape=shape,dtype=np.float32)
            dataset2 = np.ndarray(shape=shape,dtype=np.float32)
            dataset3 = np.ndarray(shape=shape_mirna,dtype=np.float32)
            i=0
            for item in training_list:
                img3 = load_img("data/" + path_omics3 + '/' + item, target_size=(42,42), color_mode='grayscale')  # this is a PIL image
                # Convert to Numpy Array
                x3 = img_to_array(img3) 
                dataset3[i] = x3
                i += 1
                if i % 100 == 0:
                    print("%d images to array" % i)
            print("All mirna images done!")       

        elif self.omics=='mrna_meth':     #for mRNA and Methylation omics
            training_list=np.intersect1d(training_list1,training_list2)
            print('mRNA_meth common patients:', len(training_list))
            training_list.sort()
            training_list = np.asarray(training_list, dtype=object)

            shape = (len(training_list), 122, 122, 1)
            shape_mirna = (len(training_list), 42, 42, 1)        

            dataset1 = np.ndarray(shape=shape,dtype=np.float32)
            dataset2 = np.ndarray(shape=shape,dtype=np.float32)
            dataset3 = np.ndarray(shape=shape_mirna,dtype=np.float32)

            i=0
            for item in training_list:
                img1 = load_img("data/" + path_omics1 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                img2 = load_img("data/" + path_omics2 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                # Convert to Numpy Array
                x1 = img_to_array(img1) 
                x2 = img_to_array(img2)  
                dataset1[i] = x1
                dataset2[i] = x2
                i += 1
                if i % 100 == 0:
                    print("%d images to array" % i)
            print("All mrna_meth images done!")

        elif self.omics=='mrna_mirna':        #for mRNA and miRNA omics
            training_list=np.intersect1d(training_list1,training_list3)
            print('mrna_mirna common patients:', len(training_list))
            training_list.sort()
            training_list = np.asarray(training_list, dtype=object)

            shape = (len(training_list), 122, 122, 1)
            shape_mirna = (len(training_list), 42, 42, 1)        

            dataset1 = np.ndarray(shape=shape,dtype=np.float32)
            dataset2 = np.ndarray(shape=shape,dtype=np.float32)
            dataset3 = np.ndarray(shape=shape_mirna,dtype=np.float32)

            i=0
            for item in training_list:
                img1 = load_img("data/" + path_omics1 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                img3 = load_img("data/" + path_omics3 + '/' + item, target_size=(42,42), color_mode='grayscale')  # this is a PIL image
                # Convert to Numpy Array
                x1 = img_to_array(img1) 
                x3 = img_to_array(img3)  
                dataset1[i] = x1
                dataset3[i] = x3
                i += 1
                if i % 100 == 0:
                    print("%d images to array" % i)
            print("All mrna_mirna images done!")

        elif self.omics=='mrna_meth_mirna':       #for mRNA, Methylation and miRNA omics
            training_list = reduce(np.intersect1d, (training_list1, training_list3, training_list2))
            training_list.sort()    
            print('mrna_meth_mirna common patients:', len(training_list))
            training_list = np.asarray(training_list, dtype=object)
            #Reference: https://www.kaggle.com/lgmoneda/data-augmentation-regression

            shape = (len(training_list), 122, 122, 1)
            shape_mirna = (len(training_list), 42, 42, 1)        

            dataset1 = np.ndarray(shape=shape,dtype=np.float32)
            dataset2 = np.ndarray(shape=shape,dtype=np.float32)
            dataset3 = np.ndarray(shape=shape_mirna,dtype=np.float32)

            i = 0
            for item in training_list:
                img1 = load_img("data/" + path_omics1 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                img2 = load_img("data/" + path_omics2 + '/' + item, target_size=(122,122), color_mode='grayscale')  # this is a PIL image
                img3 = load_img("data/" + path_omics3 + '/' + item, target_size=(42,42), color_mode='grayscale')  # this is a PIL image
                # Convert to Numpy Array
                x1 = img_to_array(img1) 
                x2 = img_to_array(img2)  
                x3 = img_to_array(img3)
                #x = x.reshape((3, 120, 160))
                # Normalize
                #x = (x - 128.0) / 128.0
                dataset1[i] = x1
                dataset2[i] = x2
                dataset3[i] = x3
                i += 1
                if i % 100 == 0:
                    print("%d images to array" % i)
            print("All mrna_meth_mirna images done!!")

        return dataset1, dataset2, dataset3, training_list

    #Data process2
    def input_process2(self, training_list, clinical):
        print("Data processing-II...")
        sample, t, f, age = [], [], [], []

        for list in tqdm(training_list):
            for i in range(len(clinical)):
                if clinical.iloc[i]['sample'] + '.png' == str(list):
                    p_id = clinical.iloc[i]['sample']
                    time = clinical.iloc[i]['os_time']
                    status = clinical.iloc[i]['vital_status']
                    a = clinical.iloc[i]['age']

                    sample.append(p_id)
                    t.append(time)
                    f.append(status)
                    age.append(a)
                    continue
                else:
                    pass
        t  = np.asarray(t)
        f  = np.asarray(f)
        sample  = np.asarray(sample)
        age = np.asarray(age)

        br=np.arange(0.,365.*10,365./4)
        nl=len(br)-1
        y_t = nnet_survival.make_surv_array(t,f,br)
        ind = range(len(f))
        print('Done!')

        if self.omics=='mrna':
            rand_range=[1,2]
        if self.omics=='meth':
            rand_range=[3,4]
        if self.omics=='mirna':
            rand_range=[4,5]
        if self.omics=='mrna_meth':
            rand_range=[6,7]
        if self.omics=='mrna_mirna':
            rand_range=[8,9]
        if self.omics=='mrna_meth_mirna':
            rand_range=[10,11]

        return t, f, sample, age, br, nl, y_t, ind, rand_range

    def train_val_results(self, model, train_omics_data, test_omics_data, batch_size):
        # Inference: mrna
        if self.omics=='mrna':
            if self.clinical:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)
            else:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)

        # Inference: meth
        if self.omics=='meth':
            if self.clinical:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)
            else:  
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)

        # Inference: mirna
        if self.omics=='mirna':
            if self.clinical:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)
            else:  
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)

        # Inference: mrna+meth
        if self.omics=='mrna_meth':
            if self.clinical:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)
            else:  
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)

        # Inference: mrna+mirna
        if self.omics=='mrna_mirna':
            if self.clinical:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)
            else:  
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)

        # Inference: mrna+meth+mirna
        if self.omics=='mrna_meth_mirna':
            if self.clinical:
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)
            else:  
                pred_train = model.predict(train_omics_data, verbose=0, batch_size=batch_size)
                pred_val = model.predict(test_omics_data, verbose=0, batch_size=batch_size)     
        return pred_train, pred_val

    def surv_prob(self, pred, pred_val, breaks, year):
        prob = np.cumprod(pred[:,0:np.nonzero(breaks>365*year)[0][0]], axis=1)[:,-1]
        prob_val = np.cumprod(pred_val[:,0:np.nonzero(breaks>365*year)[0][0]], axis=1)[:,-1]
        median = np.median(prob)
        median_val = np.median(prob_val)
        #print("Training and validation median probabilities are:", median, median_val)

        return prob, prob_val, median, median_val
    
    def process3(self, clinical, train_list):
        print('\nProcessing clinical features')
        enc = OneHotEncoder(handle_unknown='ignore')
        one_hot_T = pd.DataFrame(enc.fit_transform(clinical[['pathology_T_stage']]).toarray())
        one_hot_N = pd.DataFrame(enc.fit_transform(clinical[['pathology_N_stage']]).toarray())
        one_hot_M = pd.DataFrame(enc.fit_transform(clinical[['pathology_M_stage']]).toarray())
        one_hot_G = pd.DataFrame(enc.fit_transform(clinical[['gender']]).toarray())

        clinical_feat = pd.concat([one_hot_T, one_hot_N, one_hot_M, one_hot_G, clinical['age']], axis=1)
        clinical_feat = clinical_feat.reset_index(drop=True)
        clinical_feat = clinical_feat.set_index([clinical['sample'].values])

        train_id = []
        for patient in tqdm(train_list):
            train_id.append(patient.split('.')[0])
        clinical_feat = clinical_feat.loc[train_id,:]
        print("Features Processed")

        return clinical_feat, train_id


    def metrices(self, T, surv_prob, F, y, year, train_val, median, breaks):
        brier_true = np.cumprod(y[:,0:np.nonzero(breaks>365*year)[0][0]], axis=1)[:,-1]
        conc = concordance_index(T, surv_prob, F)
        brier = brier_score_loss(brier_true, surv_prob)
        
        T1 = T[surv_prob >= median]
        T2 = T[surv_prob < median]
        E1 = F[surv_prob >= median]
        E2 = F[surv_prob < median]
        result = logrank_test(T1, T2, E1, E2)
        p = result.p_value

        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

        # fig, ax = plt.subplots(ncols=1, figsize=(8,8))
        # #plt.figure(figsize=(12,4))
        # #plt.subplot(1,2,1)
        # days_plot = 9*365

        # kmf = KaplanMeierFitter()
        # for i in range(2):
        #     if i==0:
        #         kmf.fit(T1, event_observed = E1)
        #     elif i==1:
        #         kmf.fit(T2, event_observed = E2)
        #     kmf.plot()  
        # N1='N='+ str(len(T1))
        # N2='N='+ str(len(T2))

        # ax.set_xticks(np.arange(0, days_plot, 365))
        # ax.set_yticks(np.arange(0, 1.125, 0.125))
        # ax.tick_params(axis='x', labelsize=12)
        # ax.tick_params(axis='y', labelsize=12)
        # ax.set_xlim([0, days_plot])
        # ax.set_ylim([0,1])
        # ax.text(50, 0.025, 'logrank p-value = ' +str('%.3g'%(p)), bbox=dict(facecolor='red', alpha=0.3), fontsize=10)

        # ax.set_xlabel('Follow-up time (days)', fontsize = 14)
        # ax.set_ylabel('Probability of survival', fontsize = 14)
        # ax.legend(['Low Risk Individuals ' + N1 ,'High Risk Individuals ' + N2 ])
        # ax.set_title('%s set Kaplan-Meier Curves'%(train_val), fontweight = 'bold', fontsize = 14)
        # ax.grid()  
        # plt.show()

        print("%s year %s concordance index for %s:"%(str(year), train_val, str(self.omics)), conc)
        print("%s year %s brier score for %s:"%(str(year), train_val, str(self.omics)), brier)
        print("P-value:", p)
        return conc, brier, p

    def ipcw(self, F_train, F_test, T_train, T_test, survival_prob_valid):
        struct_train = np.zeros(len(F_train), dtype={'names':('F_train', 'T_train'),'formats':('?','i4')})
        struct_test = np.zeros(len(F_test), dtype={'names':('F_test', 'T_test'),'formats':('?','i4')})
        struct_train['F_train'] = F_train.astype('bool')
        struct_train['T_train'] = T_train
        struct_test['F_test'] = F_test.astype('bool')
        struct_test['T_test'] = T_test

        c_ipcw = '%.5g'%(1-concordance_index_ipcw(struct_train, struct_test, survival_prob_valid)[0])
        return c_ipcw