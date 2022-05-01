#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import wave
import librosa
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import wave
import audioread
from pydub import AudioSegment
from datetime import datetime
import pickle
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import ast
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Input,Model
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.train import Checkpoint,CheckpointManager
path= 'D:\\Work\\Speaker classification\\'


# In[40]:


class reading_audio_and_removing_silence():
    
    def __init__(self,path_audio_dir,path_csv_file,sample_rate=22050):
    '''
    path_audio_dir= path to the directory where audios are stored
    path_csv_file= path to the csv file which contains three columns "Speaker", "path to the audio" and it's "duratoin"
    sample_rate= the sampling rate at which we want to sample the audio
    '''
        self.dir_audio= path_audio_dir
        self.dir_csv_file= path_csv_file
        self.sub_path= None
        self.sample_rate= sample_rate
        
        #Reading the padnas self.dataframe
        self.dataframe= pd.read_csv(self.dir_csv_file)
        self.info= {}

        #Droping the nan values from these rows
        self.dataframe= self.dataframe[self.dataframe['Source']=='GOV']
        self.dataframe.dropna(axis=0,subset= ['Name','Link'],inplace=True)
        
        #include the path of the audio in the self.dataframe
        self.dataframe['path']= [self.dir_audio+Name+'.MP3' for Name in self.dataframe['Name'].values]


    def creating_a_dataframe(self,min_silence):   
        dataset= pd.DataFrame(columns=['Name','Gender','Features','Duration'])
        all_feature= []
        
        if self.sub_path==None:
            
            if self.dir_audio.split('\\')[-1]=='':
                self.sub_path= '\\'.join(self.dir_audio.split('\\')[:len(self.dir_audio.split('\\'))-2])

            else:
                self.sub_path= '\\'.join(self.dir_audio.split('\\')[:len(self.dir_audio.split('\\'))-1])

        #Getting the self.dataframe
        print('Creating an self.dataframe and removing the silence.....\n')

        for name in tqdm(self.dataframe['Name'].values):
            feature,duration= self.reading_a_file(name)

            #removing silence from the audio
            feature= self.remove_silence(feature,min_silence)

            self.info['Features']= feature
            self.info['Duration']= duration
            all_feature.append(feature)
            self.info['Gender']= self.dataframe[self.dataframe['Name']==name]['Sex'].values[0]
            self.info['Name']= name

            dataset= dataset.append(self.info,ignore_index=True)

        #saving the self.dataframe to the disk
        dataset.to_csv(self.sub_path+r'\features_dataset.csv')

        with open(self.sub_path+r'\features','wb') as f:
            pickle.dump(all_feature,f)

        return  
    
    
    def getduration(self,temp):
        #converting time 10:30 format to total number of seconds
        minute,second= temp.split(':')
        duration= int(minute)*60+int(second)
        return duration

    
    
    def reading_a_file(self,name):
        
        path= self.dataframe[self.dataframe['Name']==name]['path'].values[0]
        arr_audio,_=librosa.load(path,sr= self.sample_rate)
        
        if len(arr_audio.shape)>1: #dual channel audio
            #arr_audio.shape= (2,n) where 2 represent dual channel value
            arr_audio= arr_audio.sum(0)/2
            

        #Removing noises from the audio
        time_stamp= self.dataframe[self.dataframe['Name']==name]['Time Stamp'].values[0]
        
        if isinstance(time_stamp,str):

            if time_stamp=='Remove':
                #Removing first and last 1 minutes of audio
                arr= arr_audio[self.sample_rate*60:arr_audio.shape[0]-self.sample_rate*60]
                duration= arr//self.sample_rate
                
            elif len(time_stamp.split('-'))>1:
                start,end= time_stamp.split('-')

                #converting 10:40 time to seconds
                start_duration= self.getduration(start)
                end_duration= self.getduration(end)
                arr= arr_audio[self.sample_rate*start_duration:self.sample_rate*end_duration]
                duration= end_duration-start_duration

            else: 
                arr= arr_audio
                duration= arr//self.sample_rate

        return arr,duration

    
    def remove_silence(self,array_feature,min_silence):
        '''
        array_feature:array containing amplitude of an audio
        min_silence:threshould below which the silence will be removed
        '''
        
        #Step1:Create an audio_segment
        feature= self.float_to_int(array_feature)
        audio = AudioSegment(feature.tobytes(),frame_rate = self.sample_rate,sample_width = feature.dtype.itemsize                             ,channels = 1)
        
        #Step2:removing the silence from the audio segment
        audio_chunks = split_on_silence(audio,min_silence_len=min_silence,silence_thresh= -30,keep_silence= 100)

        #Step3:converting it back to the array
        feature= sum(audio_chunks)
        feature= np.array(feature.get_array_of_samples())
        feature= self.int_to_float(feature)
        return feature
        
        
#Citation:https://github.com/huseinzol05/malaya-speech/blob/master/malaya_speech/utils/astype.py
    def float_to_int(array, type=np.int16):

        if array.dtype == type:
            return array

        if array.dtype not in [np.int16, np.int32, np.int64]:
            if np.max(np.abs(array)) == 0:
                array[:] = 0
                array = type(array * np.iinfo(type).max)
            else:
                array = type(array / np.max(np.abs(array)) * np.iinfo(type).max)
        return array


    def int_to_float(array, type=np.float32):

        if array.dtype == type:
            return array

        if array.dtype not in [np.float16, np.float32, np.float64]:
            if np.max(np.abs(array)) == 0:
                array = array.astype(np.float32)
                array[:] = 0
            else:
                array = array.astype(np.float32) / np.max(np.abs(array))

        return array
    
    
    def __call__(self,min_silence):
        '''
        min_silence= if the duration of silence exceed this value than it is removed from the audio
        
        Output:
        The feature vector of every single audio contains its amplitute value
        '''
        start= datetime.now()
        self.creating_a_dataframe(min_silence)
        print('The time requires to read an audio as array and remove the silence is:',datetime.now()-start)


# In[41]:


reading_audio= reading_audio_and_removing_silence('D:\\Work\\Speaker classification\\Audio data\\AUDIO\\' ,'D:\\Work\\Speaker classification\\data.csv')
reading_audio(min_silence=400)


# In[42]:





# In[ ]:




