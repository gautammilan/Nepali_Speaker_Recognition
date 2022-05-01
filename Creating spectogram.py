#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


class getting_audio_dataset():
    
    def __init__(self,path_audio_dir,path_csv_file,second,sample_rate=22050,window_size=512):
        '''
        window_size: duration at which the short time fourier transform is calculated
        for speech recognition its default value is 502 at sampling rate 22050 which 
        corresponds to 23 miliseconds
        
        path_audio_dir:path to all the audio file
        path_csv_file: path to the csv_file
        sample_rate: rate at which the audio is going to be sample
        seconds: duration during which the spectrogram is going to be computed 
        if 3 seconds than return spectrogram of 3 second of audio
        '''
        
        self.sample_rate= sample_rate
        self.window= window_size
        self.second= second
        self.dir_audio= path_audio_dir
        self.dir_csv_file= path_csv_file
        self.image_path= []
        self.image_dir= None
  
        
        #Reading the padnas self.dataframe
        self.dataframe= pd.read_csv(self.dir_csv_file)
        self.info= {}

        #Droping the nan values from these rows
        self.dataframe= self.dataframe[self.dataframe['Source']=='GOV']
        self.dataframe.dropna(axis=0,subset= ['Name','Link'],inplace=True)
        
        #include the path of the audio in the self.dataframe
        self.dataframe['path']= [self.dir_audio+Name+'.MP3' for Name in self.dataframe['Name'].values]
        

        if self.dir_audio.split('\\')[-1]=='':
            self.sub_path= '\\'.join(self.dir_audio.split('\\')[:len(self.dir_audio.split('\\'))-2])

        else:
            self.sub_path= '\\'.join(self.dir_audio.split('\\')[:len(self.dir_audio.split('\\'))-1])


#Step1: Creating a features self.dataframe by sampling the audios       
    def reading_dataframe(self):   
        '''
        reading the saved dataframe from the disk
        '''
        print('Reading the dataframe from the disk...\n')
        dataframe= pd.read_csv(self.sub_path+r'\features_dataset.csv')
        dataframe.drop(axis=1,columns='Unnamed: 0',inplace=True)

        with open(self.sub_path+r'\features','rb') as f:
            features= pickle.load(f)

        dataframe['Features']= features
        return  dataframe
    
#Step2:creating an spectrogram of the array which contains amplitude    
    def spectrogram(self,segment,path):
        #Getting the spectrogram of the audio segment

        mel_spec= librosa.feature.melspectrogram(y=segment, sr=self.sample_rate, n_mels=64,n_fft= self.window)
        
        #Converting applitude to Decibal
        mel_spec_dB = librosa.power_to_db(mel_spec, ref=np.max)
        
        #Saving the spectrogram in the disk
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot()
        p = librosa.display.specshow(mel_spec_dB, ax=ax, y_axis='log', x_axis='time')
        image_path= self.sub_path+''
        fig.savefig(path)
        
        return 
    
    def save_spectrogram(self,dataframe,train=True):
        
        empty_dataframe=pd.DataFrame(columns=['id','audio1_path','audio2_path','label'])
        
        name= 'train' if train==True else 'test'
        print('Creating the spectrogram of ',name)
        if os.path.isdir(self.sub_path+r'\images')==False:
            os.mkdir(self.sub_path+r'\images')

        if self.image_dir==None:
            self.image_dir= self.sub_path+r'\images'

        if os.path.isdir(self.sub_path+'\\images\\'+name)==False:
            os.mkdir(self.sub_path+'\\images\\'+name)

        for i in tqdm(range(dataframe.shape[0])):
            audio1= dataframe['audio1'].values[i]
            audio2= dataframe['audio2'].values[i]
            audio_id= dataframe['id'].values[i]
            label= dataframe['label'].values[i]
            
            #Saving an image
            for j in range(2):
                #Getting the image path
                if j==0:
                    image_path1= self.image_dir+'\\'+name+'\\audio1_'+str(audio_id)
                    self.spectrogram(audio1,image_path1)
                else:
                    image_path2= self.image_dir+'\\'+name+'\\audio2_'+str(audio_id)
                    self.spectrogram(audio2,image_path2)
                
            empty_dataframe= empty_dataframe.append({'id':audio_id,'audio1_path':image_path1,'audio2_path':image_path2                                                        ,'label':label},ignore_index=True)

        return empty_dataframe
    
    def train_test_split(self):
        '''
        Splitting the audio in train(80%) and test(20%)
        '''
        
        dataframe= self.reading_dataframe()
        train_features_dataframe= pd.DataFrame(columns=['Name','Gender','Features','Duration'])
        test_features_dataframe= pd.DataFrame(columns=['Name','Gender','Features','Duration'])

        
        #Getting 20% of audio in test dataset
        
        print('Creating the train and test self.dataframe')
        for val in dataframe.values:
 
            
            train_features= val[2][:int(val[2].shape[0]*0.8)]
            train_duration= train_features.shape[0]//self.sample_rate
            
            test_features= val[2][int(val[2].shape[0]*0.8):]
            test_duration= test_features.shape[0]//self.sample_rate
            
            train_features_dataframe= train_features_dataframe.append({'Name':val[0],'Gender':val[1],'Features':train_features                                                            ,'Duration':train_duration},ignore_index=True)
            
            test_features_dataframe= test_features_dataframe.append({'Name':val[0],'Gender':val[1],'Features':test_features                                                            ,'Duration':test_duration},ignore_index=True)
       
        return train_features_dataframe,test_features_dataframe

    
    def similar_pairs(self,num_of_similar_pair,dataframe,train=True):
        '''
        Getting the similar pairs of audio segment
        '''
        if train==True:
            self.train= self.train.append(self.similar(dataframe,num_of_similar_pair,train),ignore_index=True)    
        else:
            self.test= self.test.append(self.similar(dataframe,num_of_similar_pair,train),ignore_index=True) 
        return
    
    
    def disimilar_pairs(self,num_of_disimilar_pair,dataframe,train=True):
        #Getting the disimilar pairs of audio segment
        if train==True:
            self.train= self.train.append(self.disimilar(dataframe,num_of_disimilar_pair,train),ignore_index=True)    
        else:
            self.test= self.test.append(self.disimilar(dataframe,num_of_disimilar_pair,train),ignore_index=True)  
        return

    
    def similar(self,dataframe,num_of_similar_pair,train=True):
        
        num_of_similar_pair= int(num_of_similar_pair*0.2) if train==False else num_of_similar_pair
 
        #Creating an empty self.dataframe
        empty_dataframe= pd.DataFrame(columns=['audio1','audio2','label'])
        for row in dataframe.values:

            #Creating hundred audio sample using audio of an individual speaker
            for i in range(num_of_similar_pair):    
                
                for j in range(2):
         
                    #Getting the starting index for that audio
                    start= np.random.randint(low=0,high= row[2].shape[0]-self.second*self.sample_rate)

                    #Getting an segment of audio
                    if j==0:
                        audio_segment1= (row[2][start:start+self.second*self.sample_rate])
                    else:
                        audio_segment2= (row[2][start:start+self.second*self.sample_rate])

                empty_dataframe= empty_dataframe.append({'audio1':audio_segment1                                               ,'audio2':audio_segment2,'label':1},ignore_index=True)

        
        return empty_dataframe
            
    
    
    def disimilar(self,dataframe,num_of_disimilar_pairs,train=True):
        
        num_of_disimilar_pairs= int(num_of_disimilar_pairs*0.2) if train==False else num_of_disimilar_pairs
        
        #Creating an empty self.dataframe
        empty_dataframe= pd.DataFrame(columns=['audio1','audio2','label'])
        
        for i in range(dataframe.shape[0]):
            other_audios_name= list(set(dataframe['Name'].values)                                    -set([dataframe['Name'].values[j] for j in range(dataframe['Name'].shape[0]) if j>i]))
           
            if len(other_audios_name)== dataframe.shape[0]:
                   return empty_dataframe
            
            for other_name in other_audios_name:
                    
                for j in range(num_of_disimilar_pairs): 
                    for k in range(2):   
                        
                        
                         #Getting an audio segment from one speaker
                        if k==0:
                            start= np.random.randint(low=0,high= dataframe['Features'].values[i].shape[0]-self.second*self.sample_rate)
                            audio_segment1= (dataframe['Features'].values[i][start:start+self.second*self.sample_rate])

                            
                        #Getting an audio segment from other speaker
                        else:
                            Features= dataframe[dataframe['Name']==other_name]['Features'].values[0]
      
                            start= np.random.randint(low=0,high=Features.shape[0]-self.second*self.sample_rate)
                            audio_segment2= (Features[start:start+self.second*self.sample_rate])
                    
                    #After getting the audio segment from two different spearker save it
                    empty_dataframe= empty_dataframe.append({'audio1':audio_segment1                                               ,'audio2':audio_segment2,'label':0},ignore_index=True)

 
    def creating_data(self,num_of_similar_pair,num_of_disimilar_pair,dataframe,use_previous=True,train=True):

        
        if train==True:
            
            #Creating the training dataset
            if os.path.isfile(self.sub_path+r'\train.csv')==False or use_previous== False:
                
                print('Creating the training dataset...')
                
                #Getting the similar pair
                self.similar_pairs(num_of_similar_pair,dataframe,train)
                
                #Getting the disimilar pair
                self.disimilar_pairs(num_of_disimilar_pair,dataframe,train)

        else:
            #Creating the testing dataset
            if os.path.isfile(self.sub_path+r'\test.csv')==False or use_previous== False:
                                   
                print('Creating the testing dataset...')
                self.similar_pairs(num_of_similar_pair,dataframe,train)
                self.disimilar_pairs(num_of_disimilar_pair,dataframe,train)         

        
        return
    
    
    def __call__(self,num_of_similar_pair,num_of_disimilar_pair,get_spectrogram=False,use_previous= True):
        
        start= datetime.now()
        self.train= pd.DataFrame(columns=['audio1','audio2','label'])
        self.test= pd.DataFrame(columns=['audio1','audio2','label'])

        '''
        num_of_similar_pair: 
        num_of_disimilar_pair:
        '''
        train_features_dataframe,test_features_dataframe= self.train_test_split()
        
        self.creating_data(num_of_similar_pair,num_of_disimilar_pair,train_features_dataframe,use_previous,train=True)      
        self.creating_data(num_of_similar_pair,num_of_disimilar_pair,test_features_dataframe,use_previous,train=False)
        
        self.train['id']= [i for i in range(self.train.shape[0])]
        self.test['id']=  [i for i in range(self.test.shape[0])]
        print('\nThe total time requires to get train test dataset is :',datetime.now()-start)
        
        if get_spectrogram==True:
            print('*******************************************************************\n')
            start= datetime.now()
            self.train= self.save_spectrogram(self.train,train=True)
            print('The spectrogram of train is saved')
            self.test= self.save_spectrogram(self.test,train=False)
            print('The spectrogram of test is saved')
            print('\nThe total time requires to save all the spectrogram is :',datetime.now()-start)
            
            self.train.to_csv(self.sub_path+r'\train.csv',index=False)
            self.test.to_csv(self.sub_path+r'\test.csv',index=False)
    
        return 
        
dataset=getting_audio_dataset(path+'Dataset\\',path+'data.csv',second=5)
dataset(num_of_similar_pair=1000,num_of_disimilar_pair=100,get_spectrogram=True,use_previous=False)

