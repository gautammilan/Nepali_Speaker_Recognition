#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import ast
import re
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Input,Model
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.train import Checkpoint,CheckpointManager
import shutil
from  transformers import AdamWeightDecay,WarmUp,RobertaConfig
# from pyimagesearch import utils

get_ipython().system("mkdir './evaluation'")
get_ipython().system("mkdir './Checkpoints/'")


# In[ ]:


class creating_batch:
    
    def __init__(self,path_to_train_csv,path_to_test_csv,height,width,batch):
        '''Argument:
        batch:batch size
        path_to_train_csv:path to the train csv file which contains path of pair of image with their correspoing labels
        '''
        
        self.batch= batch
        self.path_to_train_csv= path_to_train_csv
        self.path_to_test_csv= path_to_test_csv
        self.height= height
        self.width= width
        
    def read(self,path,label):

        # 1)reading the content of the path
        content1= tf.io.read_file(path[0])
        content2= tf.io.read_file(path[1])
        
        # 2)decoding the content
        image1= tf.io.decode_png(content1,channels=3)
        image1= tf.image.convert_image_dtype(image1, tf.float32)
        

        image2= tf.io.decode_png(content2,channels=3)
        image2= tf.image.convert_image_dtype(image2, tf.float32)
        
        #Reshazing the image
        image1= tf.image.resize(image1,[self.height,self.width])
        image2= tf.image.resize(image2,[self.height,self.width])
        

        
        #3) Normalizing the image
#         image1= preprocessing.Normalization()(image1)
#         image2= preprocessing.Normalization()(image2)
        
        return (image1,image2),label

         
    def read_testing(self,image_path):

        # 1)reading the content of the path
        content1= tf.io.read_file(image_path[0])
        content2= tf.io.read_file(image_path[1])
        
        # 2)decoding the content
        image1= tf.io.decode_png(content1,channels=3)
        image1= tf.image.convert_image_dtype(image1, tf.float32)
        

        image2= tf.io.decode_png(content2,channels=3)
        image2= tf.image.convert_image_dtype(image2, tf.float32)
        
        #Reshazing the image
        image1= tf.image.resize(image1,[self.height,self.width])
        image2= tf.image.resize(image2,[self.height,self.width])
        

        
        #3) Normalizing the image
#         image1= preprocessing.Normalization()(image1)
#         image2= preprocessing.Normalization()(image2)
        
        return image1,image2





    def pipeline(self,paths,label,for_testing=False):

        if for_testing==False:
                
            dataset=tf.data.Dataset.from_tensor_slices(((paths),label))

            #reading the images.....................................................
            dataset= dataset.map(self.read,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
                       
            dataset=tf.data.Dataset.from_tensor_slices(paths)

            #reading the images.....................................................
            dataset= dataset.map(self.read_testing,num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
       #Creating the batch
        dataset= dataset.batch(batch_size=self.batch,drop_remainder=True)

       #All the batches will be stored in the cache after the first iteration
        dataset.cache()

       #shuffeling 
        dataset=dataset.shuffle(buffer_size=batch_size)

    #   # repeating the dataset 
    #     dataset = dataset.repeat()

        dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    
    
    def create_tensorflow_dataset(self):
        
        '''
        Create a tensorflow dataset for training and test
        '''

        #Reading the train test dataframe

        train= pd.read_csv(self.path_to_train_csv)
        
        #shuffle the train and test dataframe
        train= train.sample(frac=1).reset_index(drop=True)
        
        a= np.array([train['audio1_path'].values,train['audio2_path'].values])
        b =np.moveaxis(a,0,-1)
    
        #Creating the train dataset
        train_dataset= self.pipeline(b,train['label'].values.reshape(train.shape[0],1)) 
        
        test= pd.read_csv(self.path_to_test_csv)
        test= test.sample(frac=1).reset_index(drop=True)
        
        a= np.array([test['audio1_path'].values,test['audio2_path'].values])
        b =np.moveaxis(a,0,-1)
    
        #Creating the train dataset
        test_dataset= self.pipeline(b,test['label'].values.reshape(test.shape[0],1))
        return train_dataset,test_dataset
        

    def testing(self):
        '''Will use this method for evaluating the model'''

        test= pd.read_csv(self.path_to_test_csv)
        test= test.sample(frac=1).reset_index(drop=True)
        c= np.array([test['audio1_path'].values,test['audio2_path'].values])
        d= np.array(c)

        test_dataset= self.pipeline(d,'not_needed',for_testing=True)
    
        return test_dataset

        


# In[ ]:


def euclidean_distance(inp):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    x,y= inp
    sum_square = tf.math.reduce_sum(tf.math.square(x - y),axis=1,keepdims=True)
    print(sum_square)
    return tf.math.sqrt(tf.math.maximum(sum_square,1e-07))


# In[ ]:


class siamese_network:
    
    def __init__(self,height,width):
        '''
        Parameters:
        network_architecture: the architecture that will be used in the network
        '''
        self.width= width
        self.height= height
        
        #Using densenet architecuture pretrained on imagenet
        self.subnetwork= tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet'                                                   ,input_tensor=None,input_shape=(self.height,self.width,3), pooling=max)
    
    def cosine_similarity(self,v1,v2):
        return tf.math.reduce_sum(v1*v2/(tf.norm(v1,ord='euclidean')*tf.norm(v1,ord='euclidean')),axis= [-1]) #[batch]

    def create_siamese_network(self):

        #Inputs for two images
        image1= Input(shape=(self.height,self.width,3))
        image2= Input(shape=(self.height,self.width,3))        
        
        #the output should be in the shape (batch,feature)
        output1= Flatten()(self.subnetwork(image1))
        output2= Flatten()(self.subnetwork(image2))
        

     
        #Therefore taking the square distance we will be taking cosine_distance as our distance function
        output = layers.Lambda(euclidean_distance)([output1, output2])
        


      
#             #Finding the distance between the two
#             #Therefore taking the square distance
#             embedded_distance =layers.Subtract()([output1, output2])
#             embedded_distance= layers.Lambda(lambda x: abs(x))(embedded_distance)
#             output= layers.Dense(1, activation='sigmoid')(embedded_distance)

        model= Model(inputs=[image1,image2],outputs=output)
        return model
    
    '''
    Function used for plotting precision and recall
    '''
    def util_plotting(self,train_precision,train_recall,test_precision,test_recall,epoch):
        

        if self.contractive== False:
            train_evaluation= self.util_dataframe(train_precision,train_recall,epoch)
            test_evaluation= self.util_dataframe(test_precision,test_recall,epoch)
        
            #Plotting the precision and recall 
            fig, ax = plt.subplots(1,2,figsize=(14,6))
            plt.title('Evaluationn')
            ax[0].set_title('Precision')
            ax[0].plot(train_evaluation['threshold'],train_evaluation['precision'],'ro--')
#             ax[0].plot(test_evaluation['threshold'],test_evaluation['precision'],'bo--')
#             ax[0].legend(['train','test'])
            ax[0].set_xlabel('Threshold')
            ax[0].set_ylabel('precision')

            ax[1].set_title('Recall')
            ax[1].plot(train_evaluation['threshold'],train_evaluation['recall'],'ro--')
#             ax[1].plot(test_evaluation['threshold'],test_evaluation['recall'],'bo--')
#             ax[1].legend(['train','test'])
            ax[1].set_xlabel('Threshold')
            ax[1].set_ylabel('recall')
            plt.show()
            fig.savefig(evaluation_dir+'fig_'+str(epoch)+'.png')
            
        else:
            train_evaluation= self.util_dataframe(train_precision,train_recall,epoch)
            test_evaluation= self.util_dataframe(test_precision,test_recall,epoch,train=False)
        
            fig, ax = plt.subplots(1,2,figsize=(14,6))

            plt.title('Evaluation')
#             plt.set_title('epoch vs distance')
            ax[0].set_title('Train')
            
            ax[0].plot(train_evaluation['epoch'],train_evaluation['avg_distance_sim'],'ro--')
            ax[0].plot(train_evaluation['epoch'],train_evaluation['avg_distance_dissimilar'],'bo--')
           
            
            ax[0].legend(['similar_distance','disimilar_distance'])
            ax[0].set_xlabel('epoch')
            ax[0].set_ylabel('distance')
            ax[0].legend(['similar_distance','disimilar_distance'])

            
            ax[1].set_title('Test')
            ax[1].plot(test_evaluation['epoch'],test_evaluation['avg_distance_sim'],'ro--')
            ax[1].plot(test_evaluation['epoch'],test_evaluation['avg_distance_dissimilar'],'bo--')
            ax[1].legend(['similar_distance','disimilar_distance'])
            ax[1].set_xlabel('epoch')
            ax[1].set_ylabel('distance')
            plt.show()
            fig.savefig(evaluation_dir+'fig_'+str(epoch)+'.png')
        return

    
    #Create dataframe of the evaluation metric
    def util_dataframe(self,precision_value,recall_value,epoch,train=True):
        
        if self.contractive==False:
            threshold= list(precision_value.keys())
            dicti={'threshold':threshold,'precision':list(precision_value.values()),'recall':list(recall_value.values())}
            dataframe= pd.DataFrame(dicti)
            
        else:
            path= evaluation_dir+'train_evaluation_df.csv' if train==True else evaluation_dir+'test_evaluation_df.csv'
            dicti= {'epoch':epoch,'avg_distance_sim':precision_value,'avg_distance_dissimilar':recall_value}
            
            if os.path.isfile(path)== True:
                df= pd.read_csv(path)
                dataframe= df.append(dicti, ignore_index=True)
            else:
                dataframe= pd.DataFrame(dicti,index= [0])
            dataframe.to_csv(path,index= False)
        return dataframe
        
        
    def get_result_and_reset(self):
        
        #Creating a dictionary of precision and recall
        if self.contractive==False:
            
            Precision={'0':precision_0,'0.2':precision_0_2,'0.4':precision_0_4,'0.6':precision_0_6,'0.8':precision_0_8}
            Recall= {'0':recall_0,'0.2':recall_0_2,'0.4':recall_0_4,'0.6':recall_0_6,'0.8':recall_0_8}
    #         auc_value= auc.result.numpy()

            #Getting the value of preicion and recall at different threshold
            precision_values= {i:Precision[i].result().numpy() for i in list(Precision.keys())}
            recall_values= {i:Recall[i].result().numpy() for i in list(Recall.keys())}

            #reseting the metrics

            precision_0.reset_state()
            precision_0_2.reset_state()
            precision_0_4.reset_state()
            precision_0_6.reset_state()
            precision_0_8.reset_state()

            recall_0.reset_state()
            recall_0_2.reset_state()
            recall_0_4.reset_state()
            recall_0_6.reset_state()
            recall_0_8.reset_state()
            return precision_values,recall_values
        
        else:
            avg_sim= mean_of_sim.result().numpy()
            avg_disim= mean_of_disim.result().numpy()
            mean_of_sim.reset_state()
            mean_of_disim.reset_state()
            return avg_sim,avg_disim
              
    
    #Getting the gradient using tf.GradientTape and updating the wieghts of the model
    @tf.function
    def train_step(self,image,label):

        with tf.GradientTape() as tape:
            prediction = model(image, training=True)
            loss_value =contractive_loss(label,prediction)
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        avg_distance_sim,avg_distance_disim= contractive_evaluator(label,prediction)

        #updating the metrics with new labels
#         precision_0.update_state(label,prediction)
#         precision_0_2.update_state(label,prediction)
#         precision_0_4.update_state(label,prediction)
#         precision_0_6.update_state(label,prediction)
#         precision_0_8.update_state(label,prediction)

#         recall_0.update_state(label,prediction)
#         recall_0_2.update_state(label,prediction)
#         recall_0_4.update_state(label,prediction)
#         recall_0_6.update_state(label,prediction)
#         recall_0_8.update_state(label,prediction)

        return loss_value,prediction,avg_distance_sim,avg_distance_disim
    
    
    
    #Creating static graph in tensorflow
    @tf.function
    def test_step(self,image,label):
        
        prediction = model(image, training=False)
        loss_value =contractive_loss(label,prediction)
        avg_distance_sim,avg_distance_disim= contractive_evaluator(label,prediction)

        #For contractive loss the output is not the class label so we need a different evaluation metric

#         precision_0.update_state(label,prediction)
#         precision_0_2.update_state(label,prediction)
#         precision_0_4.update_state(label,prediction)
#         precision_0_6.update_state(label,prediction)
#         precision_0_8.update_state(label,prediction)

#         recall_0.update_state(label,prediction)
#         recall_0_2.update_state(label,prediction)
#         recall_0_4.update_state(label,prediction)
#         recall_0_6.update_state(label,prediction)
#         recall_0_8.update_state(label,prediction)
        return loss_value,prediction,avg_distance_sim,avg_distance_disim

        
    def train_siamese_network(self,epochs,contractive=False,load_previous_model= True):
        
        '''
        Augments:
        epochs= no of epoch to train the model
        train_dataset= tensorflow dataset of train
        test_dataset= tensorflow dataset of test
        dir_to_save_checkpoints= path to the directory where the checkpoints are going to be saved
        dir_to_save_evaluation= path to the directory where evaluation dataframe is going to be saved
        '''
        self.contractive= contractive
        self.epochs= epochs
        count= 0
        #Creating an checkpoint object and manager to keep track of the checkpoint
        Checkpt= Checkpoint(root= model)
        previous_epoch= 0
        
        if load_previous_model== True:
            
            manager= tf.train.CheckpointManager(Checkpt,new_checkpoint_dir,3, checkpoint_name='./tf_ckpts')
            #Loading the previous checkpoints so that to start the training from that epoch             
            Checkpt.restore(manager.latest_checkpoint)
                   
            if manager.latest_checkpoint:
                previous_epoch= int(manager.latest_checkpoint.split('\\')[-1].split('-')[-1])
                print('Loading the model parameters from previous epoch ie:',previous_epoch)
                print('\n')
        else:
            
            manager= tf.train.CheckpointManager(Checkpt,new_checkpoint_dir,3, checkpoint_name='./tf_ckpts')
            
            #Deleting the previous checkpoints directory
            shutil.rmtree('./Checkpoints')
            get_ipython().system("mkdir './Checkpoints'")

     
        
        #Training the model using custom training loop 
        for epoch in range(self.epochs):
                train_loss= 0
             # Iterate over the batches of the dataset
                for step,[image,label] in enumerate(train_dataset):
                    
                    #Calulating loss at each batch
                    loss_value,distance,avg_distance_sim,avg_distance_disim= self.train_step(image,tf.cast(label,dtype=tf.float32))

                    train_loss= train_loss+loss_value.numpy()
        
                    #Checking if label 1 in present on that batch 
                    if label[label==1].shape[0]!=0:
                        mean_of_sim.update_state(avg_distance_sim)
                    mean_of_disim.update_state(avg_distance_disim)
                    
                    if step%500==0 and step!=0:
                        print('Epoch: ',epoch+previous_epoch,'/',self.epochs+previous_epoch)
                        print(step,'/',train_steps,'====================== loss: ',train_loss/step)
                        print(epoch,' train_avg_sim:',mean_of_sim.result().numpy(),'train_avg_disim:',mean_of_disim.result().numpy())

                        
                        
                #Calculate the value of metric and resetting the metric
                if self.contractive==False:
                    train_precision,train_recall= self.get_result_and_reset()
                else: 
                    train_avg_sim,train_avg_disim= self.get_result_and_reset()
                    
                # Run a validation loop at the end of each epoch.
                test_loss= 0
                for step, [image,label] in enumerate(test_dataset):
                    loss_value,distance,avg_distance_sim,avg_distance_disim= self.test_step(image,label)
                    
                    test_loss= test_loss+loss_value
                    
                    #updating the metrices
                    if label[label==1].shape[0]!=0:
                        mean_of_sim.update_state(avg_distance_sim)
                    mean_of_disim.update_state(avg_distance_disim)
                    
                    
                    
             #Calculate the value of precision and recall for normal loss and distance for contractive loss
                if self.contractive==False:
                    test_precision,test_recall= self.get_result_and_reset()
                else: 
                    test_avg_sim,test_avg_disim= self.get_result_and_reset()


                #Saving the result of the matrics
                if self.contractive==False:
                    self.util_plotting(train_precision,train_recall,test_precision,test_recall,epoch)
                else:
                    self.util_plotting(train_avg_sim,train_avg_disim,test_avg_sim,test_avg_disim,epoch)

                print('The test loss at epoch :',epoch+previous_epoch,'is :',test_loss/step)
                print(' test_avg_sim:',test_avg_sim,'test_avg_disim:',test_avg_disim)

                #Saving the checkpoints
                manager.save()
                print('Model is saved')
                print('************************************************************\n')


# In[ ]:


'''Note:
    We are defining these parameters outside of the class of siamese network because if it is inside of the class than it will
    create variable every single time, hence cannot be used with @tf.function, so that is why they are outside of the 
    class'''

#Custom loss function
def weighted_binary_loss(y_true,y_pred):
        
    feq_pos= tf.cast(tf.math.count_nonzero(y_true)/batch,dtype=tf.float32) 
    if feq_pos< 1/batch:
        feq_pos= 1.0
    feq_neq= tf.cast(tf.reduce_sum(tf.where(tf.equal(y_true,0),[1],[0]))/batch,dtype=tf.float32)

    loss_value= -tf.math.reduce_mean(feq_neq*(tf.cast(y_true,dtype=tf.float32)*tf.math.log(y_pred)+feq_pos*(1-tf.cast(y_true,dtype=tf.float32))*tf.math.log(1-y_pred)))
    return loss_value


#Simple binary loss
loss= tf.keras.losses.BinaryCrossentropy()


#As the distance function is cosine distance the maximum distance between two vector will be 2, so we will select
# our margin as 2
margin= 3.0
def contractive_loss(label,distance):
    
    difference= margin-distance
    temp= tf.where(tf.less(difference,[0.0]),[0.0],difference)   
    loss= tf.math.reduce_mean(tf.cast(label,dtype=tf.float32)*tf.math.square(distance)+(1-tf.cast(label,dtype=tf.float32))*tf.math.square(temp))
    return loss


# In[ ]:


#Providing paths to respective variable
train_csv= './train.csv'
test_csv= './test.csv'
previous_checkpoints_dir= '../input/checkpoint-contractive-loss-margin-5/Checkpoints'
new_checkpoint_dir= './Checkpoints/'
evaluation_dir= './evaluation/'
height= 288
width= 432


#Getting the train and test steps
train= pd.read_csv(train_csv)
test= pd.read_csv(test_csv)

#Defining batch size
batch_size= 22

train_steps= train.shape[0]//batch_size
test_steps= test.shape[0]//batch_size


# In[ ]:


beta_1= 0.9 
beta_2= 0.999  
initial_lr= 1e-3
epochs= 30

num_of_steps= epochs*train_steps #Total number of training steps
num_of_warmup= int(num_of_steps*0.1)#The learning rate is very small here


#Scheduler and warmups
scheduler= tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr , decay_steps=num_of_steps, end_learning_rate=0, power=1.0,
        cycle=False, name=None)
warm= WarmUp(initial_lr,decay_schedule_fn= scheduler,warmup_steps= num_of_warmup)


optimizer= AdamWeightDecay(learning_rate= initial_lr, beta_1=beta_1,beta_2=beta_2, epsilon=1e-07,weight_decay_rate=0.01,name='Adam')

#Evaluator for contractive loss
def contractive_evaluator(label,distance):
    #Getting the average distance for similar images
    label= tf.cast(label,dtype= tf.float32)
    distance= tf.reshape(distance,shape= [distance.shape[0],1])
    avg_distance_sim= tf.reduce_mean(distance[label==1.0]) 
    
    #Getting the average distance for dissimilar images
    avg_distance_disim= tf.reduce_mean(distance[label==0.0])
    
    return avg_distance_sim,avg_distance_disim

mean_of_sim= tf.keras.metrics.Mean(name='mean', dtype=None)
mean_of_disim= tf.keras.metrics.Mean(name='mean', dtype=None)



#Getting precision and recall at different threshold
precision_0= tf.keras.metrics.Precision(0)
precision_0_2= tf.keras.metrics.Precision(0.2)
precision_0_4= tf.keras.metrics.Precision(0.4)
precision_0_6= tf.keras.metrics.Precision(0.6)
precision_0_8= tf.keras.metrics.Precision(0.8)

recall_0= tf.keras.metrics.Recall(0)
recall_0_2= tf.keras.metrics.Recall(0.2)
recall_0_4= tf.keras.metrics.Recall(0.4)
recall_0_6= tf.keras.metrics.Recall(0.6)
recall_0_8= tf.keras.metrics.Recall(0.8)


# In[ ]:


#Training the modelS
base_class= siamese_network(height,width)
model= base_class.create_siamese_network()
model.summary()


# In[ ]:


dataset= creating_batch(train_csv,test_csv,height,width,batch_size)
train_dataset,test_dataset= dataset.create_tensorflow_dataset()
base_class.train_siamese_network(epochs= epochs,contractive= True,load_previous_model= False)

