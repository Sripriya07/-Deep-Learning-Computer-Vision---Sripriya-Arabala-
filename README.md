# -Deep-Learning-Computer-Vision---Sripriya-Arabala-
# Image Brightness Detection

Problem Statement : To detect brightness of an image on a scale of 0-10.
Our Aim is to detect the brightness of a given image and the ouput should be its brightness value on a scale of 0-10.This can be solved as a classification problem where labels are from 0 to 10.

# Loading Data:
Data can be downloaded from the following link http://download.tensorflow.org/example_images/flower_photos.tgz.
Here,Flowers Dataset is used where it consists of Images of five different categories namely,daisy,dandelion,roses,sunflowers,tulips
For convenience in training,I took flowers of two categories daisy and dandelion which are total 1531 images and used for training.
Data can be loaded as follows:
            
             !wget http://download.tensorflow.org/example_images/flower_photos.tgz
   
This is a .tgz file and it can be extracted as follows:
 
             !tar -xzf flower_photos.tgz.1 
        
Now,also load test data containing 210 images which can be downloaded here https://drive.google.com/file/d/1LCAFZm675eqbQH_j23u9Gmurg0XHNFoP/view?usp=sharing. 
Unzip the file after loading

             !unzip flower_images.zip

# Importing all necessary libraries

             import cv2                         #image and video processing library
             import numpy as np                 #mathematical library used for numerical computation
             import pandas as pd                #for data manipulation and analysis

             import matplotlib.pyplot as plt    #used for plotting images,graphs,charts etc
             %matplotlib inline

             import os                          #python package for accessing computer and file system
             import random                      #to create random numbers for shuffled data
             import gc                          #garbage collector to clean deleted data from memory
 

Here,we create a file path to our train data of category daisy and category dandelion repsectively and create two variables train_cat1,train_cat2 for one for each category.

Using the command os.listdir() to get all the images in train dataset.
             
             train_dir='/content/flower_photos/daisy'
             train_cat1=['/content/flower_photos/daisy/{}'.format(i) for i in os.listdir(train_dir)]

Now,the data is kept in variable train_imgs.

             train_imgs=train_cat1[:]+train_cat2[:]
             
Also,create a file path for test data and create a variable to load test_imgs.             
            
             test_dir='/content/flower_images'
             test_imgs=['/content/flower_images/{}'.format(i) for i in os.listdir(test_dir)]
randomly shuffling data
              
              random.shuffle(train_imgs)
              
importing plotting module from matplotlib and plotting first three images              
              
              import matplotlib.image as mpimg   
              for img in train_imgs[0:3]:
                  img=mpimg.imread(img)
                  imgplot=plt.imshow(img)
                  plt.show()
Declaring image dimensions 
              
              nrows=150
              ncolumns=150
              channels=3     

defining function which read and process image to an acceptable format for our model

         def read_and_process_image1(list_of_images):
               X=[]  #Images
               y=[]  #labels
               for image in list_of_images:
                   X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(nrows,ncolumns),interpolation=cv2.INTER_CUBIC))
                   #reading the images

              
               greyscale_image = Image.open(image).convert('L')
               histogram = greyscale_image.histogram()
               pixels = sum(histogram)
               brightness = scale = len(histogram)

               for index in range(0, scale):
                    ratio = histogram[index] / pixels
                    brightness += ratio * (-scale + index)
                
               if brightness==255:
                    y.append(1)
               else:
                    y.append(brightness/scale)
                
        
        return X,y
   
 Now,training images are passed into the function and X and y are obtained.
 
               X,y=read_and_process_image1(train_imgs)
               
We can also check the image X[0] and also y values as shown in code.
Here,the values of y are a list of values ranging from 0 to 1 ,as we want our ouput to be in a scale of 0-10 we rescale it by multiplying with 10 and rounding it off.
  
               y_new = [ '%.2f' % elem for elem in y ]
               y_new=[i*10 for i in y]
               y_new=[round(i) for i in y_new]
 plotting first 5 images 
 
               plt.figure(figsize=(20,10))
               columns=5
               for i in range(columns):
                   plt.subplot(5/columns+1,columns,i+1)
                   X[i]= cv2.cvtColor(X[i], cv2.COLOR_RGB2BGR)
                   plt.imshow(X[i])
 output:

   ![5images](https://user-images.githubusercontent.com/49706927/63223463-7c595400-c1d3-11e9-949c-8d09fb9e3a0b.png)

plotting X and y                    
               
               import seaborn as sns       #plotting package
               X=np.array(X)               #X and y are of type list and now convert to numpy array
               y_new=np.array(y_new)
               sns.countplot(y_new)
               plt.title('Labels')
               
  output:
         
![bargraph](https://user-images.githubusercontent.com/49706927/63223467-998e2280-c1d3-11e9-886e-470f9a13d681.png)
               
printing shapes of X and y

               print("shape of train images is :",X.shape)
               print("Shape of labels is :",y_new.shape)
              
   output:
              
              shape of train images is : (1531, 150, 150, 3)
              Shape of labels is : (1531,)   
     
Now,Using SKlearn,splitting the data into training data and testing data where by default it splits the data in the ratio of 80:20 of training and testing data.

               from sklearn.model_selection import train_test_split
               X_train,X_val,y_train,y_val=train_test_split(X,y_new,test_size=0.2,random_state=2)

               print("shape of train images is :",X_train.shape)
               print("shape of validation images is :",X_val.shape)
               print("shape of labels is :",y_train.shape)
               print("shape of labels is :",y_val.shape)
               
  output:
              
              shape of train images is : (1224, 150, 150, 3)
              shape of validation images is : (307, 150, 150, 3)
              shape of labels is : (1224,)
              shape of labels is : (307,)

getting the length of train and validation data
           
              ntrain=len(X_train)
              nval=len(X_val)
              batch_size=32
              
# Building the Model             
            
# importing necessary keras modules
            from keras import layers
            from keras import models
            from keras import optimizers
            from keras.preprocessing.image import ImageDataGenerator
            from keras.preprocessing.image import img_to_array,load_img

Using VGGnet as the network Architecture and building the convolutional layers.
             
             model=models.Sequential()
             model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
             model.add(layers.MaxPooling2D((2,2)))
             model.add(layers.Conv2D(64,(3,3),activation='relu'))
             model.add(layers.MaxPooling2D(2,2))
             model.add(layers.Conv2D(128,(3,3),activation='relu'))
             model.add(layers.MaxPooling2D(2,2))
             model.add(layers.Conv2D(128,(3,3),activation='relu'))
             model.add(layers.MaxPooling2D(2,2))
             model.add(layers.Flatten())
             model.add(layers.Dropout(0.5))
             model.add(layers.Dense(512,activation='relu'))
             model.add(layers.Dense(10,activation='softmax'))
 Here,activation function softmax is used as we have muti-labels of brightness range ranging from 0-10 for the images.
 Model summary can be printed as below:
 
            model.summary() #printing model summary
            
   output:
             
            _________________________________________________________________
             Layer (type)                 Output Shape              Param #   
            =================================================================
            conv2d_13 (Conv2D)           (None, 148, 148, 32)      896       
            _________________________________________________________________
            max_pooling2d_13 (MaxPooling (None, 74, 74, 32)        0         
            _________________________________________________________________
            conv2d_14 (Conv2D)           (None, 72, 72, 64)        18496     
            _________________________________________________________________
            max_pooling2d_14 (MaxPooling (None, 36, 36, 64)        0         
            _________________________________________________________________
            conv2d_15 (Conv2D)           (None, 34, 34, 128)       73856     
            _________________________________________________________________
            max_pooling2d_15 (MaxPooling (None, 17, 17, 128)       0         
            _________________________________________________________________
            conv2d_16 (Conv2D)           (None, 15, 15, 128)       147584    
            _________________________________________________________________
            max_pooling2d_16 (MaxPooling (None, 7, 7, 128)         0         
            _________________________________________________________________
            flatten_4 (Flatten)          (None, 6272)              0         
            _________________________________________________________________
            dropout_4 (Dropout)          (None, 6272)              0         
            _________________________________________________________________
            dense_7 (Dense)              (None, 512)               3211776   
            _________________________________________________________________
            dense_8 (Dense)              (None, 9)                 4617      
            =================================================================
            Total params: 3,457,225
            Trainable params: 3,457,225
            Non-trainable params: 0
            _________________________________________________________________
     
Compiling the model:
          
          model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
          
create the augmentation configuration.
This helps prevent overfitting since we are using a small dataset
          
          train_datagen=ImageDataGenerator(rescale=1./255,  #scale the image between 0 and 1
                                rotation_range=40,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)   

          val_datagen=ImageDataGenerator(rescale=1./255)  #Do not augment validation data,perform only resclae.
          
create the image generators
          
          train_generator=train_datagen.flow(X_train,y_train,batch_size=batch_size)
          val_generator=val_datagen.flow(X_val,y_val,batch_size=batch_size)

Running the model: 

        history=model.fit_generator(train_generator,steps_per_epoch=ntrain,epochs=5,validation_data=val_generator,validation_steps=nval)

Predicting for first 10 images of test set:
       
        X_test,y_test=read_and_process_image(test_imgs[0:10]) 
        x=np.array(X_test)
        test_datagen=ImageDataGenerator(rescale=1./255)
        
    i=0
    text_labels=[]
    plt.figure(figsize=(30,20))
    for batch in test_datagen.flow(x,batch_size=1):
    pred=model.predict(batch)
    #print(pred)
    pred1=np.argmax(pred)                  #returns index of the maximum probability in prediction numpy array
    #print(pred1)
    if pred1==0:
        text_labels.append('1')
    elif pred1==1:
        text_labels.append('2')
    elif pred1==2:
        text_labels.append('3')
    elif pred1==3:
        text_labels.append('4')
    elif pred1==4:
        text_labels.append('1')
    elif pred1==5:
        text_labels.append('2')
    elif pred1==6:
        text_labels.append('3')
    elif pred1==7:
        text_labels.append('4')
    elif pred1==8:
        text_labels.append('1')
    elif pred1==9:
        text_labels.append('2')
    plt.subplot(5/columns+1,columns,i+1)
    plt.title("this is of brightness "+text_labels[i] + "on a scale of 10")
    imgplot=plt.imshow(batch[0])
    i+=1
    if i%10 == 0:
        break

   plt.show()
