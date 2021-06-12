# gesture-project
pgdm project
![](https://komarev.com/ghpvc/?username=arunit737&color=blue)


# Gesture Recognition from video frames using custom generator

Developed by:

### Github Details:

 Arunit Mukherjee 

[<img src="https://img.shields.io/badge/github-%2312100E.svg?&style=for-the-badge&logo=github&logoColor=white" />](https://github.com/arunit737?tab=repositories) [<img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/arunit-mukherjee-7924191b4/) [<img src = "https://img.shields.io/badge/kaggle-%3390FF.svg?&style=for-the-badge&logo=kaglle&logoColor=white">](https://www.kaggle.com/arunitm) 


### Problem Statement

Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart
televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by
the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

| Gesture | Corresponding Action |
| --- | --- | 
| Thumbs Up | Increase the volume. |
| Thumbs Down | Decrease the volume. |
| Left Swipe | 'Jump' backwards 10 seconds. |
| Right Swipe | 'Jump' forward 10 seconds. |
| Stop | Pause the movie. |

Each video is a sequence of 30 frames (or images).

### Objectives:

1. **Generator**:  The generator should be able to take a batch of videos as input without any error. Steps like
   cropping, resizing and normalization should be performed successfully.
   
   ### Please note that the code will work where two csv files;one for train and test has been created. Each entry in the csv table corresponds to the path of sub folders containing several images in sequence for each gesture. In case you want to create your own video generator that uses individual csv paths then follow the below links:
   https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
   https://medium.com/@anuj_shah/creating-custom-data-generator-for-training-deep-learning-models-part-3-c239297cd5d6
   

2. **Model**: Develop a model that is able to train without any errors which will be judged on the total number of
   parameters (as the inference(prediction) time should be less) and the accuracy achieved. As suggested by Snehansu,
   start training on a small amount of data and then proceed further.

3. **Write up**: This should contain the detailed procedure followed in choosing the final model. The write up should
   start with the reason for choosing the base model, then highlight the reasons and metrics taken into consideration to
   modify and experiment to arrive at the final model.

### Installation:

Run ***pip install -r requirements.txt*** to install all the dependencies.

### Dataset:

The training data consists of a few hundred videos categorised into one of the five classes. Each video (typically 2-3
seconds long) is divided into a sequence of 30 frames(images). These videos have been recorded by various people
performing one of the five gestures in front of a webcam - similar to what the smart TV will use.
Note that all images in a particular video subfolder have the same dimensions but different videos may have different
dimensions. Specifically, videos have two types of dimensions - either 360x360 or 120x160 (depending on the webcam used
to record the videos).

## Two Architectures: 3D Convs-FC  and TimeDistributed 2D CNN-RNN-FC 

CONV 3D ARCHITECTURES

3D convolutions applies a 3 dimentional filter to the dataset and the filter moves 3-direction (x, y, z) to calcuate the low level feature representations. Their output shape is a 3 dimentional volume space such as cube or cuboid. They are helpful in event detection in videos, 3D medical images etc. They are not limited to 3d space but can also be applied to 2d space inputs such as images.
![image](https://user-images.githubusercontent.com/69101964/120886169-353b8d80-c60a-11eb-9556-91c4d9ca4be4.png)


When applied to video analysis problems,it is desirable to capture the motion information en-coded in multiple contiguous frames. To this end, wepropose to perform 3D convolutions in the convolutionstages of CNNs to compute features from both spa-tial and temporal dimensions. The 3D convolution isachieved by convolving a 3D kernel to the cube formedby stacking multiple contiguous frames together. Bythis construction, the feature maps in the convolutionlayer is connected to multiple contiguous frames in theprevious layer, thereby capturing motion information.A 3D convolutional kernel can only extractone type of features from the frame cube, since thekernel weights are replicated across the entire cube. Ageneral design principle of CNNs is that the numberof feature maps should be increased in late layers bygenerating multiple types of features from the sameset of lower-level feature maps.  Similar to the caseof 2D convolution, this can be achieved by applyingmultiple 3D convolutions with distinct kernels to the same location in the previous layer.
![image](https://user-images.githubusercontent.com/69101964/120886642-9e240500-c60c-11eb-93eb-49dc3824891b.png)
![image](https://user-images.githubusercontent.com/69101964/120886739-1c80a700-c60d-11eb-867f-c7d660f75ca9.png)

TIME DISTRIBUTED CONV2D LAYERS FEEDING INTO LSTM/GRU

Detecting an action is possible by analyzing a series of images that are taken in time. This is a very nice visualisation of the architecture
![image](https://user-images.githubusercontent.com/69101964/120886968-2ce55180-c60e-11eb-87e2-aef29e362e83.png)
We need to pass the sequences of images that creates a video file one after the other to the convolution layers to detect spatial features and then pass the feature vector to a LSTM/GRU layer to extract temporal features. This will help extract the spatiotemporal features of the video clip for classification. 'TimeDistributed layer apply the same layer to several inputs. And it produce one output per input to get the result in time'. 
![image](https://user-images.githubusercontent.com/69101964/120887172-fb20ba80-c60e-11eb-8a1e-295e674deeec.png)
We now need the network to use memory and enhance the prediction by taking advantage of LSTM. The same Conv2D layer is trained for all the images frames in time and feeds it into the LSTM/GRU layer to capture the temporal relations. 
![image](https://user-images.githubusercontent.com/69101964/120889068-23adb200-c619-11eb-95d3-32598f5d1a4f.png)

For more information on LSTM and GRUs refer to the following:
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21



## General overview of program

<span style="color: green"> We create super class Model builder that is an Abstract base class and implements an abstract method to define models. This structure of code helps to create several models,to test the various accuracies without having to retype code. The usage becomes more compact. 

We use Abstract Base Class to create superclass 'Model Builder' that  will not allow other users to create 
objects of this super class. The abstract method decorator define_model forces the user to create this method whenever
the user instantiates any subclass inheriting this superclass. 
The benefits of using ABCMeta classes to create abstract classes is that your IDE and Pylint will 
indicate to you at development time whether your inheriting classes conform to the class definition 
that you've asked them to.

Abstract interfaces are not instantiated directly in your scripts, 
but instead implemented by subclasses that will provide the implementation code for 
the abstract interface methods. 
An abstract interface method is a method that is declared, but contains no implementation. 
The implementation happens at the class that inherits the abstract class.

We use 3 methods namely;initialize_path, initialize_image_properties,initialize_hyperparams to repeatedly
provide hyperparameters to our models for every model we create.
We initialize the number of image frame sequences and provide that to the method called generator.
The method generator is used to create multiple batches of data for training. 
Each video file has been converted to sequence of images. We read from the train csv file,and randomly select the image folders
and ask the generator method to create total number of batches based on our batch size. 

For each batch we call on the method one_batch_data that creates smaller sub-batches equal to batch-size 
and depending on the number of frames to sample hyperparameter it picks up sequences of images based on temporal stride given.
The sequence of images are then processed and augmented using open cv code. We resize the image to user-requested image shape
, as our data has two types of resolutions, and then normalize them. We use matrix transformations both linear and rotational 
using warp_affine transformations where a matrix is multiplied to shift or rotate our image and then we crop out the dark portions
If the user desires augmentation we double our train data that is being generated. Augmentation minimizes overfitting

One_batch_method is a method and returns sub-batches and labels for each batch iteration to the generator which yields these
to the train_model method. This method accepts the user defined model as a parameter and uses the fit generator keras
method to train our model using our custom generator method. Keras callbacks using Model_checkpoint and LRonPLateau
also have been used to save our model checkpoints for easy model saving for later use. Whenever there is validation loss
improvement the model creates a .h5 model which can be reused later. The learning rate is reduced whenever these is no 
improvement in validation loss. </span>

#### Data Preprocessing

We can apply several of the image procesing techniques for each of image in the frame.

#### Cropping
We use our 'model_builder' class to set the image shape as desired by the user, this will transform all images to the same dimension. Also the training time is affected.

#### Normalization

We will use mean normaliztion for each of the channel in the image.

#### Data Agumentation

We have a total of 600+ for test set and 100 sampels for validation set. We will increase this 2 fold by usign a simple
agumentiaton technique of affine transforamtion.

#### Affine Transformation

The cv2.warpAffine() function mainly uses the transformation matrix M to transform the image such as rotation, affine, translation, etc. We only need to provide a 2*3 transformation matrix M to transform the image. It is generally used together with the two functions cv2.getRotationMatrix2D and cv.GetAffineTransform. These two functions are used to obtain the transformation matrix M, so that we don’t need to set M ourselves.
Refer to the below link for good examples of opencv transformations
https://towardsdatascience.com/transformations-with-opencv-ff9a7bea7f8b



#### Generators
##### Reading Video as Frames

We use 3 methods namely;initialize_path, initialize_image_properties,initialize_hyperparams to repeatedly
provide hyperparameters to our models for every model we create.
We initialize the number of image frame sequences and provide that to the method called generator.
The method generator is used to create multiple batches of data for training. 
Each video file has been converted to sequence of images. We read from the train csv file,and randomly select the image folders
and ask the generator method to create total number of batches based on our batch size. 

For each batch we call on the method one_batch_data that creates smaller sub-batches equal to batch-size 
and depending on the number of frames to sample hyperparameter it picks up sequences of images based on temporal stride given.
The sequence of images are then processed and augmented using open cv code. We resize the image to user-requested image shape
, as our data has two types of resolutions, and then normalize them. We use matrix transformations both linear and rotational 
using warp_affine transformations where a matrix is multiplied to shift or rotate our image and then we crop out the dark portions
If the user desires augmentation we double our train data that is being generated. Augmentation minimizes overfitting

One_batch_method is a method and returns sub-batches and labels for each batch iteration to the generator which yields these
to the train_model method. This method accepts the user defined model as a parameter and uses the fit generator keras
method to train our model using our custom generator method. Keras callbacks using Model_checkpoint and LRonPLateau
also have been used to save our model checkpoints for easy model saving for later use. Whenever there is validation loss
improvement the model creates a .h5 model which can be reused later. The learning rate is reduced whenever these is no 
improvement in validation loss

'The image generator yields (N, W, H, C) data, where N is the batch size, W and H are width and height, and C is the number of channels (3 for RGB, 1 for grayscaled images).
But we need to send a sequence, we need to send several frames. The needed shape is (N, F, W, H, C) — where F is the number of frames for our sequence. For example, if we train a sequence of 5 images that are RBG and with 112x112 size, the shape should be (N, 5, 112, 112, 3).
The image data generator from Keras cannot produce such a format.' ### Luckily our dataset has same number of frames and the main action has been captured across these frames.



##### Implementation

Testing of different combinations of batch-size, number of image frames and resolution to reach max GPU performance before OOM is done.
We find the following by testing few epochs on a CONV 3D model:
 Inferences:
More training time: increase frames, increase resolution
No effect on training time: batch size change
OOM error of VRAM: increase in batch size



### Model #1

Params:    
             Type: 4 X Conv3D + FC layers
             3D kernel size: 3,3,3
             Dense Layer units: 64
             Resolution:160x160
             frames:20
             batch_size:20

![image](https://user-images.githubusercontent.com/69101964/120893523-64192a00-c631-11eb-8dde-a298185d68c5.png)
We see large overfitting problem, so we will use data augmentation during using the generator and check performance. We have introduced a rotation code using cv2 rotation transformation in addition to the above transformations.


###Model #2

###### we create a new model builder super class that includes a rotation matrix for augmentation. 

Params:    
             Type: 4 X Conv3D + FC layers
             3D kernel size: 3,3,3
             Dense Layer units: 64
             Resolution:160x160
             frames:16
             batch_size:10
             Conv3d units reduced
             
             
![image](https://user-images.githubusercontent.com/69101964/120897703-26260100-c645-11eb-9cd7-635a75e2b048.png)
             
             
Model Name  Model Accuracy  Model Loss  Val Accuracy    Val Loss
0  reduced batch size       58.220214  111.723995     17.000000  326.866364
0          conv3d_aug       74.057317   69.493556     83.999997   40.065694





### Model #3

We will now use time distributed 2D-CNN layer instead of 3d-CNN and we will add LSTM layer
Params:    
             Type: 4 X Time distributed conv2d  + GRU + FC
             3D kernel size: 3,3
             Dense Layer units: 64
             Resolution:140x140
             frames:16
             batch_size:20
             
   ![image](https://user-images.githubusercontent.com/69101964/120897679-0abaf600-c645-11eb-9c83-24f246faf22c.png)
         
    Model Name  Model Accuracy  Model Loss  Val Accuracy    Val Loss
0  reduced batch size       58.220214  111.723995     17.000000  326.866364
0          conv3d_aug       74.057317   69.493556     83.999997   40.065694
0     conv2d_time_GRU       85.746604   47.679263     81.000000   71.832681         




### Model #4

Params:    
             Type: 2 X time distributed conv2d + conv2dLSTM + FC
             3D kernel size: 3,3
             Dense Layer units: 64
             Resolution:130x130
             frames:16
             batch_size:15
             
 ![image](https://user-images.githubusercontent.com/69101964/120910003-aa55a400-c698-11eb-99b6-7108b8130514.png)
Model Name  Model Accuracy  Model Loss  Val Accuracy  \
0      reduced batch size       58.220214  111.723995     17.000000   
0              conv3d_aug       74.057317   69.493556     83.999997   
0         conv2d_time_GRU       85.746604   47.679263     81.000000   
0  conv2d_time_conv2dlstm       67.948717   81.311673     72.000003   

     Val Loss  
0  326.866364  
0   40.065694  
0   71.832681  
0   86.353225              



### Model #5

Params:    
             Type: 4 X time distributed conv2d + LSTM + FC
             3D kernel size: 3,3
             Dense Layer units: 64
             Resolution:130x130
             frames:16
             batch_size:5
             
 ![image](https://user-images.githubusercontent.com/69101964/120910019-cb1df980-c698-11eb-9d3b-0543046f7bed.png)
             Model Name        Model Accuracy     Model Loss   Val Accuracy  
          reduced batch size       58.220214      111.723995     17.000000   
           conv3d_aug              74.057317      69.493556     83.999997   
        conv2d_time_GRU             85.746604      47.679263     81.000000   
        conv2d_time_conv2dlstm      67.948717       81.311673     72.000003   
         TimeConv2DLSTM             88.536954       36.345375     73.000002   

  
 
 
 
             
             
### Model #6   

Params:    
             Type: time distributed MobileNet with all layers set for training + LSTM + FC
             Dense Layer units: 64
             Resolution:130x130
             frames:16
             batch_size:5
             
             
             
 ![image](https://user-images.githubusercontent.com/69101964/120910103-8cd50a00-c699-11eb-9bb0-e52c03f8ad84.png)
            
             
   Model Name               Model Accuracy          Model Loss       Val Accuracy  
0  reduced batch size            58.220214          111.723995         17.000000   
0  conv3d_aug                    74.057317           69.493556         83.999997   
0  conv2d_time_GRU               85.746604           47.679263         81.000000   
0  conv2d_time_conv2dlstm        67.948717           81.311673         72.000003   
0  TimeConv2DLSTM                88.536954           36.345375         73.000002     









### Model #7


Params:    
             Type: time distributed MobileNet with all layers set for training + GRU + FC
             Dense Layer units: 128
             Resolution:130x130
             frames:16
             batch_size:5
             GRU cells: 128
             
             
 ![image](https://user-images.githubusercontent.com/69101964/120910110-98283580-c699-11eb-8c95-1a7ae4a96611.png)





### REFERENCES
   * https://www.dbs.ifi.lmu.de/~yu_k/icml2010_3dcnn.pdf
   * https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00
              
             
