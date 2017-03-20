# **Traffic Sign Recognition** 

---

## Project Goals

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[example_images]: ./example_images.png "Example Images"
[unbalanced_dataset]: ./unbalanced_dataset.png "Unbalanced Dataset"
[equalized_images]: ./equalized_images.png "Equalized Images"
[balanced_dataset]: ./balanced_dataset.png "Balanced Dataset"
[no_entry]: ./no_entry.png "No Entry"
[priority_road]: ./priority_road.png "Priority Road"
[right_of_way]: ./right_of_way.png "Right of Way"
[road_work]: ./road_work.png "Road Work"
[stop_sign]: ./stop_sign.png "Stop Sign"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/mkolod/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in code cell #2 of the IPython notebook.  

In cell #2, I used NumPy to determine the number of training examples (34,799), as well as validation and testing examples (4,410 and 12,630, respectively). 

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in code cells #5 and #7 of the IPython notebook.  

In code cell #5, I display 15 examples of each traffic sign class, for all 43 classes. As one can see on those images, the image quality isn't uniform - some have very good contrast and brightness, others don't. This persuaded me to pursue image adjustments prior to feeding the data into the convolutional neural network, e.g. histogram equalization (see later sections).

![Example Images][example_images]

In code cell #7, I plotted the image counts per class. It is clear from that image that this is a very unbalanced dataset, with some classes having about 180 examples, while others included close to 2,000 examples. This prompted me to oversample the rare classes, and to include image augmentation (e.g. rotation, saturation adjustment, etc.) to increase the effective size of the dataset, and to help the model learn to recognize sign patterns while ignoring certain image distortions or other idiosyncracies.

![Unbalanced Dataset][unbalanced_dataset]


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I decided not to convert images to grayscale, since color has significance for human users of these signs. For example, even if the number inside a white circle with a red border isn't clearly visible, one can still narrow down the sign to a speed limit. I tried grayscaling the images and found that this caused a drop in accuracy, so I used all 3 color channels.

The preprocessing that I did do included histogram normalization, to make contrast and brightness in the images more equal. This helped improve accuracy and produced images that were also easier to assess to a human (I couldn't decipher some images without histogram equalization myself). This code can be found in the equalize_hist() function in code cell #8. The result of this adjustment can be seen below - compare these images to the original images presented earlier. 

![Equalized Images][equalized_images]

Another issue had to do with balancing the dataset. Since the class with the highest count had 2,010 examples but the smallest had 180, I oversampled the less frequent classes to improve the learning of those classes. The code for balancing the classes by oversampling can be found in code cell #12. As you can see below, the class membership got balanced as a result - this visualization was executed in code cell #15.

![Balanced Dataset][balanced_dataset]

Since this also artificially put more emphasis on identical images from the oversampled images, this could have lead to memorization (overfitting) rather than generalization. To address this, I augmented the dataset by applying random rotations and saturation adjustments The code for these rotations and saturation adjustments can also be found in code cell #8.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The training, validation and testing data were already given. I could have probably concatenated the dataset and reshuffled it, then split it again into training, validation and testing, but I assumed that the idea was to use the splits provided in the pickled files, which were loaded in code cell #1.

To recap, the breakdown was as follows:
* training data: 34,799 images
* validation data: 4,410 images
* testing data: 12,630 images

The reason for both a training and a validation split was that I kept using the validation dataset to tweak the hyperparameters. Therefore, even though the model hasn't seen the validation data during training, it has indirectly seen validation data, since my hyperparameter changes were aimed at improving validation accuracy. Therefore, the final accuracy needed to be based on data that has not been seen at all, either for training or for hyperparameter tuning.

Note that the effective size of the training set was actually larger than the above mentioned 34,799 images, due to oversampling of the rare classes, which resulted in the total dataset size being  86,430 (most frequent class with 2,010 images times 43 classes, with the remaining 42 being oversampled to a count of 2,010 each). Also, since I kept applying random rotations and saturation adjustments, the theoretical size of the training set could have been closer to each image being unique, so 86,430 times the number of epochs. Since I ran for 30 epochs, the number of "unique" images after augmentation could have been 2,592,900, if none of the augmentations were the same for the same base image.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located code cell #16 in the IPython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 16x16x16 	 		  	    |
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x32  	|
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 8x8x32                    |
| Flattening layer      | output: 2,048 units                           |
| Fully connected		| 2,048x86 units        						|
| RELU                  |                                               |
| Dropout               | Keep probability = 0.5                        |
| Fully connected       | 86x43 units                                   |
| Softmax				|       									    |
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in code cell #20 of the IPython notebook, however some setup for things such as the learning rate and some placeholders was done in code cells #17-18.

I chose a learning rate of 0.001. Higher learning rates didn't work well in terms of final accuracy on the validation set, and there was no point lowering the rate further than necessary for convergence. Since using pure SGD is usually fraught with problems (one has to either choose a very low learning rate from the beginning, or start with a higher rate and apply a learning rate annealing schedule), I used the Adam optimizer here. The literature suggests that nothing really beats very heavily tuned SGD (in terms of weight decay, learning rate decay, etc.), but optimizers with a quasi "automatic" adaptive learning rate are a good place to start.

I used a batch size of 128. I tried smaller batches and they weren't helping convergence, while they were hurting the efficient use of the GPU. Sometimes smaller batches are better to provide a greater stochasticity to the optimization process, to avoid getting stuck in a bad local optimum. Paradoxically, as we use a bigger batch size and get a better estimate of the gradient, it's easier to fall into such a "sharp minumum. [This paper](https://arxiv.org/pdf/1609.04836.pdf) discusses the problem of large batches in detail.


#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in cell #18 of the IPython notebook.
fir
My final model results were:
* validation set accuracy of 0.969 
* test set accuracy of 0.950

I didn't care to report training set accuracy here, because it's irrelevant from the perspective of model generalization. Of course, we can determine if the model is underfitting if both training and validation errors are low, or overfitting when training error is low and validation error is high. However, ultimately, the tuning was based on the value of the validation error, since the model hasn't seen the validation data during training. The final assessment was done on the test set, which hasn't been indirectly seen by the model during hyperparameter tuning, which was the case with the validation dataset.

I chose an iterative approach to finding the best model. 

The starting point was LeNet-5, because the dataset was of a similar size in terms of the number of samples, as well as the image size. Another point of inspiration was the model typically used for the CIFAR-10 dataset, which had 60,000 images and 10 image classes, but unlike MNIST, it was based on color images and real-life objects and animals, such as planes and cats, rather than digits. However, number of classes was higher in the traffic sign dataset, and the data more diverse, so I determined that I would have to do some tuning. In particular, I iterated over choosing the number of convolutional filters, and ultimately increased them to 16 in the first convolutional layer and 32 in the second convolutional layer.

The other hyperparameter tuning had to do with the amount of dropout between the first fully connected layer and the second fully connected layer. Since fully connected layers have many more parameters than convolutional layers and hence contribute more to overfitting, I kept tuning the dropout "keep probability" rate, and ultimately I settled on 0.5.

Lastly, since the dataset was small, I had to tune the amount of image augmentation, such as the maximum angle by which to rotate the images, and the amount of saturation adjustment. I finally chose a maximum rotation of 20 degrees, and a maximum saturation adjustment of 10% from the original image.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![No Entry][no_entry]

![Priority Road][priority_road]

![Right of Way][right_of_way]

![Road Work][road_work]

![Stop Sign][stop_sign]

I wouldn't say these images are necessarily difficult to classify - after all, just like in the case of the original German traffic sign dataset, we have images that are cropped to mostly show the traffic sign, rather than a big image where the sign only takes a small portion of the image, so there isn't much other "noise" to confuse the classifier. Still, there may be idiosyncracies associated with data collection of the original dataset, such as model of the camera, the quality of the resizing of the original image to the 32x32 image (nearest neighbors vs. bilinear or bicubic), the effect of lossy compression (JPG) vs. lossless compression (PNG), the degree of lossiness (Q factor in JPEG compression), etc. Even the quality of the decoding of images can matter ("regular" vs. "fast" IDCT algorithm in libjpeg or libjpeg-turbo, which trades off speed for quality at decoding time). Also, I'm not sure what the quality of the original images was that were resized down to 32x32, but sometimes the original resolution matters even when resizing down. I used pretty high quality images (several megapixels) in the original images, and then took a screenshot of these big images at high resolution. Even these small thumbprints were about 300+ pixels on each side. These images also included some background, e.g. the no entry sign had the triangular shape of a yield sign which was turned the other way, the priority road sign had a house in the background, etc. So, one way or another, these images were likely more different from the training set than even the test set - they were taken by a different set of people (the original images, I mean), with different cameras, etc.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in code cell #23 of the IPython notebook


![No Entry][no_entry]

![Priority Road][priority_road]

![Right of Way][right_of_way]

![Road Work][road_work]

![Stop Sign][stop_sign]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Priority Road     	| Priority Road 								|
| Right of Way			| Right of Way									|
| Road Work	      		| Road Work 					 				|
| Stop Sign 			| Stop Sign        							    |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%. Of course, in practice one wouldn't want to only test on new images. The test set had 12,630 images, and once we test the model on a fixed dataset, we should collect more images "from the wild" to assess real-world generalization, as opposed to just generalization on the held out test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

