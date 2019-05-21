# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./bin/Label_Plots.png "Visualization"
[image2]: ./bin/Viz_Noise.png "Adding Noise"
[image3]: ./bin/Noise_Gray_Equ_Centre.png "... and other transformations"
[image4]: ./bin/100_generated.png "Generation Techniques"
[image5]: ./bin/Accuracy_breakdown_10.png "Accuracy Analysis By Class"
[image6]: ./bin/Label_Distributions.png "Unbalanced Data"
[image7]: ./Prediction_examples/p_main_street_has_right_of_way.jpg "Traffic Sign 1"
[image8]: ./Prediction_examples/p_no_entry.jpg "Traffic Sign 2"
[image9]: ./Prediction_examples/p_schule.jpeg "Traffic Sign 3"
[image10]: ./Prediction_examples/p_stop.jpg "Traffic Sign 4"
[image11]: ./Prediction_examples/p_work-progress-road-sign-triangle-isolated-cloudy-background.jpg "Traffic Sign 5"
[image12]: ./bin/Success_Conv1.png "Success Conv1"
[image13]: ./bin/Success_Conv2.png "Success Conv2"
[image14]: ./bin/Failure_Conv1.png "Failure Conv1"
[image15]: ./bin/Failure_Conv2.png "Failure Conv2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a collection of histograms showing the distribution of the labels in the three data sets. Clearly some labels are more well represented than others; I compensate for this with data augmentation (see below).

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

I experimented with lots of pre-processing with the aim of reducing the computational budget without overly impacting accuracy. Converting to grayscale achieved this (which is intuitive - humans can recognise signs just as well in grayscale); as did histogram-equalization and normalisation. Normalisation has the effect of conifying the loss landscape, thereby equalizing weights of each feature in the the learning process and reducing the length of journeys from random starting points to points near an apex, and so speeding up gradient descent-type learning.

Adding noise (see image) appeared to cause under-fitting so I abandoned it; of course it may be that by spending more time tweeking parameters here the model could be improved.

![alt text][image2]

This is what images look like after undergoing the following processes.

![alt text][image3]


To address the unbalanced data-set, I generated new data until all classes had at least 2000 instances. I generated new data by performing a series of 100 random affine transformations on enough relevant instances. This shows the kind of transformations used.

![alt text][image4]



Once a balanced data set was produced for learning, I applied only three pre-processing steps to all data sets (including validation and testing, of coure): conversion to grayscale, histogram-equalization and normalization. The normalization I used was an approximation; the image set (considered 32 by 32 arrays of pixels) was centred around zero (by subtracting the mean) then rescaled so the standard deviation was closer to 1 by dividing by the mean. A stricter normalization would divide by the standard deviation. This process took a very long time (10 mins) in spite of vectorization in numpy.



My final model consisted of the following layers:

| Layer         		|     Description	        					| Parameters    |
|:---------------------:|:---------------------------------------------:| :------------:|
| Input         		| 32x32x3 RGB image   							| 416           |
| Convolution (1) 5x5  	| 1x1 stride, same padding, outputs 32x32x16 	|               |
| RELU (1)				|												|               |
| Max pooling (1)      	| 2x2 stride,  outputs 16x16x16 				|               |
| Convolution (2) 5x5   | 1x1 stride, same padding, outputs 16x16x32 	|12,832         |
| RELU (2)				|												|               |
| Max pooling (2)     	| 2x2 stride,  outputs 16x16x16 				|               |
| Flatten        		| 2048        									|               |
| Dropout       		| 70%        									|               |
| Fully connected (3)	| 1256        									|2,573,544      |
| Relu          		|           									|               |
| Fully connected (4)	| 64          									|80,448         |
| Relu          		|           									|               |
| Fully connected (5)	| 43 (number of classes)        				|2,795          |
| Softmax				|            									|               |
|						|												|               |
|						|								Total params:   |2,670,035      |
 


To train the model, I used the Adam Optimizer having tried SGD initially.
Batch size: 128
Learning rate: 0.001
Epochs: 60

30 epochs were almost enough.


My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 93.7% 
* test set accuracy of 90.5%

I chose an iterative process adapting a well-known architecture.
* LeNet was chosen as a similar image classification problem. See "Iterations Log" in the notebook for a record of my trial and improvement.
* CNN is an intuitive choice for image classification because it captures multi-pixel features on several levels.
* The number of classes was multiples greater, requiring more nodes. I experimented with breadth and depth.
* Models that were too large underfitted
* Most models over-fitted; though data-balancing helped, it was notably those classes under-represented in the original data set that were the most difficult to recognize. Note how drop in accuracy of the model on valid and test sets in the 20s and 30s, corresponding to the classes with fewer instances provided. This suggests that my generation techniques didn't fully capture variation in the underlying set. Ideally an analysis would be performed of the variations, which would then be applied to the underpresented classes, in order to predict valid and test test properties (intrinsic features and contingent variation).

![alt text][image5] ![alt text][image6]


### Test a Model on New Images

Here are five German traffic signs that I found on the web.

![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]
![alt text][image11] 

The stop, no entry and priority signs look easy to classify. However the schule and roadworks signs don't; the former takes up a very small proportion of the image, meaning that non-sign characteristics feature as strongly as intrinsic ones do. The latter suffers from this issue as well as having particularly strong background features (sky, clouds, buildings).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No entry    			| No entry 										|
| Priority				| Priority										|
| Schule	      		| Roadworks 					 				|
| Roadworks 			| Double stop       							|


The model's predictions on this set score 60% accuracy. That this is well below the figures in the 90%-99% the model achieves on the other sets comes as no surprise. Looking at the first two photos, the sign takes up so little of the pictures that when rescaled to 32 by 32 it is impossible even for a human to recognise, giving a much lower Bayesian error (show the second image to someone who hasn't seen the original and they will probably have no idea what it is). Nonetheless I wanted to test the model "to destruction".

I suspect that the strong background features in the second image - a beautiful blue sky with clouds, bright building with bold lines crossing the whole image - may have misled the model. We shall see in the further analysis below (see visualizing hidden layers).


For those the model predicted correctly it had very high certainty in its top 1 pick (87%, 98% and 100% - see the notebook). As a saving grace, for those images incorrectly classified, the predictions were suitably 'cautious'; returning probabilities below 50% (44% and 33%) expresses just how little conviction it had in the single class.

(Below we visualize the features given attention in early layers in both success and fail cases from the newly downloaded photos.)

For the schule image, the top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .44         			| Roadworks   									| 
| .26     				| Priority 										|
| .14					| Dangerous curve to the right					|


For the roadworks image, the top three soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .33         			| Double curve 									| 
| .28     				| Priority 										|
| .14					| Wild animals crossing     					|

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

I will compare hidden features from two pictures from the downloaded set of five above: the success case of 'right of way' ('priority') and the failure case 'road works'.

![alt text][image12]
![alt text][image13]

In the success case, the model seems pretty good at identifying sign boundaries and some key features inside the sign. Filter 15 of Conv1, for example, shows a bright cross at the centre of the intersection of the priority symbol.

![alt text][image14]
![alt text][image15]

In the failure case, the first filters are already failing to identify intrinsic features of the road sign and instead are picking up bold background features (and possibly the diagonal copyright mark). Compare filter 6 or 9 between the success and fail cases (at Conv1 layer), for example, to see how badly the model identifies sign borders in this case.