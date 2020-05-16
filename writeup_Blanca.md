# **Traffic Sign Recognition** 

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

[image1]: ./Writeup_Images/visualization.png "Visualization"
[image2]: ./Writeup_Images/ClassID.png "ClassID"
[image3]: ./Writeup_Images/grayscale.png "grayscale"
[image4]: ./Writeup_Images/HDR.png "HDR"
[image5]: ./Writeup_Images/accuracy.jpg "accuracy"
[image6]: ./New-images/1.jpg "Traffic Sign 1"
[image7]: ./New-images/2.jpg "Traffic Sign 2"
[image8]: ./New-images/3.jpg "Traffic Sign 3"
[image9]: ./New-images/4.jpg "Traffic Sign 4"
[image10]: ./New-images/5.jpg "Traffic Sign 5"



---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is sized to 32x32 pixels in color (hence the 3 channels). 
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

I used pandas to perform a quick analysis of the data to show the amount of each type of signs in every dataset (train, validation and test). See the attached table: 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale.

Here are the same images above after converting to grayscale:
![alt text][image3]

I tried the technique HDR like effect (Histogram equalization): I tried to perform a similar effect to the HDR with OpenCV “cv2.equalizeHist” 
Here are the same images above after processing them:
![alt text][image4]


I tried normalizing the image data because it helps the neural networks to need lees effort to get to a desired point.

After several attempts with different preprocessing techniques (as seen in cell number 8) I was able to get the desired results by performing 2 steps, first HDR and second standardization. (HDR+Std) 

Data augmentation techniques were also considered as a last resort, but finally it was not needed as I got above the minimum accuracy.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1     	| Stride 1,  Padding ‘Valid’, Output 28x28x6 	|
| RELU					|												|
| Dropout 				| Keep_prob 0.5									|
| Max pooling	      	| Stride 2 ,  Output 14x14x6 				 	|
| Convolution 2		    | Stride 1,  Padding ‘Valid’, Output 10X10X16	|
| RELU					|												|
| Dropout 				| Keep_prob 0.5									|
| Max pooling	      	| Stride 2 ,  Output 5x5x16 				 	|
| Flatten				| Input = 5x5x16. Output = 400					|
| Fully connected		| Input = 400. Output = 120						|
| RELU					|												|
| Fully connected		| Input = 120. Output = 84						|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43						|
|						|												|
 
It is based on the original LeNet Architecture, but 50% dropout layer has been included after every Convolutional layer. Test were done adding dropouts in different but best numbers were obtained in this configuration.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

During the training of the model I started using the CPU and around 20 epochs. When I confirmed my net was working properly and I had noted the set ups with the best apparent outcome I switched to running it in workspaces where I performed additional testing in a GPU instance. 

The parameters I got the best results were: 

Optimizer: AdamOptimizer  
Batch size: 196 
Epochs: 20 / 100 (CPU/GPU) 
Learning Rate: 0.001 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.946 (>0.93)  
* test set accuracy of 0.926

You can see this in the cell #15. 

I started with a LeNet model adapting the input and output sizes. I added depth of 3 in the first Convolutional layer, and changed the output to 43 classes.
But even though the validation set accuracy did reach the minimum required (0.956), I didn't want to submit a well known model. I wanted to play with different configurations to see how it would impact the accuracy. As seen in cell number 11, I added dropouts after the all the Relus. I ended up removing the last 2 dropouts because it dropped the accuracy. 

I then played around with learning rate, batch size, and epochs (once in GPU) until I got the best figures.  

![alt text][image5] 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]

The first and last images (stop and roundabout sign)  might be easy to classify because these are unique. Both are not easy to mistake with others. ON the contrary the third and fourth (50 and 100 Speed limit) are easy to mistake with the other Speed limit signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop       			| Stop   										| 
| Yield     			| Yield 										|
| 100 km/h				| 100 km/h										|
| 50 km/h				| 50 km/h 					 					|
| Roundabout mandatory	| Roundabout mandatory 							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 75%), and the image does contain a stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 73%					| Stop	 										|
| 9% 					| No entry	 									|
| 4% 					| Speed limit (70km/h) 	 						|
| 3% 					| Speed limit (120km/h) 	 					|
| 2% 					| Speed limit (60km/h) 	 						|


For the second image , the model is thinks it could be a yeild sign (probability of 47%), and the image does contain a yeild sign. The top five soft max probabilities were: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 47% 					| Yield	 										|
| 38%					| Speed limit (60km/h) 			 				|
| 4% 					| Ahead only	 								|
| 2% 					| Speed limit (30km/h) 	 						|
| 2% 					| Speed limit (80km/h) 	 						|


For the third image, the model is very sure that this is a 50km/h Speed limit sign (probability of 97%), and indeed the image does contain a 50km/h Speed limit sign. The top five soft max probabilities were: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 97% 					| Speed limit (50km/h) 	 						|
| 2% 					| Speed limit (30km/h) 	 						|
| 0% 					| Speed limit (60km/h) 	 						|
| 0% 					| Speed limit (70km/h) 	 						|
| 0% 					| Speed limit (120km/h) 	 					|

For the second image , the model is relatively sure that this is a 100km/h Speed limit (probability of 77%), and it is correct. The top five soft max probabilities were: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 77% 					| Speed limit (100km/h) 			 			|
| 8% 					| Ahead only	 								|
| 8% 					| Roundabout mandatory	 						|
| 4% 					| Speed limit (50km/h) 	 						|
| 2% 					| Speed limit (30km/h) 	 						|


For the second image , the model is absolutely sure that this is a Roundabout mandatory sign (probability of 100%), and it is right. The top five soft max probabilities were: 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100% 					| Roundabout mandatory	 						|
| 0% 					| Keep right	 								|
| 0% 					| Priority road	 								|
| 0% 					| Right-of-way at the next intersection	 		|
| 0% 					| Speed limit (50km/h) 	 						|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


