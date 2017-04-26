**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/data augement.png
[image2]: ./examples/hog_example.png
[image3]: ./examples/sliding_windows.png
[image5]: ./examples/images_and_heat.png
[image6]: ./examples/labels_and_bboxes.png
[video1]: ./project_video.mp4
[video2]: ./output_video/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Augmentation

#### 1. vehicle data Augmentation

The code for this step is contained in `flip.py`.
I flipped the pictures in vehicles/KITTI_extracted/, which increases vehicles images by 5966 to total number of 14758.

#### 2. non-vehicle data Augmentation

The code for this step is contained in `extract_frame.py` and `crop.py`.

I extracted pictures from each frame of test_video. Then I cropped each frame to 64x64 size png on the left half because there is no car in those areas. After the extraction, it generates lots of blue sky images. Based on model training experience, I removed those sky images, only left tree images or road images. This increased non-vehicle images by 3462 to total number of 12430.

Here is an examples of the `vehicle` and `non-vehicle` classes:
![alt text][image1]

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in my code) I extracted HOG features from the training images.

The code for this step is contained in the first code cell [2] of the IPython notebook. (get_hog_features function)

I started by reading in all the `vehicle` and `non-vehicle` images, and then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how I settled on my final choice of HOG parameters.

I tried various combinations of parameters and set the parameters as follows:

```
color_space = 'YCrCb'       # Color Space
orient = 9                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = "ALL"         # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)     # Spatial binning dimensions
hist_bins = 16              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off

```

#### 3. Describe how (and identify where in my code) I trained a classifier using my selected HOG features (and color features if I used them).

The code is in code cell [3]. I trained a linear SVM using sklearn.svm (LinearSVC). First need to extract
features for both car and non-car images, combined with HOG features, spatial features and hist features. Then fit a per-column scaler and apply the scaler to X training data. Later train_test_split (from sklearn.cross_validation import train_test_split) to seperate training data and test data (80:20). Final training results shows below:

```
Number of Samples =  27188
random state =  49
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8412
[LibLinear]216.97 Seconds to train SVC...
Test Accuracy of SVC =  0.9919
```


### Sliding Window Search

#### 1. Describe how (and identify where in my code) I implemented a sliding window search.  How did I decide what scales to search and how much to overlap windows?

The code is in code cell [4] find_cars function, a single function that can extract features using hog sub-sampling and make predictions.

I chose 64 as the orginal sampling rate, with 8 cells and 8 pix per cell. I set cells_per_step to 2 which means (8-2)/8 = 75% overlap. Multi-Scale windows (scales x 64) searching was used as below:

'''
# Multi-Scale prediction and window size settings
ystarts = [360, 360, 360, 360, 360]
ystops  = [560, 600, 660, 660, 660]
xstarts = [448, 480, 512, 880, 960]
xstops  = [1280,1280,1280,1280,1280]
scales  = [1.0, 1.25,1.5, 2.0, 2.25]

'''


#### 2. Show some examples of test images to demonstrate how my pipeline is working.  What did I do to optimize the performance of my classifier?

Ultimately I applied 5 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to my final video output.  My pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as I am identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_video/project_video_output.mp4)


#### 2. Describe how (and identify where in my code) I implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap and the resulting bounding boxes are drawn onto the frame in the series:

![alt text][image6]

#### 3. Describe how I implemented vehicles class for object tracking.

The code is in code cell [8], class vehicles(). I set this class parameters as follows:

```
# Video average number of frame
self.n = 7

# temp parameter to store one frame info
self.numofcar = 0
self.box_list = []
self.center = []
self.temp_box_list = []
self.temp_center = []

# parameter for n average frames
self.previous_numofcar = 0
self.avg_box_list = []
self.avg_center = []
self.car_freq = []

```

For each frame, I first update box_list and center from current frame using `get_box_list_center()`, then I use `combine_nearby_boxes()` to combine nearby boxes and reduce false detection. Later it will decide 3 cases comparing current car number and previous car number: SAME amount, LESS, or MORE.

If find MORE cars in the frame, I will apend the new box into avg_box_list and set this box frequency to 1.

If find SAME cars in the frame, I will increase each box' frequency in avg_box_list by 1. The max number of the car frequency is 5.

If find LESS cars in the frame, I will still keep the missed box in current frame but reduce its frequency by 1. Also if found 2 cars overlap and combined to a larger box in current frame, I will split the large box in the middle. If the car frequency is 0, I will remove the box from the avg_box_list.

After this I will do `average()` to get a smooth box location. To plot those boxes on the frame, I only choose those boxes which has frequncy more than 2. This could help to reduce the false detection on the video. 


---

### Discussion

#### 1. Briefly discuss any problems / issues I faced in my implementation of this project.  Where will my pipeline likely fail?  What could I do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



