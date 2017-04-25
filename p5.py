import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
import pickle
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from scipy.ndimage.measurements import label

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)
        
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
           on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows



# Read in cars and notcars
car_images = glob.glob('data/vehicles/*/*.png')
notcar_images = glob.glob('data/non-vehicles/*/*.png')
cars = []
notcars = []
for image in car_images:
    cars.append(image)

for image in notcar_images:
    notcars.append(image)


# Number of car examples
n_car = len(car_images)
# Number of non-car examples.
n_noncar = len(notcar_images)

print("Number of car examples =", n_car)
print("Number of non-car examples =", n_noncar)



color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

## Read data
#with open('X.p','rb') as f:
#    X = pickle.load(f)
#
#with open('y.p','rb') as f:
#    y = pickle.load(f)


# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

print("Number of Samples = ", format(len(y)))
print("Number of Features = ", format(X.shape))

## Store X and y data to pickle
#with open('X.p','wb') as f:
#    pickle.dump(scaled_X ,f)
#
#with open('y.p','wb') as f:
#    pickle.dump(y,f)



# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
print('random state = ', rand_state)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Check the training time for the SVC
t=time.time()

# Use a linear SVC
clf = LinearSVC(max_iter=10000,verbose=3)
clf.fit(X_train, y_train)
joblib.dump(clf, 'classify_car.pkl')

#clf = joblib.load('classify_car.pkl')


t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, xstop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    # bbox_list to store all box found in image
    bbox_list = []
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),
                              (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart),(0,0,255),6)
                bbox_list.append(((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart)))

    return draw_img, bbox_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



class vehicles():
    def __init__(self):
        self.labels = None
        # temp parameter to store one frame info
        self.numofcar = 0
        self.box_list = []
        self.center = []

        # parameter for n average frames
        self.previous_numofcar = 0
        self.avg_box_list = []
        self.avg_center = []

        self.temp_box_list = []
        self.temp_center = []
    
    def get_box_list_center(self):
        # Iterate through all detected cars in current frame
        for car_number in range(1, self.numofcar + 1):
            
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            if((np.max(nonzerox)-np.min(nonzerox))>=32 and (np.max(nonzeroy)-np.min(nonzeroy))>32): #64
                self.center.append( ((np.min(nonzerox) + np.max(nonzerox))//2, (np.min(nonzeroy)+np.max(nonzeroy))//2) )
                self.box_list.append( ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))) )
        self.numofcar = len(self.box_list)
      
    def combine_nearby_boxes(self):
        # sort all the car objects from left to right
        if(self.numofcar >= 2):
            self.center,self.box_list= (list(t) for t in zip(*sorted(zip(self.center,self.box_list))))
            print(self.center)
            print(self.box_list)
            
            self.temp_center = self.center[:]
            self.temp_box_list = self.box_list[:]
            
            # combine box nearby if more than 2 boxes
            if(self.previous_numofcar>0):
                for i in range(0,self.numofcar-1):
                    center_avg_x = (self.center[i][0] + self.center[i+1][0])//2
                    center_avg_y = (self.center[i][1] + self.center[i+1][1])//2
                    
                    for j in range(0,self.previous_numofcar):
                        temp_x = (self.avg_box_list[j][1][0]-self.avg_box_list[j][0][0])*0.2
                        temp_y = (self.avg_box_list[j][1][1]-self.avg_box_list[j][0][1])*0.2
                        
                        # combine when average_center_location < 0.1*dimension +/- previous_locationa
                        if( self.avg_center[i][0]-temp_x<center_avg_x<self.avg_center[i][0]+temp_x
                           and self.avg_center[i][1]-temp_y<center_avg_y<self.avg_center[i][1]+temp_y):
                        # Replace 2 boxes with new box in list
                            self.temp_center.remove(self.center[i])
                            self.temp_center.remove(self.center[i+1])
                            self.temp_center.append((center_avg_x,center_avg_y))
                        
                            new_box_x0 = np.min(self.box_list[i][0][0],self.box_list[i+1][0][0])
                            new_box_x1 = np.max(self.box_list[i][1][0],self.box_list[i+1][1][0])
                            new_box_y0 = np.min(self.box_list[i][0][1],self.box_list[i+1][0][1])
                            new_box_y1 = np.max(self.box_list[i][1][1],self.box_list[i+1][1][1])
                            
                            self.temp_box_list.remove(self.box_list[i])
                            self.temp_box_list.remove(self.box_list[i+1])
                            self.temp_box_list.append(((new_box_x0,new_box_y0),(new_box_x1,new_box_y1)))
                                    
        # sort again for all the car objects from left to right
        if(len(self.temp_box_list) >= 2):
            self.temp_center,self.temp_box_list= (list(t) for t in
                                                  zip(*sorted(zip(self.temp_center,self.temp_box_list))))
            print(self.temp_center)
            print(self.temp_box_list)
                        
                        
                        
    def update(self):
        
        print(self.labels[1])
        
        self.numofcar = self.labels[1]
        self.box_list = []
        self.center = []
        self.get_box_list_center()
                                                 
        if(self.previous_numofcar == 0):
            self.avg_box_list = self.box_list[:]
            self.avg_center = self.center[:]
        
        self.combine_nearby_boxes()
        self.numofcar = len(self.temp_box_list)
        print('number of car after combine: ', self.numofcar)
                                                 
        # if find same amount of cars in next pciture
        if(self.numofcar == self.previous_numofcar):
           print('enter 1')
        
        # if find more cars in next pciture
        elif (self.numofcar < self.previous_numofcar):
           print('enter 2')
        
        # if find less cars in next pciture
        elif (self.numofcar > self.previous_numofcar):
           print('enter 3')
    
        # ToDo: Do average
        self.avg_center = self.temp_center[:]
        self.avg_box_list = self.temp_box_list[:]
        self.previous_numofcar = len(self.avg_box_list)



# Multi-Scale prediction and window size settings
ystarts = [360, 360, 360, 360, 360]
ystops  = [560, 600, 660, 660, 660]
xstarts = [448, 480, 512, 880, 960]
xstops  = [1280,1280,1280,1280,1280]
scales  = [1.0, 1.25,1.5, 2.0, 2.25]


#test_images = glob.glob('test_images/*.jpg')
#for idx, fname in enumerate(test_images):
#    test_img = plt.imread(fname)


def process_image(test_img):

    bbox_list_all = []
    for scale, ystart, ystop, xstart, xstop in zip(scales, ystarts, ystops, xstarts, xstops):

        out_img, bbox_list = find_cars(test_img, ystart, ystop, xstart, xstop,
                                   scale, clf, X_scaler, orient, pix_per_cell, cell_per_block,
                                   spatial_size, hist_bins)
##### for processing image debug output
#        write_name = 'output_images/find_cars/'  + 'scale_' + str(scale) + '_' + fname.split('/')[-1]
#        plt.imsave(write_name, out_img)
##### end debug

        for box in bbox_list:
            bbox_list_all.append(box)

##### for processing image debug output
#    pickle.dump(bbox_list_all, open(fname+'_bbox_pickle.p', 'wb' ))
#        #    box_list = pickle.load(open(fname+'_bbox_pickle.p', 'rb'))
#    print(fname, bbox_list_all)
##### end debug

    heat = np.zeros_like(test_img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,bbox_list_all)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    car.labels = label(heatmap)
    car.update()

    draw_img = draw_boxes(test_img, car.avg_box_list)
               

##### for processing image debug output
#    # Store Heatmap
#    write_name = 'output_images/heatmap/' + fname.split('/')[-1]
#    plt.imsave(write_name, heatmap)
#    
#    # Store Final Draw Box on Image
#    write_name = 'output_images/final_draw_debug/' + fname.split('/')[-1]
#    plt.imsave(write_name, draw_img)
##### end debug

    return draw_img




car = vehicles()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


project_video_output = 'output_videos/project_video_output.mp4'
#clip1 = VideoFileClip('project_video.mp4')
clip1 = VideoFileClip('test_video.mp4')

project_video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
project_video_clip.write_videofile(project_video_output, audio=False)

