import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
import scipy.misc

test_images = glob.glob('test/*.png')
for idx, fname in enumerate(test_images):
    #image = plt.imread(fname)
    image = mpimg.imread(fname)
    print(image.shape)
    width = int(1280/64/2)
    height = int(720/64)
    for i in range(0, width):
        for j in range(0, height):
            img_new = np.copy(image)
            img_new = image[64*i:64*(i+1),64*j:64*(j+1)]
            write_name = 'data/non-vehicles/ExtractFromVideo/'+str(i)+'_'+str(j)+'_'+fname.split('/')[-1]
            #plt.imsave(write_name,img_new)
            scipy.misc.toimage(img_new).save(write_name)

