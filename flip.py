from PIL import Image
import matplotlib.pyplot as plt
import glob
import PIL
import cv2

test_images = glob.glob('vehicles/KITTI_extracted/*.png')
for idx, fname in enumerate(test_images):
    #image = Image.open(fname)
    #image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    image = cv2.imread(fname)
    image = cv2.flip(image, 1 ) 
    write_name = 'vehicles/flip_KITTI/'+fname.split('/')[-1]
    cv2.imwrite(write_name,image)

