# create a folder to store extracted images
# http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
# use opencv to do the job
import cv2
import os
folder = 'test'

print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture('test_video.mp4')
count = 0
while True:
    success,image = vidcap.read()
    if not success:
        break
    cv2.imwrite(os.path.join(folder,"frame{:d}.png".format(count)),image)     # save frame as JPEG file
    count += 1
print("{} images are extacted in {}.".format(count,folder))
