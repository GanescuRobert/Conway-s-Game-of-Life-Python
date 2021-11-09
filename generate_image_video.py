
import cv2
import os
import glob
from PIL import Image as im
import numpy as np
indx = -1


def resize_image(filename, result):
    # open image for resize
    img = cv2.imread(filename)
    output = cv2.resize(img, (600, 600), fx=0, fy=0,
                        interpolation=cv2.INTER_NEAREST)

    # save it
    cv2.imwrite(filename, output)


def generate_name(name):
    global indx
    indx += 1

    indx_str_size = len(str(indx))
    num_of_zeros = 6-indx_str_size if 6-indx_str_size > 0 else 1
 
    zeroes = ''.join([str(_) for _ in np.zeros(num_of_zeros,int)])

    return 'temp/'+name+zeroes+str(indx)+'.png'


def preprocess_data(data):
    return (data*255).astype(np.uint8)


def delete_images():
    files = glob.glob('temp/*')
    for file in files:
        os.remove(file)


def generate_image(name, data):
    # preprocess data
    filename = generate_name(name)
    data = preprocess_data(data)
    # write image
    cv2.imwrite(filename, data)
    resize_image(filename, data)


def generate_video(file_name):
    image_folder = 'temp'
    video_name = 'output_videos/' +file_name+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define the video codec

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    fps = 1.5
    if(len(images) > 100):
        fps = 3
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for img in images:
        video.write(cv2.imread(os.path.join(image_folder, img)))

    cv2.destroyAllWindows()
    video.release()
    delete_images()
