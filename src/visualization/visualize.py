from logging import raiseExceptions
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        Image.fromarray(array).save(os.path.join(save_dir, img_name))


def read_image(img_path,module='PIL',size=(192,256)): # module: 'PIL' or 'cv2'
    """
    Read image from path and resize it to size
    
    Args:
        img_path: path to image
        module: 'PIL' or 'cv2'
        size: size of image
    Returns:
        image: image"""
    module = module.lower()
    if module == 'pil':
        return Image.open(img_path).resize(size)

    elif module == 'cv2':
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        return np.flip(img, axis=-1) 

def show_raw_images(images, module='PIL',size=(400,400)):
    """
    Show raw images

    Args:
        images: list of images path
        module: 'PIL' or 'cv2'
        size: size of image
    """
    #show images up to 8 image-cloth pairs, choice of module: 'PIL' or 'cv2' for the reading 
    titles=[]
    shows = []
    for i in range(len(images)):
        img_path,cloth_path = images[i]
        titles.append(Path(img_path).parts[-1].replace('.jpg',''))
        shows.append((read_image(img_path, module, size=size)))
        titles.append(Path(cloth_path).parts[-1].replace('.jpg',''))
        shows.append((read_image(cloth_path, module, size=size)))

    plt.figure(figsize = (10,15))
    for i in range(len(shows)):
        plt.subplot(4,4,i+1),plt.imshow(shows[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])


def show_images(images, titles=[]):
    """
    Show raw images

    Args:
        images: list of images
        module: 'PIL' or 'cv2'
        size: size of image
    """
    #show images up to 8 image-cloth pairs, choice of module: 'PIL' or 'cv2' for the reading 

    plt.figure(figsize = (10,15))
    for i in range(len(images)):
        plt.subplot(4,4,i+1),plt.imshow(images[i],'gray')
        if len (titles) > 0:
            plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

#to concatenate 3 images horizontally (for visualization)
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

