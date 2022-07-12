"""
Make updated body shape from updated segmentation
"""

from cmath import e
import os
import numpy as np
import cv2
from PIL import Image
import sys
import matplotlib.pyplot as plt
import skimage.exposure


(cv_major, _, _) = cv2.__version__.split(".")
if cv_major != '4' and cv_major != '3':
    print('doesnot support opencv version')
    sys.exit()


# @TODO this is too simple and pixel based algorithm
def body_detection(image, seg_mask):
    # binary thresholding by blue ?
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(image, image, mask=mask)

    # binary threshold by green ?
    b, g, r = cv2.split(result)
    filter = g.copy()
    ret, mask = cv2.threshold(filter, 10, 255, 1)

    # at least original segmentation is FG
    mask[seg_mask] = 1

    return mask


def make_body_mask(data_dir, seg_dir, image_name, save_dir=None):
    """
    Make body mask from initial person image
    
    args:
        data_dir: directory of initial person image
        seg_dir: directory of parse-new image
        image_name: name of initial person image
        save_dir: directory where to save body mask
    """

    mask_name=image_name.replace("jpg","png")

    # define paths
    img_pth = os.path.join(data_dir, image_name)
    seg_pth = os.path.join(seg_dir, mask_name)

    mask_path = None
    if save_dir is not None:
        mask_path = os.path.join(save_dir, mask_name)

    # Load images
    img = cv2.imread(img_pth)
    # segm = Image.open(seg_pth)
    # the png file should be 1-ch but it is 3 ch ^^;
    gray = cv2.imread(seg_pth, cv2.IMREAD_GRAYSCALE)
    _, seg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    body_mask = body_detection(img, seg_mask)
    body_mask = body_mask + seg_mask
    body_mask[seg_mask] = 1
    cv2.imwrite(mask_path, body_mask)



def make_cloth_mask(cloth_path, mask_dir,viz=True):
    """
    Make cloth mask from initial cloth image
    
    args:
        cloth_path: directory of initial cloth image
        mask_dir: directory where to save the obtained mask
        viz: if True, visualize the obtained mask
    """

    img = cv2.imread(cloth_path, 0)
    img1 = Image.open(cloth_path).convert('RGB')
    lo = 233
    hi = 255

    ret,th_bin = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY_INV)

    # Filling operation:
    # Copy the thresholded image.
    im_floodfill = th_bin.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th_bin.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    th_filled = th_bin | im_floodfill_inv

    # Morphology operation:
    kernel = np.ones((2,2),np.uint8)

    # opening for salt noise removal
    th_opened = cv2.morphologyEx(th_filled, cv2.MORPH_OPEN, kernel)

    # erosion for thinning out boundary
    kernel = np.ones((3,3),np.uint8)
    th_eroded = cv2.erode(th_opened,kernel,iterations=1)

    if viz:
        # plot figures:
        titles = ['Original Image','Binary thresholding', 'Filling',  'Image', 'Opening', 'Erosion']
        images = [img, th_bin, th_filled, img1, th_opened, th_eroded]

        for i in range(6):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])

        plt.show()
    # cv2.imwrite(cloth_path, cv2.resize(img,(192,256)))
    # cv2.imwrite(mask_dir.replace("jpg","png"), cv2.resize(th_eroded,(192,256)))

    cv2.imwrite(mask_dir, cv2.resize(th_eroded,(192,256)))




def main():
    # define paths

    # root_dir = "data/viton_resize"
    root_dir = "data/"
    img_mask_folder = "image-mask"
    cloth_mask_folder = "cloth-mask"
    seg_folder = "image-parse-new"

    # data_mode = "train"
    data_mode = "external"
    image_folder = "image"
    cloth_folder = "cloth2"

    image_dir = os.path.join(os.path.join(root_dir, data_mode), image_folder)
    cloth_dir = os.path.join(os.path.join(root_dir, data_mode), cloth_folder)
    seg_dir = os.path.join(os.path.join(root_dir, data_mode), seg_folder)

    image_list = sorted(os.listdir(image_dir))
    cloth_list = sorted(os.listdir(cloth_dir))
    seg_list = sorted(os.listdir(seg_dir))

    mask_dir = os.path.join(os.path.join(root_dir, data_mode), img_mask_folder)
    cloth_mask_dir = os.path.join(os.path.join(root_dir, data_mode), cloth_mask_folder)

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.exists(cloth_mask_dir):
        os.makedirs(cloth_mask_dir)

    # for each in zip(image_list, seg_list):
    #     make_body_mask(image_dir, seg_dir, each[0], each[1], mask_dir)
    
    for each in cloth_list:
        cloth_path = os.path.join(cloth_dir, each)
        print(cloth_path,os.path.join(cloth_mask_dir,each))
        make_cloth_mask(cloth_path, os.path.join(cloth_mask_dir,each))
        

if __name__ == '__main__':
    main()