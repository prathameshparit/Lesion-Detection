import matplotlib.pyplot as plt
import cv2
from CCA_Analysis import *

# %matplotlib inline
# ##Plotting - RESULT Example
# img=cv2.imread("/content/Data/Images/107.png")#original img 107.png
#
# predict1 = cv2.resize(predict, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
#
# mask=np.uint8(predict1*255)#
# _, mask = cv2.threshold(mask, thresh=255/2, maxval=255, type=cv2.THRESH_BINARY)
# cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(cv2.UMat(img[:,:,0]), cnts, -1, (255, 0, 0) , 2)
# img = cv2.UMat.get(img)
# cv2_imshow(img)


def pred_data_2(img_path):
    # to actually visualize the effect of `CHAIN_APPROX_SIMPLE`, we need a proper image

    # Detecting contours


    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    img = cv2.imread(img_path, 1)

    print(img.shape)  # this should give you (img_h, img_w, 3)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(img2, (25, 25), 0)  # apply blur for contour
    ret, binary = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)  # apply threshold to blur image

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find countour
    obj_index = contours.index(max(contours, key=len))  # find index of largest object
    contour_img = cv2.drawContours(img, contours, obj_index, (0, 255, 0), 3)  # draw coutour on original image

    plt.imshow(contour_img, cmap='gray')
    plt.show()
    plt.imsave("static/images_uploaded/image_2.png", contour_img)


img_path = "images/107.png"
pred_data_2(img_path)
