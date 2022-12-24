import matplotlib.pyplot as plt
import cv2
from CCA_Analysis import *

# plt.figure(figsize = (20,10))
# plt.title("Predict Mask",fontsize = 40)
# plt.imshow(predict)
# #For CCA, we saved
# plt.imsave("/predict.png",predict)
#
# ##Plotting - RESULT Example
# img=cv2.imread("images_uploaded/107.png")#original img 107.png
#
# predict1 = cv2.resize(predict, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
#
# mask=np.uint8(predict1*255)#
# _, mask = cv2.threshold(mask, thresh=255/2, maxval=255, type=cv2.THRESH_BINARY)
# cnts,hieararch=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# img = cv2.drawContours(cv2.UMat(img[:,:,0]), cnts, -1, (255, 0, 0) , 2)
# img = cv2.UMat.get(img)
# # cv2_imshow(img)

##Plotting - RESULT Example with CCA_Analysis

def pred_data(img_path):
    img=cv2.imread(img_path)#original img 107.png

    #load image (mask was saved by matplotlib.pyplot)
    predicted=cv2.imread("predict.png")

    predicted = cv2.resize(predicted, (img.shape[1],img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    cca_result,teeth_count=CCA_Analysis(img,predicted,3,2)
    # cv2_imshow(cca_result)

    plt.imsave("static/images_uploaded/image.png",cca_result)


img_path = "images/107.png"
pred_data(img_path)