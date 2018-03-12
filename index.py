import cv2;
import imutils;
from skimage import exposure
from skimage.feature import hog;
import os;
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

directory = "/Users/ozercevikaslan/Desktop/vw_logo_train_set/"

top_features = []
labels = []

# some playground
TestImg = cv2.imread("/Users/ozercevikaslan/Desktop/pear.jpg")
TestImg = np.float32(TestImg)
gx = cv2.Sobel(TestImg, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(TestImg, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.imshow("mag", mag)
cv2.imshow("angle", angle)
cv2.waitKey()
cv2.destroyAllWindows()

'''
# image preprocessing
for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print ("filename : ", filename)
            rawImage = cv2.imread("/Users/ozercevikaslan/Desktop/vw_logo_train_set/%s"%filename,0)
            print("%s" %filename)
            autoCanny = imutils.auto_canny(rawImage)
            cv2.imwrite("/Users/ozercevikaslan/Desktop/MachineLearning/assets/canny_%s.JPG" %filename,autoCanny)
            (_,cnts,_) = cv2.findContours(autoCanny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            draw = cv2.drawContours(autoCanny,cnts,-1,(0,255,0),3)
            cv2.imwrite("/Users/ozercevikaslan/Desktop/MachineLearning/assets/contour_%s.JPG"%filename,draw)
            print("bakalım",cv2.contourArea(cnts[0]));
            c = max(cnts, key=cv2.contourArea)
            (x,y,w,h) = cv2.boundingRect(c)
            logo = rawImage[y:y+h,x:x+w]
            logo = cv2.resize(logo,(200,200))
            resizedImage = cv2.resize(rawImage,(200,200))
            (H, hogFeatures) = hog(logo,orientations=9,pixels_per_cell=(8,8),cells_per_block=(3,3),block_norm='L2-Hys',transform_sqrt=True,visualise=True)
            top_features.append(H)
            labels.append("volkswagen")
            hogImage = exposure.rescale_intensity(hogFeatures,out_range=(0,255))
            hogImage = hogImage.astype("uint8")
            print("Dimensions : ",H.size);
            cv2.imwrite("/Users/ozercevikaslan/Desktop/MachineLearning/car_assets/output_%s.JPG" %filename,hogImage)

model = KNeighborsClassifier()
model.fit(top_features, labels)
modelname = 'brand_detection.sav'
joblib.dump(model, modelname)
predictedImage = cv2.imread("/Users/ozercevikaslan/Desktop/test.jpg", 0)
outcome = model.predict(X=predictedImage)

'''
