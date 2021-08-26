import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

FPS = cvzone.FPS()

listImg=os.listdir("photos")
print(listImg)
imgList = []
for imgPath in listImg:
    img= cv2.imread(f'photos/{imgPath}')
    imgList.append(img)
print(len(imgList))

indexImg= 0;


resized = cv2.resize(imgList[indexImg], (640, 480), interpolation=cv2.INTER_CUBIC)
imgList[indexImg] = resized


cap=cv2.VideoCapture(1)

cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS,60)
segmentor = SelfiSegmentation()
fpsReader= FPS


while True:
    success, img=cap.read()
    imgOut = segmentor.removeBG(img,imgList[indexImg], threshold=0.7)

    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    _, imgStacked=fpsReader.update(imgStacked,color=(0,0,225))
    print(indexImg)
    cv2.imshow("Image", imgStacked)

    key=cv2.waitKey(1)
    if key == ord('a'):
        indexImg -= 1
        resized = cv2.resize(imgList[indexImg], (640, 480), interpolation=cv2.INTER_CUBIC)
        imgList[indexImg] = resized

    elif key == ord('d'):
        indexImg += 1
        resized = cv2.resize(imgList[indexImg], (640, 480), interpolation=cv2.INTER_CUBIC)
        imgList[indexImg] = resized

    elif key == ord('q'):
        break