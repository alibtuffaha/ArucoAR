import cv2
import cv2.aruco as aruco
import numpy as np
import os



def loadAugImages(path):
        myList = os.listdir(path)
        augDics = {}
        for imgPath in myList:
                key = int(os.path.splitext(imgPath)[0])
                imgAug = cv2.imread(f'{path}/{imgPath}')
                augDics[key] = imgAug
        return augDics



def findArucoMarkers(img, draw=True):
        # change current image to gray scale for detection
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # define which dictionary to get for markers,
        arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        
        # create parameters for detection
        arucoParam = aruco.DetectorParameters_create()

        # detect markers in img
        boundingBoxes, ids, rejectedMarkers = aruco.detectMarkers(imgGray,
                                                         arucoDict,
                                                         parameters=arucoParam)

        if draw:
                aruco.drawDetectedMarkers(img, boundingBoxes)
                print(ids)


        return [boundingBoxes, ids]





def main():
        camera = cv2.VideoCapture(0)
        augDics = loadAugImages("Markers")

        while True:
            _, img = camera.read()
            arucoFound = findArucoMarkers(img)


            cv2.imshow("Image", img)
            cv2.waitKey(1)




if __name__ == "__main__":
        main()
