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


def augmentAruco(bbox, img, imgAug):
        
        # get four corner points of bounding box
        tl = bbox[0][0][0], bbox[0][0][1]
        tr = bbox[0][1][0], bbox[0][1][1]
        br = bbox[0][2][0], bbox[0][2][1]
        bl = bbox[0][3][0], bbox[0][3][1]

        # get size of image to augment
        h, w, c = imgAug.shape

        # warping image process
        pts1 = np.array([tl, tr, br, bl])
        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix, _ = cv2.findHomography(pts2, pts1)
        imgWarped = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
        
        # replacing aruco marker with black box
        cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
        

        return img



def main():
        camera = cv2.VideoCapture(0)
        augDics = loadAugImages("Markers")

        while True:
            _, img = camera.read()
            arucoFound = findArucoMarkers(img)

            
            # loop through all the markers and augment each one
            if len(arucoFound[0])!=0:
                    # loop through boundingBox and id at the same time
                    for boundingBox, id in zip(arucoFound[0], arucoFound[1]):
                            
                            # if id exist in Markers, augment image
                            if int(id) in augDics.keys():
                                img = augmentAruco(boundingBox, img, augDics[int(id)])


            cv2.imshow("Image", img)
            cv2.waitKey(1)




if __name__ == "__main__":
        main()
