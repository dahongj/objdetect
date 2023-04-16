import cv2
import numpy as np
import time
from kalmanfilter import KalmanFilter

def ballBox(bboxs, bboxIdx, classLabelIDs, classesList):
    res = []
    for i in range(0, len(bboxIdx)):
        #Determine the bbox
        classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
        print(classLabelID)
        classLabel = classesList[classLabelID]
        bbox = bboxs[np.squeeze(bboxIdx[i])]
        if classLabelID == 37:
            res.append(bbox)
                        
    return res

#Setup video detector class
class Detector:
    def __init__(self, video, config, model, classes):
        self.video = video
        self.config = config
        self.model = model
        self.classes = classes
        
        self.net = cv2.dnn_DetectionModel(self.model, config)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    #Read in coco objects
    def readClasses(self):
        with open(self.classes, 'r') as f:
            self.classesList = f.read().splitlines()
        
        self.classesList.insert(0, 'temp')
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))

    def onVideo(self):
        #Attempt opening file
        cap = cv2.VideoCapture(self.video)

        #If file cannot be opened give error
        if(cap.isOpened()==False):
            print("Error opening file")
            return
        
        (ret,image) = cap.read()

        #Get video size
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        #Determine write file name
        filename = ""
        if(self.video == "videos/ball.mp4"):
            filename = "singleball.avi"
        elif(self.video == "videos/objectTracking_examples_multiObject.avi"):
            filename = "multiball.avi"

        res = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),10,size)

        #Initialize kalman filter
        kf = KalmanFilter()

        while ret:
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.2)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.2, nms_threshold = 0.2)

            x, y, a, b, cx, cy = 0, 0, 0, 0, 0, 0
            #Determine the existence of bounding boxes
            if len(bboxIdx) != 0:
  
                    bbox = ballBox(bboxs, bboxIdx, classLabelIDs, self.classesList)
                    displayText = "{}".format("sports ball", thickness = 1)
                    #If there exists a sports ball bbox apply the filter prediction
                    if(bbox):
                        for i in bbox:
                            x,y,a,b = i
                            cx,cy = int((2*x + a) / 2), int((2*y + b) / 2)
                            predicted = kf.visible_predict(cx, cy)
                
                            cv2.rectangle(image, (x,y), (x+a,y+b), (255,255,0), thickness=2)
                            cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 2)
                            cv2.circle(image, (predicted[0], predicted[1]), 5, (0, 0, 0), 4)
                    else:
                        #Apply the occlusion filter prediction
                        predicted = kf.hidden_predict()
                        cv2.circle(image, (predicted[0], predicted[1]), 5, (255, 0, 0), 4)

            #Write the bounding box onto the video
            res.write(image)
            cv2.imshow("Result",image)

            #Quit video if q keyis pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (ret, image) = cap.read()

        cap.release()
        res.release()
        cv2.destroyAllWindows()