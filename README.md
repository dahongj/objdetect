# objdetect
Object Detection of Ball Objects in a Video

Problem:
We want to predict the trajectory of a ball object in a video using machine learning libraries from Python.
The program must detect and recognize a ball object in the video and record its location on the frame. From here
we want the program to use Kalman Filter to predict the trajectory of the ball flying through the air even if
it is not visible, such as when it is behind another object. 

main.py:
- Runs the COCO modelset and Detector on the inserted video

Detector.py:
- Creates Detector class
- Reads all the COCO classes and runs the modelset on the video
- Recognize the sports ball index of 37 from the dataset and creates bounding boxes on the object
- Runs kalman filter on visible sports ball object and obscured kalman filter on hidden sports ball objects
- Writes the image onto the video and downloads the video before closing the window.

kalmanfilter.py:
- Uses the opencv function for KalmanFilter()
- Takes into account the measurement and transition calculation in its respective matrix
- Visible objects calculate their predicted trajectory based on inputted x and y values of the centroids
- The centroids are the central values of both the x and y values of the bounding boxes
- Hidden trajectory is calculated by applying prediction matrix on the most recent point
