#Speed detection module

import cv2
import dlib
import math


def main():

    car_classifier = cv2.CascadeClassifier("haarcascade_car.xml") #loading the pretrained classifier
    video = cv2.VideoCapture("cars2.mp4")  #loading the video

    frameWidth = 1280
    frameHeight= 720

    fc = 0
    currentCarID = 0
    

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    # Loop once video is successfully loaded
    while video.isOpened():

        # Read first frame
        ret, frame = video.read()

        # resize image
        frame = cv2.resize(frame, (frameWidth, frameHeight))
        outputFrame = frame.copy()
        
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #changing the video to gray scale

        
        fc = fc + 1

        discardedCars = []

# trackers after analysing the image
        for car in carTracker:
            trackingQuality = carTracker[car].update(grayFrame)

            if trackingQuality < 10:
                discardedCars.append(car)

# popping them out from our dictionary 
        for car in discardedCars:
            carTracker.pop(car, None)
            carLocation1.pop(car, None)
            carLocation2.pop(car, None)

#updating tracker every alternate frame
        if fc % 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_classifier.detectMultiScale(gray, 1.3, 13) #passing image into classifier for detecting all cars

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

# we are getting coordinates of the car 
                for carID in carTracker:
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if (
                        (t_x <= x_bar <= (t_x + t_w))
                        and (t_y <= y_bar <= (t_y + t_h))
                        and (x <= t_x_bar <= (x + w))
                        and (y <= t_y_bar <= (y + h))
                    ):
                        matchCarID = carID

# if new car create a tracker or else do nothing 
                if matchCarID is None:
                    

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(gray, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

# getting the new position of the car 
        for carID in carTracker:
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

# drawing the rectangle on location specified by tracker
            cv2.rectangle(
                outputFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 255), 2
            )

            carLocation2[carID] = [t_x, t_y, t_w, t_h] #new position of the car 

# for all the cars that i have a location for
        for i in carLocation1:

            [x1, y1, w1, h1] = carLocation1[i]
            [x2, y2, w2, h2] = carLocation2[i]

            carLocation1[i] = [x2, y2, w2, h2] #moves the frame of reference to calculate the distance moved in each frame

            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                if (speed[i] == None or speed[i] == 0) and y1 >= 265 and y1 <= 285:
                    speed[i] = calcSpeed([x1, y1], [x2, y2])
                    #speed[i] = 67 debug

                # if y1 > 275 and y1 < 285: within this range of image we display speed
                if speed[i] != None and y1 >= 160:
                    cv2.putText(
                        outputFrame,
                        str(int(speed[i])) + " km/hr",
                        (int(x1 + w1 / 2), int(y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                    )

                # print ('CarID ' + str(i) + ': speed is ' + str(speed[i]) + ' km/h.\n')

                else: #if not in that range we show just tracking
                    cv2.putText(
                        outputFrame,
                        "Tracking..",
                        (int(x1 + w1 / 2), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        cv2.imshow("result", outputFrame) #output frame or video
        print "Frame processed: "+str(fc)
      
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    cv2.destroyAllWindows() #destroys all windows

# module for calculating speed
def calcSpeed(loc1,loc2):
    dist = math.sqrt(math.pow(loc2[0] - loc1[0], 2) + math.pow(loc2[1] - loc1[1], 2))
    ppi = 3.6  #pixel per inch depends on the video camera resolution
    distKM = dist / ppi
    fps = 24
    speed = distKM * fps
    return speed


if __name__ == "__main__":
    main()
