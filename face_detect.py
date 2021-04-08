import cv2
import sys
# this program applies basic object detection using OpenCV
# requires Python, the OpenCV library, and a video capture device to work
# openCV docs: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

cascPath = "haarcascade_frontalface_alt.xml" # path to the pretaught model for frontal-faces
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0) # determines path to video capture device

# runs forever - as long as the program doesn't end
while True:
    # capture the source frame-by-frame
    ret, frame = video_capture.read()

    # casts a filter onto the frame so the computer can actually work with it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects faces using whatever settings
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # displays the final frame onto the video frame
    cv2.imshow('Video', frame)

    # exits the program if you press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the final capture when everything is done
video_capture.release()
cv2.destroyAllWindows()
