import cv2

#load the haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
#load the haar Cascade Classifier for eye detection
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
#start webcam
cap = cv2.VideoCapture(0)

while True:
    #read the frame
    ret, frame = cap.read()
    if not ret:
        break

    #convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    faces=face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60,60)
    )

    #draw rectangles around detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,10) 

        for (ex,ey,ew,eh) in eyes:
            cv2.circle(
                roi_color,
                (ex+ew//2,ey+eh//2),
                min(ew,eh)//2,
                (255,0,0),
                2)

        #show output
    cv2.imshow("Face Detection",frame)

    #press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #release resources
cap.release()
cv2.destroyAllWindows()
      