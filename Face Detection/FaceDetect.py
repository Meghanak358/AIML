import cv2

# Load cascade from OpenCV’s built-in haarcascades folder
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        break  # safety check if webcam fails
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', im)

    # Exit on ESC
    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
