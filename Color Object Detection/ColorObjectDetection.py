import imutils
import cv2

# Define red color range in HSV
redLower = (157, 93, 203)
redUpper = (179, 255, 255)

# Open webcam (0 or 1 depending on your system)
camera = cv2.VideoCapture(0)

while True:
    grabbed, frame = camera.read()
    if not grabbed:
        break

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Mask for red color
    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    center = None

    if len(cnts) > 0:
        # Find the largest contour
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > 10:
                # Draw circle + center
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                print(center, radius)

                # Movement logic
                if radius > 250:
                    print("Stop")
                else:
                    if center[0] < 150:
                        print("Left")
                    elif center[0] > 450:
                        print("Right")
                    elif radius < 250:
                        print("Front")
                    else:
                        print("Stop")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
