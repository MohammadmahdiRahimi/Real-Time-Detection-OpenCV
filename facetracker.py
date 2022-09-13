import cv2

facexml = cv2.CascadeClassifier('face.xml')
eyexml = cv2.CascadeClassifier('eye.xml')
smilexml = cv2.CascadeClassifier('smile.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facexml.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
        cv2.putText(frame, "x = %i y = %i" %(int(x+w/2),int(y+h/2)) , (int(x),int(y+h+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        crop = gray[y:y+h,x:x+w]
        eyes = eyexml.detectMultiScale(crop)
        for (x1,y1,w1,h1) in eyes:
            cv2.rectangle(frame, (x+x1,y+y1), (x+x1+w1,y+y1+h1), (0,255,0), 2)
        smile = smilexml.detectMultiScale(crop)
        for (x2,y2,w2,h2) in smile:
            cv2.rectangle(frame, (x+x2,y+y2), (x+x2+w2,y+y2+h2), (255,255,0), 2)

    cv2.imshow("my frame", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()













