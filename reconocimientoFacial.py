import cv2
import os
from imutils.video import FPS

dataPath = '/Users/jaimediaz/PycharmProjects/Req_Facial/Data'
imagePaths = os.listdir(dataPath)
imagePathDeleted = imagePaths.pop(0)

print('imagePaths=', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# leyendo el modelo

face_recognizer.read('modeloEigenFace.xml')

cap = cv2.VideoCapture('imagenes_prueba/joel.mp4')
#cap = cv2.VideoCapture('imagenes_prueba/jaime2.MOV')
#cap = cv2.VideoCapture('imagenes_prueba/dalas.mp4')


faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
fps = FPS().start()

while True:
    ret, frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        #EigenFace
        if result[1] < 70:
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    fps.update()

fps.stop()

print("FPS aproximado: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()
