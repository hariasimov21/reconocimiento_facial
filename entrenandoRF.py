import cv2
import os
import imutils
import numpy as np

dataPath = '/Users/jaimediaz/PycharmProjects/Req_Facial/Data'
peopletotalList = os.listdir(dataPath)
peopleList = peopletotalList.pop(0)
print('Lista de personas: ', peopletotalList)

labels = []
facesData = []
label = 0

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

for nameDir in peopletotalList:
    personPath = dataPath + '/' + nameDir
    print("leyendo las imagenes uwu")


    for fileName in listdir_nohidden(personPath):

        print("Rostros: ", nameDir + "/" + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + "/" + fileName, 0))
        #image = cv2.imread(personPath + "/" + fileName, 0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    label = label + 1

print("labels :", labels)
print("numero de etiquetas 0: ", np.count_nonzero(np.array(labels)== 0))
print("numero de etiquetas 1: ", np.count_nonzero(np.array(labels)== 1))

#face_recognicer = cv2.face.LBPHFaceRecognizer_create()
#face_recognicer = cv2.face.FisherFaceRecognizer_create()
face_recognicer = cv2.face.LBPHFaceRecognizer_create()

print("entrenando...")
face_recognicer.train(facesData, np.array(labels))

face_recognicer.write("modeloEigenFace.xml")
print("modelo almacenado...")