import numpy as np
import cv2
from matplotlib import pyplot as plt

#Se cargan los modelos para detectar rostros y ojos. NOTA: Estos modelos ya fueron entrenados previamente, solo los estamos usando
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Revisa que este archivo está en tu carpeta
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') #Revisa que este archivo está en tu carpeta

#Se lee la imagen
img = cv2.imread('js2.jpg')

#Se convierte a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgToShow =  img[:,:,::-1] 

#Se muestra la imagen
plt.imshow(imgToShow)     
plt.show()
#Se utiliza el modelo detector de rostros que ya cargamos anteriormente
faces = face_cascade.detectMultiScale(gray, 1.3, 5)  #Indicamos los parámetros
#faces = face_cascade.detectMultiScale(gray, 1.01, 1)
for (x,y,w,h) in faces: #iterando en las coordenadas en donde se detectaron rostros con el modelo
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #Se dibujan los rectángulos en donde se encontraron rostros
    roi_gray = gray[y:y+h, x:x+w]  #region de interés de la imagen en gris
    roi_color = img[y:y+h, x:x+w]  #region de interés de la imagen a color
    #Se utiliza el detector de ojos que ya cargamos anteriormente
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes: #Iterando en las coordenadas de los ojos encontrados
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #Se dibujan rectángulos en donde se encontraron ojos

plt.imshow(img)     
plt.show()
a = np.array([1, 3, 4, 5, 6 ,7])
print("Arreglo a original: ", a)
#Se crea el arreglo b con un segmento de a
b = a[:3]
#Se modifica un elemento de b
b[0]=9
#Como los valores se pasaron como referencia, también se modifica a
print("Arreglo b: ", b)
print("Arreglo a modificado: ", a)