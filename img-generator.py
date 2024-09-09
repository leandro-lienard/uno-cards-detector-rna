import cv2
import numpy as np
import imutils


BILLETE = 100
image = cv2.imread(f"imagenes/train/{BILLETE}/billete-de-{BILLETE}-dolares-estados-unidos.png")

# achicamos con imutils
image_smalled = imutils.resize(image,width=300)
# imageOut_smalled2 = imutils.resize(image,height=300)


# Rotación
ancho = 300
alto = 300

# M = cv2.getRotationMatrix2D((ancho/2, alto/2), 35, 1)
# imageOut = cv2.warpAffine(image_smalled, M, (ancho,alto))

#Recorte
#imageOut = image[60:220,280:480]



for i in range(12):

    angulo = i * 30
    M = cv2.getRotationMatrix2D(((ancho/2, alto/2)), angulo, 1)
    image_rotada = cv2.warpAffine(image_smalled, M, (ancho,alto))

    # cv2.imshow("Imagen de salida", image_rotada)
    cv2.imwrite(f'imagenes/train/{BILLETE}/{BILLETE}_{i}.png', image_rotada)


cv2.imshow("Imagen de entrada", image_smalled)
cv2.imshow("Imagen de salida", image_rotada)
cv2.waitKey(0)
cv2.destroyAllWindows(0)
