import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model

model=load_model("myModel.h5")

file=input("enter the path to the image to be predited : ")
img=cv2.imread(file)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img=cv2.resize(img,(28,28),cv2.INTER_AREA)

plt.imshow(img)
img=np.reshape(img,[1,28,28,1])
print(model.predict_classes(img))
plt.show()
