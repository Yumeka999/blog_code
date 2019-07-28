import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


np_lena_img = cv2.imread("Lenna_(test_image).png", cv2.IMREAD_UNCHANGED)
np_lena_img_resize = cv2.resize(np_lena_img ,(8, 8), cv2.INTER_AREA)
print(np_lena_img_resize.shape)
for i in range(3):
    print(np_lena_img_resize[:,:,i])


plt.figure(1)
plt.imshow(np_lena_img_resize)

np_lena_img_gray = cv2.cvtColor(np_lena_img_resize, cv2.COLOR_BGR2GRAY)
print(np_lena_img_gray.shape)


plt.figure(2)
plt.imshow(np_lena_img_gray,cmap='gray')

for i in range(8):
    ls_row = [str(e) for e in np_lena_img_gray[i,:].tolist()]
    print("\t".join(ls_row))

a = np.mean(np_lena_img_gray)
print("a", np.mean(np_lena_img_gray))
for i in range(8):
    ls_row_hash = ["1" if e >=a else "0" for e in np_lena_img_gray[i,:].tolist() ]
    print("\t".join(ls_row_hash))



# plt.show()
