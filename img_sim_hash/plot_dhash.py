import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


np_lena_img = cv2.imread("Lenna(origin).png", cv2.IMREAD_UNCHANGED)
np_lena_img_resize = cv2.resize(np_lena_img ,(8, 9), cv2.INTER_AREA)
np_lena_img_gray = cv2.cvtColor(np_lena_img_resize, cv2.COLOR_BGR2GRAY)
for i in range(8):
    ls_row = [str(np.round(e,2)) for e in np_lena_img_gray[i,:].tolist()]
    print("\t".join(ls_row))
print()

plt.figure(1)
plt.imshow(np_lena_img_gray,cmap='gray')

np_lena_img_diff = np.int32(np_lena_img_gray[1:9,:]) - np.int32(np_lena_img_gray[0:8,:])


for i in range(8):
    ls_row = [str(np.round(e,2)) for e in np_lena_img_diff[i,:].tolist()]
    print("\t".join(ls_row))
print()

np_lena_img_bool = np_lena_img_diff >= 0
np_lena_img_bool = np_lena_img_bool * 1
for i in range(8):
    ls_row = [str(np.round(e,2)) for e in np_lena_img_bool[i,:].tolist()]
    print("\t".join(ls_row))
print()

print("".join(["1" if e ==1 else "0" for e in np_lena_img_bool.flatten().tolist()]))




plt.show()
