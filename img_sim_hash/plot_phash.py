import json
import numpy as np
import cv2
import matplotlib.pyplot as plt


np_lena_img = cv2.imread("Lenna(origin).png", cv2.IMREAD_UNCHANGED)
np_lena_img_resize = cv2.resize(np_lena_img ,(32, 32), cv2.INTER_AREA)
np_lena_img_gray = cv2.cvtColor(np_lena_img_resize, cv2.COLOR_BGR2GRAY)

plt.figure(1)
plt.imshow(np_lena_img_gray,cmap='gray')

np_lena_img_dct = cv2.dct(np_lena_img_gray.astype(np.float32))
np_leana_img_dct_show = np.uint8(np_lena_img_dct)

plt.figure(2) 
plt.imshow(np_leana_img_dct_show,cmap='gray')

np_leana_img_dct_low_freq = np_lena_img_dct[0:8, 0:8]
for i in range(8):
    ls_row = [str(np.round(e,2)) for e in np_leana_img_dct_low_freq[i,:].tolist()]
    print("\t".join(ls_row))

a = np.mean(np_leana_img_dct_low_freq)
print("a", a)
for i in range(8):
    ls_row_hash = ["1" if e >=a else "0" for e in np_leana_img_dct_low_freq[i,:].tolist() ]
    print("\t".join(ls_row_hash))



plt.show()
