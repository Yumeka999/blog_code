import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_dhash(s_img_url):
    np_lena_img = cv2.imread(s_img_url, cv2.IMREAD_UNCHANGED)

    # 缩放为32x32图片
    np_lena_img_resize = cv2.resize(np_lena_img ,(8, 9), cv2.INTER_AREA)

    # RGB三通道转为灰度图
    np_lena_img_gray = cv2.cvtColor(np_lena_img_resize, cv2.COLOR_BGR2GRAY)

    #  得到查分矩阵
    np_lena_img_diff = np.int32(np_lena_img_gray[1:9,:]) - np.int32(np_lena_img_gray[0:8,:])
    np_lena_img_diff = np_lena_img_diff >=  0

    dhash_bi = ''.join(str(b) for b in 1 * np_lena_img_diff.flatten())
    dhash_hex = '{:0>{width}x}'.format(int(dhash_bi, 2), width=16)
    return dhash_bi, dhash_hex

def get_hanming(s_hash_a, s_hash_b):
    if len(s_hash_a) != len(s_hash_b):
        print("two hash dim is not same!")
        return 100000000

    n_hanmming = 0
    for i in range(len(s_hash_a)):
        if s_hash_a[i] != s_hash_b[i]:
            n_hanmming += 1
    return  n_hanmming


if __name__ == "__main__":
    s_img_url_a = "Lenna(origin).png"
    s_img_url_b = "Lenna(noise).png"
    s_img_url_c = "Barbara.png"
    dhash_bi_a, dhash_hex_a = get_dhash(s_img_url_a)
    print("Lenna(origin).png")
    print(dhash_bi_a, dhash_hex_a)
    print()

    dhash_bi_b, dhash_hex_b = get_dhash(s_img_url_b)
    print("Lenna(noise).png")
    print(dhash_bi_b, dhash_hex_b)
    print()

    dhash_bi_c, dhash_hex_c = get_dhash(s_img_url_c)
    print("Barbara.png")
    print(dhash_bi_c, dhash_hex_c)
    print()

    print("hanming Lenna(origin) vs Lenna(noise)")
    print(get_hanming(dhash_bi_a, dhash_bi_b) )
    print()

    print("hanming Lenna(origin) vs Barbara")
    print(get_hanming(dhash_bi_a, dhash_bi_c) )
    print()
