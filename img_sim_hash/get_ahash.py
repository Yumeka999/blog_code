import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_ahash(s_img_url):
    np_lena_img = cv2.imread(s_img_url, cv2.IMREAD_UNCHANGED)

    # 缩放为8x8图片
    np_lena_img_resize = cv2.resize(np_lena_img ,(8, 8), cv2.INTER_AREA)

    # RGB三通道转为灰度图
    np_lena_img_gray = cv2.cvtColor(np_lena_img_resize, cv2.COLOR_BGR2GRAY)

    # 计算8x8灰度矩阵的均值
    a = np.mean(np_lena_img_gray)

    # 根据均值得到hash值
    diff = np_lena_img_gray > a
    ahash_bi = ''.join(str(b) for b in 1 * diff.flatten())
    ahash_hex = '{:0>{width}x}'.format(int(ahash_bi, 2), width=16)
    return ahash_bi, ahash_hex

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
    ahash_bi_a, ahash_hex_a = get_ahash(s_img_url_a)
    print("Lenna(origin).png")
    print(ahash_bi_a, ahash_hex_a)
    print()

    ahash_bi_b, ahash_hex_b = get_ahash(s_img_url_b)
    print("Lenna(noise).png")
    print(ahash_bi_b, ahash_hex_b)
    print()

    ahash_bi_c, ahash_hex_c = get_ahash(s_img_url_c)
    print("Barbara.png")
    print(ahash_bi_c, ahash_hex_c)
    print()

    print("hanming Lenna(origin) vs Lenna(noise)")
    print(get_hanming(ahash_bi_a, ahash_bi_b) )
    print()

    print("hanming Lenna(origin) vs Barbara")
    print(get_hanming(ahash_bi_a, ahash_bi_c) )
    print()
