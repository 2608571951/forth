from PIL import Image
from PIL import ImageEnhance
import numpy as np
from scipy.signal import convolve2d
import scipy.stats as stats
import math
import cv2
from sklearn import metrics
import os
IMAGE_PATH1 = '/home/root3203/lishanshan/TheWholeBrainAtlas/test1/MR-T1/'
IMAGE_PATH2 = '/home/root3203/lishanshan/TheWholeBrainAtlas/test1/MR-T2/'
IMAGE_PATH3 = './result_05/'
FILE_NAME = './test_result_index5.txt'

# # # # # # # # # # # # # # # # # # # #
# 计算psnr，数值越大表示失真越小
# # # # # # # # # # # # # # # # # # # #
def compute_psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return  100
    return 10 * math.log10(255.0**2/mse)




# # # # # # # # # # # # # # # # # # # #
# 计算ssim, 取值范围[-1, 1]
# 越接近1,代表相似度越高，融合质量越好
# # # # # # # # # # # # # # # # # # # #
def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(img1, img2, k1=0.01, k2=0.03, win_size=11, L=255.):
    if not img1.shape == img2.shape:
        print("Input images must have the same dimension...")
        raise ValueError("Input images must have the same dimension...")
    # M, N = img1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if img1.dtype == np.uint8:
        img1 = np.double(img1)
    if img2.dtype == np.uint8:
        img2 = np.double(img2)

    mu1 = filter2(img1, window, 'valid')
    mu2 = filter2(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = filter2(img1 * img2, window, 'valid') - mu1_mu2



    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(np.mean(ssim_map))



# # # # # # # # # # # # # # # # # # # #
# 计算熵，熵越高表示融合图像的信息量越丰富，
# 质量越好
# # # # # # # # # # # # # # # # # # # #
def compute_entropy(img):
    tmp = []
    for i in range(256):
        tmp.append(0)
    k = 0
    res = 0
    img = np.array(img)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[int(val)] = float(tmp[int(val)] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res





# # # # # # # # # # # # # # # # # # # #
# 计算均值和标准差，值越大表示灰度级分布越分散，
# 图像携带信息就越多，融合图像质量越好。
# 均值反应亮度信息，均值适中，质量越好。
# # # # # # # # # # # # # # # # # # # #
def compute_MEAN_STD(img):
    (mean, stddv) = cv2.meanStdDev(img)
    return mean, stddv




# # # # # # # # # # # # # # # # # # # #
# 计算标准化互信息NMI, 取值范围[0, 1]
# 越接近1,代表相似度越高，融合质量越好
# # # # # # # # # # # # # # # # # # # #
def compute_NMI(img1, img2):
    nmi = metrics.normalized_mutual_info_score(img1, img2)
    mi = metrics.mutual_info_score(img1, img2)
    return nmi, mi

############################################
# 计算融合图像的空间频率
##############################################
def compute_SF(img):
    rf = 0
    cf = 0
    for i in range(0, img.shape[0]):
        for j in range(1, img.shape[1]):
            a = img[i][j] - img[i][j-1]
            b = img[j][i] - img[j-1][i]
            rf += pow(a, 2)
            cf += pow(b, 2)
    c = rf + cf
    return pow(c, 0.5)


############################################
# 计算相关系数
############################################
def comput_CC(img1, img2):
    return stats.pearsonr(img1, img2)


def comp_idx():
    doc = open(FILE_NAME, 'w')
    SSIM1 = 0
    SSIM2 = 0
    NMI1 = 0
    NMI2 = 0
    CC1 = 0
    CC2 = 0
    EN = 0
    SF = 0

    for i in range(18):
        num = 3 * (i + 14)
        image1 = Image.open(IMAGE_PATH1 + str(num) + '.png').convert('L')
        image2 = Image.open(IMAGE_PATH2 + str(num) + '.png').convert('L')
        image3 = Image.open(IMAGE_PATH3 + str(num) + '.png').convert('L')
        image_norm1 = np.array(image1) / 255.0
        image_norm2 = np.array(image2) / 255.0
        image_norm3 = np.array(image3) / 255.0
        image_no1 = np.array(image1)
        image_no2 = np.array(image2)
        image_no3 = np.array(image3)

        psnr1 = compute_psnr(image_no1, image_no3)
        psnr2 = compute_psnr(image_no2, image_no3)

        ssim1 = compute_ssim(image_no1, image_no3)
        SSIM1 += ssim1
        ssim2 = compute_ssim(image_no2, image_no3)
        SSIM2 += ssim2

        image_no11 = image_no1.reshape([256 * 256])
        image_no22 = image_no2.reshape([256 * 256])
        image_no33 = image_no3.reshape([256 * 256])

        nmi1, mi1 = compute_NMI(image_no11, image_no33)
        NMI1 += nmi1
        nmi2, mi2 = compute_NMI(image_no22, image_no33)
        NMI2 += nmi2

        mean, stddv = compute_MEAN_STD(image_no3)
        entropy = compute_entropy(image3)
        EN += entropy
        sf = compute_SF(image_norm3)
        SF += sf

        cc1, _ = comput_CC(image_no11, image_no33)
        CC1 += cc1
        cc2, _ = comput_CC(image_no22, image_no33)
        CC2 += cc2


        print('\t\t\t\tPSNR\t\t\t\tSSIM\t\t\t\tNMI\t\t\t\t\t\t\tMI\t\t\t\tCC\t\t\t\t\tMEAN\t\t\tSTDDV\t\t\t\tentropy\t\t\t\tSF\nfusion_img1\t\t{0}\t{1}\t{2}\t{9}\t{12}\t{3}\t{4}\t{8}\t{11}\n'
              'fusion_img2\t\t{5}\t{6}\t{7}\t{10}\t{13}'.format(psnr1, ssim1, nmi1, mean, stddv, psnr2, ssim2, nmi2, entropy, mi1, mi2, sf, cc1, cc2), file=doc)
        print(
            '\t\t\t\tPSNR\t\t\t\tSSIM\t\t\t\tNMI\t\t\t\t\t\tMI\t\t\t\t\t\tCC\t\t\t\tMEAN\t\t\t\tSTDDV\t\t\t\tentropy\t\t\t\tsf\nfusion_img1\t\t{0}\t{1}\t{2}\t{9}\t{12}\t{3}\t{4}\t{8}\t{11}\n'
            'fusion_img2\t\t{5}\t{6}\t{7}\t{10}\t{13}'.format(psnr1, ssim1, nmi1, mean, stddv, psnr2, ssim2, nmi2, entropy, mi1, mi2, sf, cc1, cc2))
    SSIM1 /= 18
    SSIM2 /= 18
    NMI1 /= 18
    NMI2 /= 18
    CC1 /= 18
    CC2 /= 18
    EN /= 18
    SF /= 18

    print('SSIM1: {0}; SSIM2: {1}; NMI1: {2}; NMI2: {3}; CC1: {4}; CC2: {5}; EN: {6}; SF: {7}'.format(SSIM1, SSIM2, NMI1, NMI2, CC1, CC2, EN, SF))

if __name__ == '__main__':
    comp_idx()
