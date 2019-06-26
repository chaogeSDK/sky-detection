import cv2
import os
import math
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy import spatial
from scipy.optimize import curve_fit


#加载原始图像
def load_image(image_file_path):

    if not os.path.exists(image_file_path):
        print("图像文件不存在！")
        #sys.exit()
    else:
        img = cv2.imread(image_file_path)
        if img is None:
            print('读取图像失败!')
            #sys.exit()
        else:
            return img

#提取图像天空区域
def extract_sky(src_image):

    height, width = src_image.shape[0:2]

    sky_border_optimal = extract_border_optimal(src_image)
    border_correct = correct_border_polynomial(sky_border_optimal,src_image)
    sky_exists = has_sky_region(sky_border_optimal, height / 30, height / 10, 5)

    if sky_exists == 0:
        print('没有检测到天空区域')
        #sys.exit()

    """
    if has_partial_sky_region(border_correct, width / 3):
        border_new = refine_border(border_correct, src_image)
        sky_mask = make_sky_mask(src_image, border_new,1)
        return sky_mask, sky_exists
        #sky_image = display_sky_region(src_image, sky_border_optimal)
    """


    sky_mask = make_sky_mask(src_image, border_correct, 1)

    return sky_mask, sky_exists

#检测图像天空区域
def detect(image_file_path, output_path):

    #加载图像
    src_image = load_image(image_file_path)
    src_image = cv2.pyrDown(src_image)
    #x, y = src_image.shape[0:2]
    #src_image = cv2.resize(src_image, (int(2*y/3),int(2*x/3)), cv2.INTER_CUBIC)

    #提取图像天空区域
    sky_mask,sky_exists = extract_sky(src_image)

    #制作掩码输出
    tic = time.time()
    height = src_image.shape[0]
    width = src_image.shape[1]

    """
    sky_image_full = np.zeros(src_image.shape, dtype= np.uint8)
    for row in range(height):
        for col in range(width):
            if sky_mask[row, col] != 0:
                sky_image_full[row, col, 0] = 0
                sky_image_full[row, col, 1] = 0
                sky_image_full[row, col, 2] = 255

    sky_image = cv2.addWeighted(src_image, 1, sky_image_full, 1, 0)
    """

    for row in range(height):
        for col in range(width):
            if sky_mask[row, col] != 0:
                src_image[row, col, 0] = 0
                src_image[row, col, 1] = 0
                src_image[row, col, 2] = 255

    cv2.imwrite(output_path, src_image)
    toc = time.time()
    print('display mask time: ',(toc - tic), 's')
    print('图像检测完毕!')

#检测图像天空区域--批量
def batch_detect(image_dir, output_dir):

    img_filelist = os.listdir(image_dir)

    print('开始批量提取天空区域')
    i = 1
    for img_file in img_filelist:
        src_img = load_image(image_dir + img_file)
        src_img = cv2.pyrDown(src_img)

        sky_mask,sky_exists = extract_sky(src_img)
        if sky_exists == 0:
            i += 1
            cv2.imwrite(output_dir+img_file, src_img)
            continue
        height = src_img.shape[0]
        width  = src_img.shape[1]

        #sky_image_full = np.zeros(src_img.shape,dtype= src_img.dtype)
        for row in range(height):
            for col in range(width):
                if sky_mask[row, col] != 0:
                    src_img[row, col, 0] = 0
                    src_img[row, col, 1] = 0
                    src_img[row, col, 2] = 255
        #sky_img = cv2.addWeighted(src_img, 1, sky_image_full, 1, 0)
        cv2.imwrite(output_dir+img_file, src_img)

        print('已提取完成第',i,'张')
        i += 1

    print('批量提取完毕')

#计算天空灭点
def compute_vanish(image_file_path):
    # 加载图像
    src_img = load_image(image_file_path)
    src_img = cv2.pyrDown(src_img)
    src_img = cv2.pyrDown(src_img)
    height, width = src_img.shape[0:2]

    # 计算天空边界线
    sky_border_optimal = extract_border_optimal(src_img)
    border_correct = correct_border_polynomial(sky_border_optimal, src_img)

    # 判断是否存在天空
    sky_exists = has_sky_region(border_correct, height / 30, height / 10, 5)
    if sky_exists == 0:
        #print('没有检测到天空区域')
        #cv2.imwrite(output_path, src_img)
        return 2*(src_img.shape[0]//3)-15

    # 计算天空消失点的高度，并画图
    vanish_h = refine_vanishpoint(border_correct, src_img)
    #cv2.circle(src_img, (src_img.shape[1]//2, vanish_h), 4, (0, 255, 0), 8)
    #cv2.imwrite(output_path, src_img)

    return 2*vanish_h

#计算天空灭点--批量
def batch_compute_vanish(image_dir, output_dir):

    vanishs = []
    img_filelist = sorted(os.listdir(image_dir))

    print('开始批量计算天空灭点')
    i = 1
    for img_file in img_filelist:
        #加载图像
        src_image = load_image(image_dir + img_file)
        src_img = cv2.pyrDown(src_image)
        height, width = src_img.shape[0:2]

        #计算天空边界线
        sky_border_optimal = extract_border_optimal(src_img)
        border_correct = correct_border_polynomial(sky_border_optimal, src_img)

        #判断是否存在天空
        sky_exists = has_sky_region(border_correct, height / 30, height / 10, 5)
        if sky_exists == 0:
            print('没有检测到天空区域')
            i += 1
            cv2.imwrite(output_dir + img_file, src_image)
            continue

        #计算天空消失点的高度，并画图
        vanish_h = refine_vanishpoint(border_correct, src_img)
        vanishs.append(2*vanish_h)
        cv2.circle(src_image, (src_image.shape[1]//2, 4*vanish_h), 4, (0, 255, 0), 8)
        cv2.imwrite(output_dir+img_file, src_image)

        print('已计算完成第',i,'张')
        i += 1

    print('批量计算完毕')
    return vanishs

#提取图像梯度信息
def extract_image_gradient(src_image):
    #转灰度图像
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    #Sobel算子提取图像梯度信息
    x_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, 3)
    y_gradient = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, 3)

    #计算梯度幅值
    gradient_image = np.hypot(x_gradient, y_gradient)
    ret, gradient_image = cv2.threshold(gradient_image, 40, 1000, cv2.THRESH_BINARY)
    #gradient_image = np.uint8(np.sqrt(np.multiply(x_gradient,x_gradient) + np.multiply(y_gradient,y_gradient)))

    return gradient_image

#利用能量函数优化计算计算天空边界线
def extract_border_optimal(src_image, thres_sky_min = 5, thres_sky_max = 600, thres_sky_search_step = 6):

    #提取梯度信息图
    gradient_info_map = extract_image_gradient(src_image)

    n = math.floor((thres_sky_max - thres_sky_min)/ thres_sky_search_step) + 1

    border_opt = None
    jn_max = 0

    for i in range(n + 1):
        t = thres_sky_min + (math.floor((thres_sky_max - thres_sky_min) / n) - 1) * i
        b_tmp = extract_border(gradient_info_map, t)
        jn = calculate_sky_energy(b_tmp, src_image)
        #print('threshold= ',t,'energy= ',jn)

        if jn > jn_max:
            jn_max = jn
            border_opt = b_tmp

    return border_opt

# 计算天空图像能量函数
def calculate_sky_energy(border, src_image):

    # 制作天空图像掩码和地面图像掩码
    sky_mask = make_sky_mask(src_image, border, 1)
    ground_mask = make_sky_mask(src_image, border, 0)

    # 扣取天空图像和地面图像
    sky_image_ma = np.ma.array(src_image, mask = cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2BGR))
    ground_image_ma = np.ma.array(src_image, mask = cv2.cvtColor(ground_mask, cv2.COLOR_GRAY2BGR))

    # 计算天空和地面图像协方差矩阵
    sky_image = sky_image_ma.compressed()
    ground_image = ground_image_ma.compressed()

    sky_image.shape = (sky_image.size//3, 3)
    ground_image.shape = (ground_image.size//3, 3)

    sky_covar, sky_mean = cv2.calcCovarMatrix(sky_image, mean=None, flags=cv2.COVAR_ROWS|cv2.COVAR_NORMAL|cv2.COVAR_SCALE)
    sky_retval, sky_eig_val, sky_eig_vec = cv2.eigen(sky_covar)

    ground_covar, ground_mean = cv2.calcCovarMatrix(ground_image, mean=None,flags=cv2.COVAR_ROWS|cv2.COVAR_NORMAL|cv2.COVAR_SCALE)
    ground_retval, ground_eig_val, ground_eig_vec = cv2.eigen(ground_covar)

    gamma = 2  # 论文原始参数

    sky_det = cv2.determinant(sky_covar)
    #sky_eig_det = cv2.determinant(sky_eig_vec)
    ground_det = cv2.determinant(ground_covar)
    #ground_eig_det = cv2.determinant(ground_eig_vec)

    sky_energy = 1 / ((gamma * sky_det + ground_det) + (gamma * sky_eig_val[0,0] + ground_eig_val[0,0]))

    return sky_energy

# 判断图像是否存在天空区域
def has_sky_region(border, thresh_1, thresh_2, thresh_3):

    border_mean = np.average(border)

    #求天际线位置差，取绝对值，取均值
    border_diff_mean = np.average(np.absolute(np.diff(border)))

    sky_exists = 0
    if border_mean < thresh_1 or (border_diff_mean > thresh_3 and border_mean < thresh_2):
        return sky_exists
    else:
        sky_exists = 1
        return sky_exists

#判断图像是否有部分区域为天空区域
def has_partial_sky_region(border, thresh_4):

    border_diff = np.diff(border)

    '''
    if np.any(border_diff > thresh_4):
        index = np.argmax(border_diff)
        print(border_diff[index])
    '''

    return np.any(border_diff > thresh_4)

#计算天空边界线
def extract_border(gradient_info_map, thresh):

    height, width = gradient_info_map.shape[0:2]
    border = np.full(width, height - 1)

    for col in range(width):
        #返回该列第一个大于阈值的元素的索引
        border_pos = np.argmax(gradient_info_map[:, col] > thresh)
        if border_pos > 0:
            border[col] = border_pos

    return border

#天空区域和原始图像融合图,显示天空区域
def display_sky_region(src_image, border):

    height = src_image.shape[0]
    width = src_image.shape[1]

    #制作天空图掩码
    sky_mask = make_sky_mask(src_image, border, 1)

    #天空和原始图像融合
    sky_image_full = np.zeros(src_image.shape, dtype = src_image.dtype)
    for row in range(height):
        for col in range(width):
            if sky_mask[row, col] != 0:
                src_image[row, col, 0] = 0
                src_image[row, col, 1] = 0
                src_image[row, col, 2] = 255
    sky_image = cv2.addWeighted(src_image, 1, sky_image_full, 1, 0)

    return sky_image

#制作天空掩码图像,type: 1: 天空 0: 地面
def make_sky_mask(src_image, border, type):

    height = src_image.shape[0]
    width = src_image.shape[1]

    mask = np.zeros((height,width),dtype= np.uint8)

    if type == 1:
        for col, row in enumerate(border):
            mask[0:row +1, col] = 255
    elif type == 0:
        for col, row in enumerate(border):
            mask[row + 1:, col] = 255
    else:
        assert type is 0 or type is 1,'type参数必须为0或1'

    return mask

#改善天空边界线
def refine_border(border, src_image):

    sky_covar, sky_mean, ic_s, ground_covar, ground_mean, ic_g = true_sky(border, src_image)

    for col in range(src_image.shape[1]):
        cnt = np.sum(np.greater(spatial.distance.cdist(src_image[0:border[col], col], sky_mean, 'mahalanobis', VI=ic_s), spatial.distance.cdist(src_image[0:border[col], col], ground_mean, 'mahalanobis', VI=ic_g)))

        if cnt < (border[col] / 2):
            border[col] = 0

    return border

#改善天空边界线————alpha版本
def refine_border_alpha(border, src_image):

    sky_covar, sky_mean, ic_s, ground_covar, ground_mean, ic_g = true_sky(border, src_image)

    for col in range(src_image.shape[1]):
        for row in range(src_image.shape[0]):
            mahalanobis_sky = spatial.distance.cdist(src_image[row, col].reshape(1, 3), sky_mean, 'mahalanobis',VI=ic_s)
            mahalanobis_gr = spatial.distance.cdist(src_image[row, col].reshape(1, 3), ground_mean, 'mahalanobis',VI=ic_g)
            delta1 = abs(src_image[row, col, 0] - sky_mean[0,0]) < sky_mean[0,0] / 3.6
            delta2 = abs(src_image[row, col, 1] - sky_mean[0,1]) < sky_mean[0,1] / 3.6
            delta3 = abs(src_image[row, col, 2] - sky_mean[0,2]) < sky_mean[0,2] / 3.6
            if mahalanobis_sky < mahalanobis_gr and delta1 and delta2 and delta3:
                border[col] = row

    """
    sky_mean = np.mean(sky_image_true, axis= 0)
    for col in range(width):
        for row in range(height):
            delta1 = abs(src_image[row,col,0] - sky_mean[0]) < sky_mean[0]/3.6
            delta2 = abs(src_image[row,col,1] - sky_mean[1]) < sky_mean[1]/3.6
            delta3 = abs(src_image[row,col,2] - sky_mean[2]) < sky_mean[2]/3.6
            if delta1 and delta2 and delta3:
                border[col] = row
    """
    return border

#获取更真实天空像素和地面像素的均值、协方差及其逆
def true_sky(border, src_image):

    #制作天空图像掩码和地面图像掩码
    sky_mask = make_sky_mask(src_image, border, 1)
    ground_mask = make_sky_mask(src_image, border, 0)

    #扣取天空图像和地面图像
    sky_image_ma = np.ma.array(src_image, mask = cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2BGR))
    ground_image_ma = np.ma.array(src_image, mask = cv2.cvtColor(ground_mask, cv2.COLOR_GRAY2BGR))

    #将天空和地面区域shape转换为n*3
    sky_image = sky_image_ma.compressed()
    ground_image = ground_image_ma.compressed()

    sky_image.shape = (sky_image.size//3, 3)
    ground_image.shape = (ground_image.size//3, 3)

    # k均值聚类调整天空区域边界--2类
    sky_image_float = np.float32(sky_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(sky_image_float, 2, None, criteria, 10, flags)

    sky_label_0 = sky_image[labels.ravel() == 0]
    sky_label_1 = sky_image[labels.ravel() == 1]

    sky_covar_0, sky_mean_0 = cv2.calcCovarMatrix(sky_label_0, mean= None, flags= cv2.COVAR_ROWS + cv2.COVAR_NORMAL + cv2.COVAR_SCALE)
    sky_covar_1, sky_mean_1 = cv2.calcCovarMatrix(sky_label_1, mean= None, flags= cv2.COVAR_ROWS + cv2.COVAR_NORMAL + cv2.COVAR_SCALE)
    ground_covar, ground_mean = cv2.calcCovarMatrix(ground_image, mean= None, flags= cv2.COVAR_ROWS + cv2.COVAR_NORMAL + cv2.COVAR_SCALE)

    ic_s0 = cv2.invert(sky_covar_0, cv2.DECOMP_SVD)[1]
    ic_s1 = cv2.invert(sky_covar_1, cv2.DECOMP_SVD)[1]
    ic_g = cv2.invert(ground_covar, cv2.DECOMP_SVD)[1]

    #推断真实的天空区域
    if cv2.Mahalanobis(sky_mean_0, ground_mean, ic_s0) > cv2.Mahalanobis(sky_mean_1, ground_mean, ic_s1):
        sky_mean = sky_mean_0
        sky_covar = sky_covar_0
        ic_s = ic_s0
    else:
        sky_mean = sky_mean_1
        sky_covar = sky_covar_1
        ic_s = ic_s1


    return sky_covar,sky_mean,ic_s,ground_covar, ground_mean,ic_g

#修正天空灭点
def refine_vanishpoint(border,src_image):

    src_image = cv2.GaussianBlur(src_image, (7,7), 0)
    index = np.argmax(border)

    if border[index] >= 3*(src_image.shape[0]//4):
        dist = np.full(border[index], 0)
        width = src_image.shape[1]
        sky_covar,sky_mean,ic_s,ground_covar, ground_mean,ic_g = true_sky(border, src_image)
        for row in range(border[index]):
            distance = spatial.distance.cdist(src_image[width // 2, row].reshape(1, 3), sky_mean, 'mahalanobis',VI=ic_s)
            dist[row] = distance
        diff1 = np.diff(dist)
        diff2 = abs(np.diff(diff1))
        vanish_h = np.argmax(diff2)
    elif border[index] < src_image.shape[0]//2 :
        dist = np.full(src_image.shape[0], 0)
        width = src_image.shape[1]
        sky_covar,sky_mean,ic_s,ground_covar, ground_mean,ic_g = true_sky(border, src_image)
        for row in range(src_image.shape[0]):
            distance = spatial.distance.cdist(src_image[width//2, row].reshape(1, 3), sky_mean, 'mahalanobis', VI=ic_s)
            dist[row] = distance
        diff1 = np.diff(dist)
        diff2 = abs(np.diff(diff1))
        vanish_h = np.argmax(diff2)
    else:
        vanish_h = border[index]

    return vanish_h

#修正错误边界线--多项式拟合
def correct_border_polynomial(border, src_image):

    x = np.arange(0, src_image.shape[1], 1)
    border_line_argument = np.polyfit(x, border, 10)
    border_line_function = np.poly1d(border_line_argument)
    border_polynomial = np.int64(border_line_function(x))

    outlier = np.percentile(border,90)
    for col in range(len(border)):
        if border[col] >= outlier: # or abs(border[col]-border_polynomial[col]) > src_image.shape[0]/3 :
            border[col] = border_polynomial[col]
        #elif border[col] <= src_image.shape[0]//3:
            #border[col] = border_polynomial[col]

    return border

'''
#修正错误边界线--二次函数拟合
def correct_border_quardratic(border, src_image):
    outlier = np.percentile(border, 90)
    for col in range(len(border)):
        if border[col] >= outlier:
            if col == 0:
                border[col] = border[col + 1]
            elif col == src_image.shape[1] - 1:
                border[col] = border[col - 1]
            else:
                border[col] = (border[col - 1] + border[col + 1]) / 2
    x = np.arange(0, src_image.shape[1], 1)
    def fun(x,a,b,c):
        return a*(x**2) + b*x +c
    ppot,pcov = curve_fit(fun, x, border)
    a = ppot[0]
    b = ppot[1]
    c = ppot[2]
    border_new = np.int64(fun(x,a,b,c))

    return border_new
'''


if __name__ == '__main__':

    image_file_path = '/data/数据/35/'
    out_path = '/home/chaoshui/Pictures/天空检测/output_rec/'

    path = '/home/chaoshui/Pictures/天空检测/original/183477758211552024655154.jpg'
    out = '/home/chaoshui/Pictures/sky.jpg'

    tic = time.time()
    batch_compute_vanish(image_file_path, out_path)
    toc = time.time()
    times = 1000*(toc- tic)
    print('运行时间:',times,'ms')
