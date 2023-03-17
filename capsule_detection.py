# 利用图像技术实现胶囊质检
import cv2
import numpy as np
import os


# 气泡检测
def bub_detection(img_path, img_file, im, im_gray):
    # 进行高斯滤波，进行模糊处理
    im_blur = cv2.GaussianBlur(im_gray, (3, 3), 0)
    # canny边沿提取
    im_canny = cv2.Canny(im_blur, 60, 240)
    cv2.imshow("im_canny", im_canny)

    # 查找轮廓
    img, cnts, hie = cv2.findContours(
        im_canny,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE)

    # 过滤轮廓
    new_cnts = []
    for i in range(len(cnts)):
        area = cv2.contourArea(cnts[i])  # 计算轮廓面积
        cir_len = cv2.arcLength(cnts[i], True)  # 计算周长
        if area >= 10000 or cir_len > 900 or area < 5:
            continue

        if hie[0][i][3] != -1:  # 当前轮廓父节点存在
            new_cnts.append(cnts[i])

    im_cnt = cv2.drawContours(im, new_cnts, -1, (0, 0, 255), 2)
    cv2.imshow("im_cnt", im_cnt)

    if len(new_cnts) > 0:
        print("气泡:", img_file)
        new_path = os.path.join("capsules/bub", img_file)
        os.rename(img_path, new_path)  # 文件移动
        print("文件移动成功: %s ==> %s" % (img_path, new_path))
        return True
    else:
        print("非气泡", img_file)
        return False


# 空胶囊检测
def empty_detection(img_path, img_file, im, im_gray):
    # 模糊
    im_blur = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # 二值化
    t, im_bin = cv2.threshold(im_blur,
                              210, 255,
                              cv2.THRESH_BINARY)
    cv2.imshow("im_bin", im_bin)

    # 查找轮廓
    img, cnts, hie = cv2.findContours(im_bin,
                                      cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_NONE)
    # 轮廓过滤
    new_cnts = []  # 经过过滤后的轮廓数据
    for i in range(len(cnts)):
        cir_len = cv2.arcLength(cnts[i], True)  # 计算周长
        # print("cir_len:", cir_len)
        if cir_len > 1000:
            new_cnts.append(cnts[i])  # 大于1000的保留

    im_cnt = cv2.drawContours(im, new_cnts,
                              -1, (0, 0, 255), 2)
    cv2.imshow("im_cnt", im_cnt)

    if len(new_cnts) == 1:
        print("空胶囊:", img_file)
        new_path = os.path.join("capsules/empty", img_file)
        os.rename(img_path, new_path)  # 文件移动
        print("文件移动成功: %s ==> %s" % (img_path, new_path))
        return True
    else:
        print("非空胶囊", img_file)
        return False


if __name__ == "__main__":
    img_dir = "capsules"  # 数据所在目录
    img_files = os.listdir(img_dir)
    for img_file in img_files:  # 遍历目录
        img_path = os.path.join(img_dir, img_file)
        if os.path.isdir(img_path):  # 目录
            continue

        # 读取图像
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imshow("im", im)
        cv2.imshow("im_gray", im_gray)

        # 空胶囊判断
        is_empty = False
        is_empty = empty_detection(img_path,
                                   img_file,
                                   im,
                                   im_gray)

        # 判断气泡
        is_bub = False
        if not is_empty:
            is_bub = bub_detection(img_path, img_file,
                                   im, im_gray)

        cv2.waitKey()
        cv2.destroyAllWindows()
