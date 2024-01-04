'''
from hashlib import md5
from turtle import down
from pyparsing import line
from sqlalchemy import true
'''
import math
import random
import numpy as np
import cv2
import os

from sympy import Polygon
'''
persPX2panoPX(pano_maps, positions) is used to convert coordinates from perspective to pano
'''
def persPX2panoPX(pano_maps, pix):
    #positions = np.array([pix[1], pix[0]])
    positions = np.array([256, 256]).astype(int)
    # for pne point
    if len(positions.shape)==1:
        p = np.stack([positions])
    else:
        p = positions.copy()
    
    p = np.swapaxes(p,0,1)
    x = pano_maps[0][p[0], p[1]]
    y = pano_maps[1][p[0], p[1]]

    if len(positions.shape) == 1:
        return np.stack([y, x], axis=-1)[0].astype(int)
    return np.stack([y, x], axis=-1).astype(int)
'''
inSameLine(pt1, pt2, line) is used to check if two points are in the same line
'''
def inSameLine(pt1, pt2, line):
    if pt1 in line and pt2 in line:
        return 1
    else:
        return 0
'''
getTheta2d(a, b) is used to calculate the angle between two vectors in 2D space
Method: Law of cosine
'''
def getTheta2d(a, b):
    origin = np.array([0, 0])
    costheta = np.dot(a, b) / (distance2d(a, origin) * distance2d(b, origin))
    rad = math.acos(costheta)
    return rad * 180 / math.pi
'''
panoPX2persPX(pano_maps, pano_pixel) is used to convert coordinates from pano to perspective
'''
def panoPX2persPX(pano_maps, pano_pixel):

    positions = np.array([pano_pixel[1], pano_pixel[0]])
    # print(positions)
    if len(positions.shape) == 1:
        persppoint = np.stack([positions])
    else:
        persppoint = positions.copy()

    p = np.swapaxes(persppoint,0,1).astype(np.int64)
    x_map, y_map, mask = pano_maps

    x = x_map[p[0], p[1]]
    y = y_map[p[0], p[1]]
    # print('x, y: ', x, y)     
    mask_index = np.where(mask[p[0], p[1]]==1)
    # print('mask_index: ', mask_index)
    points = np.stack([y, x], axis=-1)

    points = points[mask_index]
    if len(positions.shape) == 1:
        if len(points)==0:
            return []
        else:
            return points[0].astype(int)
    return points.astype(int)
def point2perspec(pano_maps, positions, with_mask_index=False):
    if len(positions.shape)==1:
        persppoint = np.stack([positions])
    else:
        persppoint = positions.copy()

    p = np.swapaxes(persppoint,0,1)
    # print(p)
    x_map, y_map, mask = pano_maps

    x = x_map[p[0], p[1]]
    y = y_map[p[0], p[1]]

    mask_index = np.where(mask[p[0], p[1]]==1)
    # print(mask_index)
    points = np.stack([y, x], axis=-1)

    if with_mask_index:
        return points, mask_index
    
    points = points[mask_index]
    if len(positions.shape) == 1:
        if len(points)==0:
            return []
        else:
            return points[0].astype(int)
    return points.astype(int)
def getPersMap(theta, phi, _img):
    FOV = 120
    width = 512
    height = 512
    wfov = 120
    hfov = 120
    pano_size = (1024, 2048)
    perspec_size = (512, 512)

    #fov對應crop的像素範圍
    w_len = np.tan(np.radians(wfov / 2.0))
    h_len = np.tan(np.radians(hfov / 2.0))

    x, y = np.meshgrid(
        np.linspace(-180, 180, pano_size[1]), np.linspace(90, -90, pano_size[0]))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map, y_map, z_map), axis=2) #shape:(1024, 2048, 3)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    #取得z軸的旋轉向量
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
    # TODO: Why? 為何要將z軸的旋轉向量投影在y軸上?
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)
    #變回單位向量坐標系
    xyz = xyz.reshape([pano_size[0] * pano_size[1], 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([pano_size[0], pano_size[1], 3])
    inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)
    #TODO: Mask是做甚麼用的?
    xyz[:, :] = xyz[:, :]/np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

    lon_map = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), (xyz[:, :, 1]+w_len)/2/w_len*perspec_size[1], 0)
    lat_map = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), (-xyz[:, :, 2]+h_len)/2/h_len*perspec_size[0], 0)
    mask = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), 1, 0)

    mask = mask * inverse_mask
    pano_maps = [lon_map.astype(np.int64), lat_map.astype(np.int64), mask]
    return pano_maps
'''
getPanoMap(theta, phi, _img) is used to get the map that used to convert coordinates 
'''
def getPanoMap(theta, phi, _img):
    FOV = 120
    width = 512
    height = 512
    wfov = 120
    hfov = 120
    pano_size = (1024, 2048)
    perspec_size = (512, 512)


    w_len = np.tan(np.radians(wfov / 2.0))
    h_len = np.tan(np.radians(hfov / 2.0))

    x, y = np.meshgrid(
        np.linspace(-180, 180, pano_size[1]), np.linspace(90, -90, pano_size[0]))

    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map, y_map, z_map), axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)

    [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([pano_size[0] * pano_size[1], 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([pano_size[0], pano_size[1], 3])
    inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

    xyz[:, :] = xyz[:, :]/np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

    lon_map = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), (xyz[:, :, 1]+w_len)/2/w_len*perspec_size[1], 0)
    lat_map = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), (-xyz[:, :, 2]+h_len)/2/h_len*perspec_size[0], 0)
    mask = np.where((-w_len < xyz[:, :, 1]) & (xyz[:, :, 1] < w_len) & (-h_len < xyz[:, :, 2])
                    & (xyz[:, :, 2] < h_len), 1, 0)

    mask = mask * inverse_mask
    pano_maps = [lon_map.astype(np.int64), lat_map.astype(np.int64), mask]
    return pano_maps

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out

class Equirectangular:
    def __init__(self, img):
        self._img = img
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # 注意BGR 寬高
        [self._height, self._width, _] = self._img.shape
        # cp = self._img.copy()
        # w = self._width
        # self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        # self._img[:, w/8:, :] = cp[:, :7*w/8, :]

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
        K_inv = np.linalg.inv(K)

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        z = np.ones_like(x)
        xyz = np.concatenate(
            [x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz)
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(
            self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

'''
getPixel(a, origin_angle, width) is used to convert the angle to pixel
'''
def getPixel(a, origin_angle, width):
    return a / origin_angle * width - 1

'''
distance2d(a, b) is used to calculate the distance between two points in 2D space
'''
def distance2d(a, b):
    d = 0.0
    d = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return d

'''
distance3d(a, b) is used to calculate the distance between two points in 3D space
'''
def distance3d(a:np.array, b:np.array):
    d = 0.0
    d = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    return d
'''
getTheta(a, b) is used to calculate the angle between two vectors in 3D space
Method: Law of cosine
'''
def getTheta(a:np.array, b:np.array):
    origin = np.array([0, 0, 0])
    costheta = np.dot(a, b) / (distance3d(a, origin) * distance3d(b, origin))
    # print('costheta: ', costheta)
    rad =  math.acos(costheta)
    return rad * 180 / math.pi
'''
wdo_vertices23D(wdo_vertices, wdo_bbox_3D_list) is used to convert origin data in zind_data to 3D door vertices
'''
def wdo_vertices23D(wdo_vertices):
    door_len = int(len(wdo_vertices)/3)
    wdo_bbox_3D_list=[]
    for wdo_idx in range(0, door_len):
        bottom_z = wdo_vertices[wdo_idx * 3 + 2][0]
        top_z = wdo_vertices[wdo_idx * 3 + 2][1]
        # wdo_bbox_3D contains four points at bottom left, bottom right, top right, top left
        wdo_bbox_3D = np.array([
                [-wdo_vertices[wdo_idx * 3][0], wdo_vertices[wdo_idx * 3][1], bottom_z,],
                [-wdo_vertices[wdo_idx * 3 + 1][0], wdo_vertices[wdo_idx * 3 + 1][1], bottom_z,],
                [-wdo_vertices[wdo_idx * 3 + 1][0], wdo_vertices[wdo_idx * 3 + 1][1], top_z,],
                [-wdo_vertices[wdo_idx * 3][0], wdo_vertices[wdo_idx * 3][1], top_z,],
            ])
        wdo_bbox_3D_list.append(wdo_bbox_3D)
    return wdo_bbox_3D_list

'''
wdo_3D2pixel(wdo_bbox_3D, wdo_bbox_pixel) is used to convert 3D door vertices to 2D door vertices
both wdo_bbox_3D and wdo_bbox_pixel are array that contains all door vertices
ex. wdo_bbox_3D_list = [
    [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]], -> door 1
    [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]], -> door 2
]
wdo_bbox_pixel_list should be a empty list when input
'''
def wdo_3D2pixel_list(wdo_bbox_3D_list, wdo_bbox_pixel_list):
    # print(type(wdo_bbox_3D_list))
    # door_len = len(wdo_bbox_3D_list)
    for wdo_bbox_3D in wdo_bbox_3D_list:
        wdo_bbox_pixel = []
        wdo_3D2pixel(wdo_bbox_3D, wdo_bbox_pixel)
        wdo_bbox_pixel_list.append(wdo_bbox_pixel)
'''
wdo_3D2pixel(wdo_bbox_3D, wdo_bbox_pixel) is used to convert door 3D coordinates to pixel coordinates
both wdo_bbox_3D and wdo_bbox_pixel are lists that store all the coordinates of a single door
ex. wdo_bbox_3D = [
    [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]
]
Note. when using this function, wdo_bbox_pixel should be initialized as an empty list
'''
# for pano only
def wdo_3D2pixel(wdo_bbox_3D, wdo_bbox_pixel):
    horizontal_start = np.array([0, -1, 0])
    vertical_start = np.array([0, 0, 1])
    # get door pixel coordinates in pano
    for point in wdo_bbox_3D:
        # print(point)
        tmp_horizontal = np.array([point[0], point[1], 0])        
        horizontal_theta = getTheta(tmp_horizontal, horizontal_start)
        if point[0] > 0:
            horizontal_theta *= -1
        vertical_theta = getTheta(point, vertical_start)
        horizontal_pixel = getPixel(horizontal_theta, 360, 2048)
        if point[0] > 0:
            horizontal_pixel += 2048
        vertical_pixel = getPixel(vertical_theta, 180, 1024)
        tmp_np = np.around(np.array([horizontal_pixel, vertical_pixel]).astype(np.float32), decimals=2)
        wdo_bbox_pixel.append(tmp_np.tolist())
from PIL import Image
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def fit_line_by_ransac(point_list, sigma, iters = 1000, P = 0.99):
    # 使用RANSAC算法拟合直线
    # 迭代最大次数 iters = 1000
    # 数据和模型之间可接受的差值 sigma
    # 希望的得到正确模型的概率P = 0.99
    if len(point_list) <= 2:
        return 0, 0
    # 最好模型的参数估计
    best_a = 0#直线斜率
    best_b = 0#直线截距
    n_total = 0#内点数目
    for i in range(iters):
        # 随机选两个点去求解模型
        sample_index = random.sample(range(len(point_list)), 2)
        # print(sample_index, len(point_list))
        # print(point_list[sample_index[0]])
        x_1 = point_list[sample_index[0]][0]
        y_1 = point_list[sample_index[0]][1]
        x_2 = point_list[sample_index[1]][0]
        y_2 = point_list[sample_index[1]][1]
        if x_2 == x_1:
            continue
            
        # y = ax + b 求解出a，b
        a = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - a * x_1

        # 算出内点数目
        total_inlier = 0
        for index in range(len(point_list)):
            y_estimate = a * point_list[index][0] + b
            if abs(y_estimate - point_list[index][1]) < sigma:
                total_inlier += 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > n_total:
            # if 1 - P <= 0 or 1 - pow(total_inlier/len(point_list), 2) <= 0:
            #     print(P, pow(total_inlier/len(point_list), 2))
            iters = math.log(1 - P) / math.log(1 - pow((total_inlier-1)/len(point_list), 2))
            n_total = total_inlier
            best_a = a
            best_b = b

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > len(point_list)//2:
            break
    # print("iter: ", iters)
    return best_a, best_b