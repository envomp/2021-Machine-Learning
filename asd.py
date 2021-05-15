import os
import numpy as np
import cv2
import imutils
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO

coco_dataset_path = "joon_resized/"
coco = COCO("coco.json")


def simplest_cb(img, percent):
    out_channels = []
    channels = cv2.split(img)
    totalstop = channels[0].shape[0] * channels[0].shape[1] * percent / 200.0
    for channel in channels:
        bc = cv2.calcHist([channel], [0], None, [256], (0,256), accumulate=False)
        lv = np.searchsorted(np.cumsum(bc), totalstop)
        hv = 255-np.searchsorted(np.cumsum(bc[::-1]), totalstop)
        lut = np.array([0 if i < lv else (255 if i > hv else round(float(i-lv)/float(hv-lv)*255)) for i in np.arange(0, 256)], dtype="uint8")
        out_channels.append(cv2.LUT(channel, lut))
    return cv2.merge(out_channels)

    # result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # avg_a = np.average(result[:, :, 1])
    # avg_b = np.average(result[:, :, 2])
    # result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    # result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    # result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    # return result


def preprocessImage(image_path):
    _img = cv2.imread(image_path)
    img = np.copy(_img)  # save original

    white_balance = simplest_cb(img, 5)

    # cv2.imshow('jooned', white_balance)
    # cv2.waitKey()
    #
    # color_area = cv2.cvtColor(white_balance, cv2.COLOR_BGR2HSV)[:,:,0]
    #
    # kernel_size = 25
    # blurred_color_area = cv2.GaussianBlur(color_area, (kernel_size, kernel_size), 0)
    #
    # cv2.imshow('jooned', blurred_color_area)
    # cv2.waitKey()

    kernel_size = 5
    blurred = cv2.GaussianBlur(white_balance, (kernel_size, kernel_size), 0)

    # cv2.imshow('jooned', blurred)
    # cv2.waitKey()
    #
    # converted = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) ## maybe
    #
    # cv2.imshow('jooned', converted)
    # cv2.waitKey()
    #
    # low_threshold = 20
    # high_threshold = 150
    # edges = cv2.Canny(converted, low_threshold, high_threshold)
    #
    # rho = 1  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 50  # minimum number of pixels making up a line
    # max_line_gap = 20  # maximum gap in pixels between connectable line segments
    # # line_image = np.copy(img) * 0  # creating a blank to draw lines on
    #
    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    #
    # if lines is not None:
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    #
    # cv2.imshow('jooned', img)
    # cv2.waitKey()

    return blurred


def drawImageWithSegments(image_path, segs):
    image = cv2.imread(image_path)
    segs = [np.array(seg, np.int32).reshape((1, -1, 2)) for seg in segs]
    for seg in segs:
        cv2.drawContours(image, seg, -1, (0, 255, 0), 2)
    cv2.imshow("polygan label", image)
    cv2.waitKey()


ann_ids = coco.getAnnIds()
anns = coco.loadAnns(ann_ids)

print("num of pictures:", coco.getImgIds())
print("num of annotations:", len(anns))


def getSegmentsForImage(image_id):
    image_segs = []
    for i, ann in enumerate(coco.loadAnns(ids=coco.getAnnIds(imgIds=[image_id]))):
        segs = ann["segmentation"]
        image_segs.append(segs[0])
    return image_segs


for image_id in coco.getImgIds():
    image = coco.loadImgs(ids=[image_id])[0]
    image_path = os.path.join(coco_dataset_path, image["file_name"])
    image_segs = getSegmentsForImage(image_id)

    processed_image = preprocessImage(image_path)
    # drawImageWithSegments(image_path, image_segs)

    # drawImageWithContours(image_path)
