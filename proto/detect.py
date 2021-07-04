#!/usr/bin/python3

import sys, os
import re
import cv2 # opencv library
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt


def showImage(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Sample")
    plt.show()


def main(argv):
    samples = os.listdir('data/')
    
    img = cv2.imread('data/'+samples[0])
    showImage(img)



if __name__ == '__main__':
    main(sys.argv[1:])
