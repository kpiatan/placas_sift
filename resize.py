#!/usr/bin/python
from PIL import Image
import os, sys

path = "f:/Downloads/VisaoComputacional/SiftRansac/placas/"
pathout = "f:/Downloads/VisaoComputacional/SiftRansac/placasresize/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            basewidth = 800
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            imResize = img.resize((basewidth,hsize), Image.ANTIALIAS)
            imResize = imResize.rotate(270)
            imResize.save(pathout+item, 'JPEG', quality=90)

resize()