import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import pickle

good = [1,2,3]

f = open('f:/Downloads/VisaoComputacional/SiftRansac/store.pckl', 'wb')
pickle.dump(good, f)
f.close()

f = open('f:/Downloads/VisaoComputacional/SiftRansac/store.pckl', 'rb')
a = pickle.load(f)
f.close()

print(a)