import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
import os

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128,
           64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128,
           0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128,
           64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0]

input_path = "/home/stua/wh/CLReg_prog/cityscapes_val_final/tentwo3whnum234selfvits5b2rightddlight1ASPP_decoderRDDwhpaper256"
output_path = "./colorresult/PSAL_ASPP"


def trans(file):
    img = Image.open(os.path.join(input_path, file))
    img = np.asarray(img)
    out = Image.fromarray(img.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out.save(os.path.join(output_path, file))



if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    files = os.listdir(input_path)
    pool = multiprocessing.Pool(processes=16)
    pool.map(trans, files)
    # pool.map(gen_gt, range(100))
    pool.close()
    pool.join()
