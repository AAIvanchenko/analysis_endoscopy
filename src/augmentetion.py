import cv2 as cv
import os
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # program ROOT

def rotate(img_dict, angle=90, scale=1.0):
    '''
       Rotate the image
       :param image: image to be processed
       :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation
       (the coordinate origin is assumed to be the top-left corner).
       :param scale: Isotropic scale factor.
     '''

    img_new = cv.convertScaleAbs(img, alpha=1.3, beta=10)
    img_new = cv.blur(img_new,(10,10))

    return img_new

def save_img_augment(img_dict_augment):
    
    for key in img_dict_augment.keys():
        count_img = 0
        for img in img_dict_augment[key]:
            # print('../data/augmentetion'+ key + '/' + str(count_img) +'_vflip.jpg')
            # Path(ROOT, 'data', 'augmentetion', key, str(count_img)
            cv.imwrite('../data/augmentetion/'+ key + '/' + str(count_img) +'_vflip.jpg', img)
            count_img += 1

if __name__ == '__main__':
    
    # считать все картинки
    # root_dir = Path(ROOT, 'data', 'original')
    root_dir = '../data/original'
    list_classes = list(sorted(os.listdir(root_dir)))
    img_dict = {list_classes[0]:[],
                list_classes[1]:[]}
        
        
    for one_class in list_classes:
        path_img = list(os.listdir(os.path.join(root_dir, one_class)))
        for path in path_img:
            path_join = os.path.join(root_dir, os.path.join(one_class, path))
            # f = open(path_join, "rb");
            # chunk = f.read()
            # chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            # img = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
            img = cv.imread(path_join)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_dict[one_class].append(rotate(img))

    save_img_augment(img_dict)
