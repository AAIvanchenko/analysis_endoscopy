import cv2 as cv
import os


def rotate(img_dict, angle=90, scale=1.0):
    '''
       Rotate the image
       :param image: image to be processed
       :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation
       (the coordinate origin is assumed to be the top-left corner).
       :param scale: Isotropic scale factor.
     '''

    list_classes = list(img_dict.keys())
    new_img_dict = {list_classes[0]:[],
                list_classes[1]:[]}

    for key in list_classes:
        # count_img = 0
        for img in img_dict[key]:
            img_copy = img.copy()
            
            weight = img_copy.shape[1]
            height = img_copy.shape[0]
            #rotate matrix
            matrix = cv.getRotationMatrix2D((weight/2,height/2), angle, scale)
            #rotate
            img_copy = cv.warpAffine(img_copy,matrix,(weight,height))
            
            new_img_dict[key].append(img_copy)

    return new_img_dict

def save_img_augment(img_dict_augment):
    
    for key in img_dict_augment.keys():
        count_img = 0
        for img in img_dict_augment[key]:
            # print('../data/'+ key + '/' + str(count_img) +'_vflip.jpg')
            cv.imwrite('../data/features'+ key + '/' + str(count_img) +'_vflip.jpg', img)
            count_img += 1

if __name__ == '__main__':
    
    # считать все картинки
    root_dir = '../data/'
    list_classes = list(sorted(os.listdir(root_dir)))
    img_dict = {list_classes[0]:[],
                list_classes[1]:[]}
        
        
    for one_class in list_classes:
        path_img = list(os.listdir(os.path.join(root_dir, one_class)))
        for path in path_img:
            path_join = os.path.join(root_dir, os.path.join(one_class, path))
            img = cv.imread(path_join)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_dict[one_class].append(img)

    filter_dict = rotate(img_dict)
    save_img_augment(filter_dict)
