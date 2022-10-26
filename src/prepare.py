import os
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # program ROOT


def get_files_and_labels(list_root_dir):
    images_path = []
    labels = []
    
    list_classes = list(sorted(os.listdir(list_root_dir[0])))
    
    for root_dir in list_root_dir:
        for i, one_class in enumerate(list_classes):
            path_img = list(os.listdir(os.path.join(root_dir, one_class)))
            for path in path_img:
                path_join = os.path.join(root_dir, os.path.join(one_class, path))
                images_path.append(path_join)
                labels.append(i)
    
    df_res = pd.DataFrame()
    df_res['paths'] = images_path
    df_res['labels'] = labels
    
    return df_res


def train_test_split_(df_img_label):
    
    # size = len(df_img_label)
    # train = df_img_label[int(size * 0.0):int(size * 0.8)]
    # test = df_img_label[int(size * 0.8):int(size * 0.95)]
    # valid = df_img_label[int(size * 0.95):int(size * 1)]
    
    X = np.array(df_img_label['paths'])
    y = np.array(df_img_label['labels'])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.10, random_state=42)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                        test_size=0.20, random_state=42)
    
    name_columns = ['paths', 'labels']
    train = pd.DataFrame(zip(X_train, y_train), columns=name_columns)
    valid = pd.DataFrame(zip(X_valid, y_valid), columns=name_columns)
    test = pd.DataFrame(zip(X_test, y_test), columns=name_columns)
    
    return train, valid, test


def save_as_csv(train, valid, test):
    save_dir = Path(ROOT, 'data', 'prepared')
    print('----------')
    print(save_dir)
    train.to_csv(Path(save_dir, 'train.csv'), index=False)
    valid.to_csv(Path(save_dir, 'valid.csv'), index=False)
    test.to_csv(Path(save_dir, 'test.csv'), index=False)


if __name__ == '__main__':
    
    # list_root_dir = [Path(ROOT, 'data', 'original'), Path(ROOT, 'data', 'augmentetion')]
    list_root_dir = [Path(ROOT, 'data', 'original')]
    
    df_res = get_files_and_labels(list_root_dir)
    train, valid, test = train_test_split_(df_res)
    save_as_csv(train, valid, test)
        
    