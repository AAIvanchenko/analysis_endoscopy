import os
import pandas as pd


def get_files_and_labels(root_dir):
    images_path = []
    labels = []
    list_classes = list(sorted(os.listdir(root_dir)))
    
    
    for one_class in list_classes:
        path_img = list(os.listdir(os.path.join(root_dir, one_class)))
        for path in path_img:
            path_join = os.path.join(root_dir, os.path.join(one_class, path))
            images_path.append(path_join)
            labels.append(one_class)
    
    df_res = pd.DataFrame()
    df_res['paths'] = images_path
    df_res['labels'] = labels
    
    return df_res


def train_test_split(df_img_label):
    
    size = len(df_img_label)
    train = df_img_label[int(size * 0.0):int(size * 0.8)]
    test = df_img_label[int(size * 0.8):int(size * 0.95)]
    valid = df_img_label[int(size * 0.95):int(size * 1)]
    
    return train, valid, test



def save_as_csv(train, valid, test):
    train.to_csv('prepared/train.csv')
    valid.to_csv('prepared/valid.csv')
    test.to_csv('prepared/test.csv')


if __name__ == '__main__':
    
    root_dir = '../data/original'
    
    df_res = get_files_and_labels(root_dir)
    train, valid, test = train_test_split(df_res)
    save_as_csv(train, valid, test)
        
    