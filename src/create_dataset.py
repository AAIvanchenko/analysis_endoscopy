import pandas as pd
from torchvision import transforms
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CreateDataset(Dataset):
    """
    Класс загрузки dataset

    :list_classes: список классов.
    :img_path_list: список путей до изображений.
    :transform: список преобразовай dataset.
    :img_list: список изображений.
    """

    def __init__(self, data_frame, transform: transforms.Compose = None):
        
        self.list_classes = data_frame['labels'].to_list()
        self.img_path_list = data_frame['paths'].to_list()
        self.transform = transform
        self.img_list = []

        for path in self.img_path_list:
            img = self.__get_img_by_path(path)
            self.img_list.append(img)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        sample = {'image': self.img_list[index],
                  'target':  self.list_classes[index]}
    
        if self.transform:
            sample["image"] = self.transform(self.img_list[index])

        return sample

    @staticmethod
    def __get_img_by_path(img_path):
        """
        Получение картинки по её пути.
        :img_path: путь до картинки
        :return: картинка, состаящая из массива цифр
        """
        f = open(img_path, "rb");
        chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
        # img = cv.imread(img_path)
        # print(img_path)
        # print(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.array(img)
        return img

def create_dataloader(path_file_df):
    df = pd.read_csv(path_file_df)
    data = CreateDataset(df,transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(size=(256,256))]))
    
    data_dl = DataLoader(data, batch_size=4, shuffle=True)
    
    return data_dl
