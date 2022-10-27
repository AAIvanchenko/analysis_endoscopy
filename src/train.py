import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import os
from create_dataset import create_dataloader
from fit import fit

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # program ROOT


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 4, 3)
        self.bn2 = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(15376, 500)
        self.fc2 = nn.Linear(500, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.bn1(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = self.bn2(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.bn4(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    
    train_dl = create_dataloader(Path(ROOT, 'data', 'prepared', 'train.csv'))
    valid_dl = create_dataloader(Path(ROOT, 'data', 'prepared', 'valid.csv'))
    
    model = Model().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
  
    train_losses, val_losses, val_accur, val_f1 = fit(10, model, loss_fn,
                                                      optimizer,
                                                      train_dl, valid_dl,device)
    
    # metrics = {"train_losses": train_losses,
    #            "val_losses": val_losses,
    #            "val_accur": val_accur,
    #            "val_f1": val_f1}
    
   
    # metrics_path = "metrics/train_val_los_auc_f1.json"
    # metrics_path.write_text(json.dumps(metrics))
    