import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from create_dataset import create_dataloader
from fit import fit

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # program ROOT

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    torch.cuda.empty_cache()
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
    