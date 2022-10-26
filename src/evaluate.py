import json
import torch

from create_dataset import create_dataloader
from train import Model
from fit import count_metrics

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # program ROOT

if __name__ == "__main__":
    test_csv_path =  root_dir = Path(ROOT, 'data', 'prepared', 'test.csv')
    test_dl = create_dataloader(test_csv_path)
    
    model = Model() # как правильно загрузить модель?
    model.load_state_dict(torch.load(Path(ROOT, 'models', 'model_best.pt')))
    
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    accur, f1_sc, _ = count_metrics(test_dl, device, model)
   
    metrics = {"accuracy": accur,
               "f1_score": f1_sc}
    
   
    metrics_path = Path(ROOT, 'metrics', 'los_auc_f1.json')
    metrics_path.write_text(json.dumps(metrics))