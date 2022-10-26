from joblib import load
import json
import torch

from train import create_dataloader
from fit import count_metrics



if __name__ == "__main__":
    test_csv_path = "../data/prepared/test.csv"
    test_dl = create_dataloader(test_csv_path)
    
    model = load("model/model.joblib")
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    accur, f1_sc, _ = count_metrics(test_dl, device, model)
   
    metrics = {"accuracy": accur,
               "f1_score": f1_sc}
    
   
    metrics_path = "metrics/los_auc_f1.json"
    metrics_path.write_text(json.dumps(metrics))