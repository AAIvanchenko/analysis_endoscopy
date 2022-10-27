import torch
from tqdm.autonotebook import tqdm
import torch.optim as optim
from torch import nn
from torchmetrics.functional import f1_score
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] # program ROOT

def count_metrics(data_dl, device, model, count_loss = False):
    loss_func = nn.CrossEntropyLoss() if count_loss else None
    model.eval()
    loss_sum = 0
    correct = 0
    f1_sc = 0
    num = 0

    with torch.no_grad():
        for target in tqdm(data_dl):
            xb, yb = target['image'].to(device),\
                     target['target'].to(device)
            
            if count_loss:
                probs = model(xb)
            else:
                probs = model(xb.float())
            
            if count_loss:
                loss_sum += loss_func(probs, yb).item()

            _, preds = torch.max(probs, axis=-1)
            correct += (preds == yb).sum().item()
            num += len(xb)
            f1_sc = f1_score(probs, yb, average='weighted', num_classes=2)
     
    
    losses = (loss_sum / len(data_dl)) if count_loss else None
    
    accuracies = 100*correct / num
    
    # print("accuracies: ", accuracies)
    # print("loss: ", losses)
    # print("f1: ", f1_sc)
    
    return accuracies, f1_sc, losses


def fit(epochs, model, loss_func, opt, train_dl, valid_dl,device, lr_scale = 0.01):
    """
    Обучение модели.

    :param epochs: количество эпох.
    :param model: модель, для обучения
    .
    :param loss_func: функция потерь сети.
    :param opt: функция оптимизации.
    :param train_dl: обучающая выборка.
    :param valid_dl: валиационная выборка.
    :return: массивы с метриками качества обученной модели.
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / train_dl.batch_size), 1)  # accumulate loss before optimizing
    
    lf = lambda x: (1 - x / epochs) * (1.0 - lr_scale) + lr_scale # linear
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    
    train_losses = []
    val_losses = []
    val_accur = []
    val_f1 = []
    best_fitness = 0
    best_acur = 0
    for epoch in range(epochs):
        print("epoch ", epoch)
        model.train()
        loss_sum = 0
        last_opt_step = 0
        for idx, target in enumerate(tqdm(train_dl)):
            xb, yb = target['image'].to(device),\
                     target['target'].to(device)
            
            pred = model(xb)  # forward
            loss = loss_func(pred, yb)  # loss scaled by batch_size
            loss_sum += loss.item()
            loss.backward()

            if idx - last_opt_step >= accumulate:
                opt.step()
                opt.zero_grad()
                last_opt_step = idx

        print("train_loss: ", loss_sum / len(train_dl))
        train_losses.append(loss_sum / len(train_dl))
        
        scheduler.step()

        accuracies_val, f1_sc, losses_val = count_metrics(valid_dl, device,
                                                   model, True)
        print("valid loss: ", losses_val)
        val_losses.append(losses_val)

        print("valid accuracies: ", accuracies_val)
        val_accur.append(accuracies_val)
        
        print("valid f1: ", f1_sc)
        val_f1.append(f1_sc.item())
        
        if val_f1[-1] > best_fitness and accuracies_val > best_acur:
            best_fitness = val_f1[-1]
       
            torch.save(model.state_dict(), Path(ROOT, 'models', 'model_best.pt'))
        
    return train_losses, val_losses, val_accur, val_f1

