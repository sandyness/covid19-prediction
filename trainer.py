import math
import os
import csv

import torch
from model import  NeuralNet
from model import prep_dataloader
from utils import get_device, same_seed, plot_learning_curve, plot_pred

"""# Config"""
train_path = 'data/covid.train.csv'
test_path = 'data/covid.test.csv'
target_only = False  
device = get_device()                 
config = {
    'seed': 3333,
    'n_epochs': 3000,               
    'batch_size': 270,              
    'optimizer': 'SGD',              
    'optim_hparas': {                
        'lr': 0.001,                 
        'momentum': 0.9             
    },
    'early_stop': 200,               
    'save_path': 'models/model.pth'
}


"""# Load data"""

same_seed(config['seed'])
train_loader = prep_dataloader(train_path, 'train', config['batch_size'], target_only=target_only)
valid_loader = prep_dataloader(train_path, 'dev', config['batch_size'], target_only=target_only)
test_loader = prep_dataloader(test_path, 'test', config['batch_size'], target_only=target_only)


"""# Trainning"""

def train(tr_set, dv_set, model, config, device):
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
    
    n_epochs = config['n_epochs']
    min_mse = math.inf
    loss_record = {'train': [], 'dev': []}     
    early_stop_cnt = 0
    epoch = 0
    
    if not os.path.isdir('./models'):
        os.mkdir('./models')
    for epoch in range(n_epochs):
        model.train()                        
        for x, y in tr_set:                 
            optimizer.zero_grad()               
            x, y = x.to(device), y.to(device)   
            pred = model(x)                    
            mse_loss = model.cal_loss(pred, y)  
            mse_loss.backward()                
            optimizer.step()                    
            loss_record['train'].append(mse_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

"""# Validation"""

def dev(dv_set, model, device):
    model.eval()                               
    total_loss = 0
    for x, y in dv_set:                         
        x, y = x.to(device), y.to(device)      
        with torch.no_grad():                  
            pred = model(x)                     
            mse_loss = model.cal_loss(pred, y)  
        total_loss += mse_loss.detach().cpu().item() * len(x)  
    total_loss = total_loss / len(dv_set.dataset)              

    return total_loss

"""# Testing"""

def test(tt_set, model, device):
    model.eval()                                
    preds = []
    for x in tt_set:                           
        x = x.to(device)                       
        with torch.no_grad():                  
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()     
    return preds


"""# Save_pred"""

def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])           

"""# main"""
model = NeuralNet(train_loader.dataset.dim).to(device)
model_loss, model_loss_record = train(train_loader, valid_loader, model, config, device)
plot_learning_curve(model_loss_record, title='deep model')


del model
model = NeuralNet(train_loader.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
plot_pred(valid_loader, model, device)  


preds = test(test_loader, model, device)  
save_pred(preds, 'pred.csv')         
