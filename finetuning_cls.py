import numpy as np
import pandas as pd
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from sklearn.metrics import *
import copy
import pickle

from transformers import AutoModel, AutoTokenizer

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

def set_seed(seed: int = 42) :
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def score(df):
    label = df['label']
    
    try:
        aupr = average_precision_score(label, df['pred'])
        auc = roc_auc_score(label, df['pred'])
        
    except ValueError as e:
        print(f"Metric calcuation error - {e}")
        return 0.0, 0.0

    return aupr, auc

def predict(model, dataloader, device):
    model.eval()
    result = []  
    label = []   

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            output = model(inputs)
            preds = torch.sigmoid(output)

            result.append(preds)
            label.append(target)

    result = torch.cat(result).cpu().detach().numpy().squeeze()
    label = torch.cat(label).cpu().detach().numpy().squeeze()

    df = pd.DataFrame()
    df['pred'] = result
    df['label'] = label

    return df

def train_model(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device):
    best_score = -100
    best_epoch = 0
    best_model = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0

        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.squeeze(1), labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in valid_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                logits = model(inputs)
                loss = criterion(logits.squeeze(1), labels)
                
                val_loss += loss.item()
                
                preds = torch.sigmoid(logits)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
            val_aupr = average_precision_score(all_labels, all_preds)
            
        except ValueError:
            val_auc = 0.0
            val_aupr = 0.0
        
        valid_score = val_auc + val_aupr

        if valid_score > best_score:
            best_epoch = epoch
            best_score = valid_score
            best_model = copy.deepcopy(model)
            
    if best_model is None:
        return model 
    print(f'Best epoch : {best_epoch}')
    return best_model

def prepare_input(tokenizer, smiles):
    inputs = tokenizer(smiles, add_special_tokens=True, truncation=True, max_length=131, padding="max_length")
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class ChemiDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=False, label = 'label'):
        self.is_train = is_train
        self.smiles = df['smiles'].values
        self.tokenizer = tokenizer
        self.inputs = [prepare_input(tokenizer, smi) for smi in df['smiles'].values]
        if not self.is_train:
            self.label = df['label'].values
            
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, item):
        inputs = self.inputs[item] 
        label = torch.tensor(self.label[item], dtype=torch.float)
        return inputs, label  
    
    
def get_dataloader(df, label_col, tokenizer, batch_size=64, shuffle=True, num_workers=0, drop_last = False):
    dataset = ChemiDataset(df, tokenizer, label = label_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last)
    return dataloader

class DrugEncoder(nn.Module):
    def __init__(self, pretrained_model='DeepChem/ChemBERTa-77M-MTR'):
        super(DrugEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
               
    def forward(self, inputs):
        embedding = self.bert(**inputs)
        embedding = embedding[0]
        hidden = embedding[:, 0, :]
        return hidden

class DrugClassifier(nn.Module):
    def __init__(self, drug_dim=384, hid_dim=384, proj_dim=128, 
                 mlp_hid_dim1=32, mlp_hid_dim2=8, output_dim=1):
        super().__init__()
        
        self.drug_encoder = DrugEncoder()
        
        self.mlp_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hid_dim, output_dim)
        )
            
    def load_pretrained_weights(self, state_dict_path, device):
        try:
            pretrained_dict = torch.load(state_dict_path, map_location=device)
            model_dict = self.state_dict()
            
            filtered_dict = {
                k: v for k, v in pretrained_dict.items() 
                if k in model_dict and v.size() == model_dict[k].size()
            }
            
            if not filtered_dict:
                print("Error occured, Check the model files")
            else:
                model_dict.update(filtered_dict)
                self.load_state_dict(model_dict)
                print("Successfully load the model files")
                
        except Exception as e:
            print(f"Error occured: {e}")
            
    def forward(self, x_drug):
        z_drug_feat = self.drug_encoder(x_drug)
        logits = self.mlp_head(z_drug_feat)
        return logits

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--task_name', type=str, default='dili')
    p.add_argument('--col_name', type=str, default='label')
    
    config = p.parse_args()

    return config

def main_finetuning(config):

    num_seed = config.random_seed
    gpu_num = config.gpu_id
    batch_size = config.batch_size
    lr = config.lr
    data_dir_name = config.task_name
    n_epochs = config.n_epochs
    col_name = config.col_name
    
    set_seed(num_seed)    
    device = torch.device("cuda:" + str(gpu_num) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = f'./finetuning_data/Classification/{data_dir_name}/'
    pretrained_path = './pre_trained_weights/pt_weights.pth'    
    
    try:
        train_df = pd.read_csv(data_path + 'train.csv')
        valid_df = pd.read_csv(data_path + 'valid.csv')
        test_df = pd.read_csv(data_path + 'test.csv')

    except FileNotFoundError as e:
        print(f"Cannot found the file({e}). Check the path of each files.")
        return

    tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')        
    
    train_loader = get_dataloader(train_df, tokenizer = tokenizer, label_col=col_name, batch_size=batch_size, shuffle=True)
    valid_loader = get_dataloader(valid_df, tokenizer = tokenizer, label_col=col_name, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_df, tokenizer = tokenizer, label_col=col_name, batch_size=batch_size, shuffle=False)
    
    model = DrugClassifier()
    model.load_pretrained_weights(pretrained_path, device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay = 1e-4
    )

    print(f"\n--- Model Training Started ({n_epochs} epochs) ---")
    
    best_model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device
    )
    print("--- Model Training Finished ---")

    df_test_results = predict(best_model, test_loader, device)
    metrics = score(df_test_results)
    metric_names = ["AUPR", "AUC"]
    metrics_dict = dict(zip(metric_names, metrics))

    print(f"\n--- Final Test Results for {data_dir_name} ---")
    for name, value in metrics_dict.items():
        print(f"{name}: {value:.4f}")

    output_dir = f'./performance/cls/seed{num_seed}/' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{data_dir_name}.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"Performance file saved : {output_path}")    
         
    pred_dir = f'./prediction/cls/seed{num_seed}/' 
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f'{data_dir_name}.csv')

    df_test_results.to_csv(pred_path, index = False)
    print(f"Prediction file saved : {pred_path}")    

if __name__ == '__main__':       
    config = define_argparser()
    main_finetuning(config)