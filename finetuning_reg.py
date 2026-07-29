import numpy as np
import pandas as pd
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from sklearn.metrics import *
from scipy.stats import pearsonr 
import copy
import pickle

from transformers import AutoModel, AutoTokenizer
import argparse

def set_seed(seed: int = 42):
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
    label = df['label'].values
    pred = df['pred'].values
    
    try:
        mae = mean_absolute_error(label, pred)
        pcc, _ = pearsonr(label, pred) 
        
    except Exception as e:
        print(f"Metric calculation error - {e}")
        return 0.0, 0.0

    return mae, pcc

def predict(model, dataloader, device):

    model.eval()
    result = []
    label = []

    with torch.no_grad():
        for inputs, target in dataloader:

            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(inputs)
            
            result.append(output)
            label.append(target)

    result = torch.cat(result).cpu().detach().numpy().squeeze()
    label = torch.cat(label).cpu().detach().numpy().squeeze()

    df = pd.DataFrame()
    df['pred'] = result
    df['label'] = label

    return df

def train_model(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device):
    best_score = 1e6
    best_epoch = 0
    best_model = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.squeeze(1), labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(inputs)
                
                all_labels.append(labels.numpy())
                all_preds.append(logits.cpu().numpy())
        
        all_labels = np.concatenate(all_labels).squeeze()
        all_preds = np.concatenate(all_preds).squeeze()

        try:
            val_mse = mean_squared_error(all_labels, all_preds)
        except:
            val_mse = -1.0
 
        if val_mse < best_score:
            best_epoch = epoch
            best_score = val_mse 
            best_model = copy.deepcopy(model)
            
    print(f'Best epoch : {best_epoch} (MSE: {best_score:.4f})')
    return best_model

def prepare_input(tokenizer, smiles):
    inputs = tokenizer(smiles, add_special_tokens=True, truncation=True, max_length=131, padding="max_length")
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs

class ChemiDataset(Dataset):
    def __init__(self, df, tokenizer, label_col='label'):
        self.smiles = df['smiles'].values
        self.tokenizer = tokenizer
        self.label = df[label_col].values
        self.inputs = [prepare_input(tokenizer, smi) for smi in df['smiles'].values]
            
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, item):
        inputs = self.inputs[item] 
        label = torch.tensor(self.label[item], dtype=torch.float)
        return inputs, label    

def get_dataloader(df, label_col, tokenizer, batch_size=64, shuffle=True):
    dataset = ChemiDataset(df, tokenizer, label_col=label_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader

class DrugEncoder(nn.Module):
    def __init__(self, pretrained_model='DeepChem/ChemBERTa-77M-MTR'):
        super(DrugEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        
    def forward(self, inputs):
        embedding = self.bert(**inputs)
        hidden = embedding[0][:, 0, :] 
        return hidden

class DrugRegressor(nn.Module):
    def __init__(self, hid_dim=384, output_dim=1):
        super().__init__()
        self.drug_encoder = DrugEncoder()
        
        self.mlp_head = nn.Sequential(
            nn.Linear(hid_dim, hid_dim//4),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hid_dim//4, hid_dim//8),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hid_dim//8, output_dim))
        
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
    p.add_argument('--batch_size', type=int, default=32) 
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

    data_path = f'./finetuning_data/Regression/{data_dir_name}/'
    pretrained_path = './pre_trained_weights/pt_weights.pth'  
    
    try:
        train_df = pd.read_csv(data_path + 'train.csv')
        valid_df = pd.read_csv(data_path + 'valid.csv')
        test_df = pd.read_csv(data_path + 'test.csv')

    except FileNotFoundError as e:
        print(f"Cannot found the file({e}). Check the path of each files.")
        return

    tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
    train_loader = get_dataloader(train_df, label_col=col_name, tokenizer=tokenizer, batch_size=batch_size, shuffle=True)
    valid_loader = get_dataloader(valid_df, label_col=col_name, tokenizer=tokenizer, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_df, label_col=col_name, tokenizer=tokenizer, batch_size=batch_size, shuffle=False)
    
    model = DrugRegressor()
    model.load_pretrained_weights(pretrained_path, device)
    model.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, weight_decay = 1e-4
    )

    print(f"\n--- Regression Training Started ({n_epochs} epochs) ---")
    best_model = train_model(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device)

    df_test_results = predict(best_model, test_loader, device)
    metrics_vals = score(df_test_results)
    metric_names = ["MAE", "PCC"]
    metrics_dict = dict(zip(metric_names, metrics_vals))

    print(f"\n--- Final Test Results for {data_dir_name} ---")
    for name, value in metrics_dict.items():
        print(f"{name}: {value:.4f}")

    output_dir = f'./performance/reg/seed{num_seed}/' 
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{data_dir_name}.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"Performance file saved : {output_path}")    
         
    pred_dir = f'./prediction/reg/seed{num_seed}/' 
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f'{data_dir_name}.csv')

    df_test_results.to_csv(pred_path, index = False)
    print(f"Prediction file saved : {pred_path}")    

if __name__ == '__main__':
    config = define_argparser()
    main_finetuning(config)