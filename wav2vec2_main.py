#!/usr/bin/env python
# coding: utf-8

# ## Import

# In[1]:


import os
import random
import warnings
from sklearn.model_selection import train_test_split
from custom import CustomDataSet
from collate import collate_fn

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

warnings.filterwarnings(action='ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ## Hyperparameter Setting

# In[27]:


CFG = {
    'SR':16_000,
    'SEED':42,
    'BATCH_SIZE':8, # out of Memory가 발생하면 줄여주세요
    'TOTAL_BATCH_SIZE':32, # 원하는 batch size
    'EPOCHS':11,
    'LR':1e-4,
}


# In[28]:


MODEL_NAME = "facebook/wav2vec2-base"


# ## Fixed Random-Seed

# In[29]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


# ## Data Pre-Processing

# In[ ]:





# In[30]:


train_df = pd.read_csv('./train.csv')


# In[31]:


train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=CFG['SEED'])


# In[32]:


train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)


# In[33]:


def speech_file_to_array_fn(df):
    feature = []
    for path in tqdm(df['path']):
        speech_array, _ = librosa.load(path, sr=CFG['SR'])
        feature.append(speech_array)
    return feature


# In[34]:


train_x = speech_file_to_array_fn(train_df)
valid_x = speech_file_to_array_fn(valid_df)


# In[35]:


processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)


# ## DataLoader

# In[36]:


def create_data_loader(dataset, batch_size, shuffle, collate_fn, num_workers=0):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=collate_fn,
                      num_workers=num_workers
                      )

train_dataset = CustomDataSet(train_x, train_df['label'], processor)
valid_dataset = CustomDataSet(valid_x, valid_df['label'], processor)

train_loader = create_data_loader(train_dataset, CFG['BATCH_SIZE'], False, collate_fn, 16)
valid_loader = create_data_loader(valid_dataset, CFG['BATCH_SIZE'], False, collate_fn, 16)


# ## Train

# In[37]:


audio_model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)


# In[38]:


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = audio_model
        self.model.classifier = nn.Identity()
        self.classifier = nn.Linear(256, 8)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output


# In[39]:


def validation(model, valid_loader, creterion):
    model.eval()
    val_loss = []

    total, correct = 0, 0
    test_loss = 0

    with torch.no_grad():
        for x, y in tqdm(iter(valid_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            output = model(x)
            loss = creterion(output, y)

            val_loss.append(loss.item())

            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += predicted.eq(y).cpu().sum()

    accuracy = correct / total

    avg_loss = np.mean(val_loss)

    return avg_loss, accuracy


# In[40]:


def train(model, train_loader, valid_loader, optimizer, scheduler):
    accumulation_step = int(CFG['TOTAL_BATCH_SIZE'] / CFG['BATCH_SIZE'])
    model.to(device)
    creterion = nn.CrossEntropyLoss().to(device)

    best_model = None
    best_acc = 0

    for epoch in range(1, CFG['EPOCHS']+1):
        train_loss = []
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.flatten().to(device)

            optimizer.zero_grad()
            
            output = model(x)
            loss = creterion(output, y)
            loss.backward()

            if (i+1) % accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss.append(loss.item())

        avg_loss = np.mean(train_loss)
        valid_loss, valid_acc = validation(model, valid_loader, creterion)

        if scheduler is not None:
            scheduler.step(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model

        print(f'epoch:[{epoch}] train loss:[{avg_loss:.5f}] valid_loss:[{valid_loss:.5f}] valid_acc:[{valid_acc:.5f}]')
    
    print(f'best_acc:{best_acc:.5f}')

    return best_model


# ## Run

# In[41]:


model = BaseModel()

optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)



# In[ ]:


infer_model = train(model, train_loader, valid_loader, optimizer, scheduler)


# ## Inference

# In[19]:


test_df = pd.read_csv('./test.csv')


# In[20]:


def collate_fn_test(batch):
    x = pad_sequence([torch.tensor(xi) for xi in batch], batch_first=True)
    return x


# In[21]:


test_x = speech_file_to_array_fn(test_df)


# In[22]:


test_dataset = CustomDataSet(test_x, y=None, processor=processor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_test)


# In[23]:


def inference(model, test_loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for x in tqdm(iter(test_loader)):
            x = x.to(device)

            output = model(x)

            preds += output.argmax(-1).detach().cpu().numpy().tolist()

    return preds


# In[24]:


preds = inference(infer_model, test_loader)


# ## Submission

# In[25]:


submission = pd.read_csv('./sample_submission.csv')
submission['label'] = preds
submission.to_csv('./baseline_submission.csv', index=False)


# In[ ]:




