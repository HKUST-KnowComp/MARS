import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange

sys.path.append(os.getcwd())


class EventDiscriminationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=80, model="roberta-base", split='trn'):
        # self.original_data = pd.read_csv('../../metaphysical_event_data/total_ensemble.csv', index_col=None)
        self.tokenizer = tokenizer
        self.model = model
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.split = split
        self.sep_token = " "

        if 'gpt2' in model and 'split' == 'trn':
            self.data = self.data[self.data.label == 1]

        self.event_after_transition = self.data['event_after_transition']
        self.inference = self.data['inference']
        self.label = self.data['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if 'bert-' in self.model or 'deberta' in self.model or 'electra' in self.model:
            prompted_sent = "If {}, then {}".format(
                self.event_after_transition[index],
                self.inference[index]
            )
        elif 'gpt2' in self.model:
            prompted_sent = "If {}, then {} [EOS]".format(
                self.event_after_transition[index],
                self.inference[index]
            )
        elif 'roberta' in self.model or 'bart' in self.model:
            prompted_sent = "If {}, then {}".format(
                self.event_after_transition[index],
                self.inference[index]
            )
        else:
            raise ValueError("Model not supported")

        source = self.tokenizer.batch_encode_plus([prompted_sent], padding='max_length', max_length=self.max_length,
                                                  return_tensors='pt', truncation=True)
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        # print(prompted_sent, self.label[index])
        return {
            'ids': source_ids.to(dtype=torch.long),
            'mask': source_mask.to(dtype=torch.long),
            'label': torch.tensor(self.label[index]).to(dtype=torch.long)
        }
