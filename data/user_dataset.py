import torch
from torch.utils.data import Dataset
from .content_dataset import ContentDataset

import numpy as np
import random

#generated synthetic plausible diff user's dataset
#gender, age, user region, watch history embed state (probably just average of the latest series embeds)

REGION_MAP = ["na", "eu", "apac", "latam"]
class UserDataset(Dataset):

    def __init__(self, db_size:int =302, size:int=1000, history_length:int=30, max_age:int=100):
        #keep track of already generated users (for consistency)
        self.histories = {}
        self.ages = [-1] * size
        self.genders = [-1] * size
        self.regions = [-1] * size

        self.ages_check = set()
        self.genders_check = set()
        self.regions_check = set()

        #other attr.
        self.db_size = db_size
        self.size = size
        self.history_length = history_length
        self.max_age = max_age

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx not in self.histories:
            self.histories[idx] = self.generate_history()
            self.ages[idx] = torch.tensor(random.randint(0, self.max_age), dtype=torch.float32).unsqueeze(0)
            self.genders[idx] = self.generate_gender()
            self.regions[idx] = self.generate_region()

        return {
            "age": self.ages[idx],
            "gender": self.genders[idx],
            "region": self.regions[idx],
            "watch_history": self.histories[idx],
        }

    def generate_region(self):
        region = random.randint(0, len(REGION_MAP) - 1)
        return torch.tensor([1. if i == region else 0. for i in range(len(REGION_MAP))], dtype=torch.float32)

    def generate_gender(self):
        male = random.randint(0, 1)
        return torch.tensor([male], dtype=torch.float32)

    def generate_history(self):
        indicies = np.random.choice(self.db_size, self.history_length, replace=True) #replace = true since can have repeated watches
        return torch.tensor(indicies, dtype=torch.long)

    def generate_random_user(self): #for inference
        return {
            "age": torch.tensor(random.randint(0, self.max_age)),
            "gender": self.generate_gender(),
            "region": self.generate_region(),
            "watch_history": self.generate_history(),
        }