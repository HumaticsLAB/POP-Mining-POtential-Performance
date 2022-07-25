import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler

class POPDataset():
    def __init__(self, data_df, img_root, pop_signal, cat_dict, col_dict, fab_dict, trend_len):
        self.data_df = data_df
        self.pop_signal = pop_signal
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = img_root

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def preprocess_data(self):
        data = self.data_df

        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        pop_signals, image_features = [], []
        img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            
            start_date, img_path =  row['release_date'], row['image_path']

            pop_signal_start = start_date - pd.DateOffset(weeks=52)
            pop_signal = self.pop_signal[row['external_code']][-52:][:self.trend_len].unsqueeze(0)

            # Read images
            img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

            # Append them to the lists
            pop_signals.append(pop_signal)
            image_features.append(img_transforms(img))

        # Convert to numpy arrays
        pop_signals = torch.stack(pop_signals)

        # Remove non-numerical information
        data.drop(['external_code', 'season', 'release_date', 'image_path'], axis=1, inplace=True)

        # Create tensors for each part of the input/output
        item_sales, temporal_features = torch.FloatTensor(data.iloc[:, :12].values), torch.FloatTensor(
            data.iloc[:, 14:18].values)
        categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category], \
                                       [self.col_dict[val] for val in data.iloc[:].color], \
                                       [self.fab_dict[val] for val in data.iloc[:].fabric]

        
        categories, colors, textures = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(textures)
        pop_signals = torch.FloatTensor(pop_signals)
        images = torch.stack(image_features)

        return TensorDataset(item_sales, torch.vstack([categories, textures, colors ]).T, temporal_features, pop_signals, images)

    def get_loader(self, batch_size, train=True):
        print('Starting dataset creation process...')
        data_with_gtrends = self.preprocess_data()
        data_loader = None
        if train:
            data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=8)
        else:
            data_loader = DataLoader(data_with_gtrends, batch_size=1, shuffle=False, num_workers=4)
        print('Done.')

        return data_loader
