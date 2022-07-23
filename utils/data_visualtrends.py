import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler

class ZeroShotDataset():
    def __init__(self, data_df, gtrends, cat_dict, col_dict, tex_dict, shape_dict):
        self.data_df = data_df
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.tex_dict = tex_dict
        self.shape_dict = shape_dict
        self.img_root = '/media/data/cjoppi/ICCV2021/dataset/images_clean/'

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def preprocess_data(self):
        data = self.data_df

        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        gtrends, image_features = [], []
        img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            cat, col, tex, fiq_attr, start_date, img_path = row['category'], row['exact_color'], row['texture'], row['finegrained_attr'],\
                row['release_date'], row['image_path']

            # Get the gtrend signal from the previous year (52 weeks) of the release date
            len_gtrend = 52

            vis_trend = self.gtrends[row['external_code']].unsqueeze(0)

            # vis_trend = MinMaxScaler().fit_transform(vis_trend[:len_gtrend])
        
            # multitrends = np.vstack([mixed_gtrend, tex_gtrend, shape_gtrend, singles_gtrends[torch.randperm(singles_gtrends.shape[0]),:]])


            # Read images
            img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

            # Append them to the lists
            gtrends.append(vis_trend)
            image_features.append(img_transforms(img))

        # Convert to numpy arrays
        gtrends = torch.stack(gtrends)

        # Remove non-numerical information
        data.drop(['external_code', 'season', 'release_date', 'image_path'], axis=1, inplace=True)

        # Create tensors for each part of the input/output
        item_sales, temporal_features = torch.FloatTensor(data.iloc[:, :12].values), torch.FloatTensor(
            data.iloc[:, 14:18].values)
        categories, colors, textures, shape = [self.cat_dict[val] for val in data.iloc[:].category], \
                                       [self.col_dict[val] for val in data.iloc[:].exact_color], \
                                       [self.tex_dict[val] for val in data.iloc[:].texture], \
                                       [self.shape_dict[val] for val in data.iloc[:].finegrained_attr]

        
        categories, colors, textures, shape = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(textures), torch.LongTensor(shape)
        gtrends = torch.FloatTensor(gtrends)
        images = torch.stack(image_features)

        return TensorDataset(item_sales, torch.vstack([categories, textures, colors ]).T, temporal_features, gtrends, images)

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
