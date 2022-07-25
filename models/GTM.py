import math, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import pipeline
from torchvision import models
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
from fairseq.optim.adafactor import Adafactor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, decoder_input_type, dropout=0.2):
        super(FusionNetwork, self).__init__()
        
        self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.input_type = decoder_input_type
        input_dim = embedding_dim*max(self.input_type,2)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        # Fuse static features together
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))
        if self.input_type == 3:
            concat_features = torch.cat([condensed_img, text_encoding, dummy_encoding], dim=1) # no img
        elif self.input_type == 2:
            concat_features = torch.cat([text_encoding, dummy_encoding], dim=1) # no img
        elif self.input_type == 1:
            concat_features = torch.cat([condensed_img, dummy_encoding], dim=1) # only img

        final = self.feature_fusion(concat_features)

        return final

class POPEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends,  gpu_num):
        super().__init__()

        self.trend_len = trend_len
        self.num_trends = num_trends
        self.embedding_dim = embedding_dim
        self.forecast_horizon = forecast_horizon
        self.input_linear = nn.Linear(num_trends, embedding_dim)
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask
    
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, trend):
        pop_emb = self.input_linear(trend.permute(0,2,1))
        pop_emb = self.pos_embedding(pop_emb.permute(1,0,2))

        input_mask = self._generate_encoder_mask(pop_emb.shape[0], self.forecast_horizon)
        if self.use_mask == 1:
            pop_emb = self.encoder(pop_emb, input_mask)
        else:
            pop_emb = self.encoder(pop_emb)
        return pop_emb, None

        
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, attrs_dict, gpu_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attrs_dict = [{v: k for k, v in x.items()} for x in attrs_dict ]
        self.word_embedder = pipeline('feature-extraction', model='bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.gpu_num = gpu_num

    def forward(self, attrs):
        # Aggregated textual description: color + category e.g "green long sleeve"
      
        # Color + Fabric + Category
        textual_description = [' '.join([self.attrs_dict[ii][attrs[:, ii].detach().cpu().numpy().tolist()[i]] for ii in range(attrs.size(1))][::-1]) for i in range(attrs.size(0))]


        # Use BERT to extract features
        word_embeddings = self.word_embedder(textual_description)

        # BERT gives us embeddings for [CLS] ..  [EOS], which is why we only average the embeddings in the range [1:-1] 
        # We're not fine tuning BERT and we don't want the noise coming from [CLS] or [EOS]
        word_embeddings = [torch.FloatTensor(x[1:-1]).mean(axis=0) for x in word_embeddings] 
        word_embeddings = torch.stack(word_embeddings).to('cuda:'+str(self.gpu_num))
        
        # Embed to our embedding space
        word_embeddings = self.dropout(self.fc(word_embeddings))

        return word_embeddings

class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):        
        img_embeddings = self.resnet(images)  
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2],-1)

        return out.view(*size).contiguous() # batch_size, 2048, image_size/32, image_size/32

class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim*4, embedding_dim)
        self.dropout = nn.Dropout(0.2)


    def forward(self, temporal_features):
        # Temporal dummy variables (day, week, month, year)
        d, w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), \
            temporal_features[:, 2].unsqueeze(1), temporal_features[:, 3].unsqueeze(1)
        d_emb, w_emb, m_emb, y_emb = self.day_embedding(d), self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, 
            memory_key_padding_mask = None):

        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

        
class GTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_heads, num_layers, 
                cat_dict, col_dict, fab_dict, trend_len, num_trends, decoder_input_type, use_encoder_mask=1, autoregressive=False, gpu_num=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.gpu_num = gpu_num
        self.learning_rate = 0.1
        self.warm_up_epochs = 25
        self.save_hyperparameters()

         # Encoder
        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, [cat_dict, fab_dict, col_dict], gpu_num)
        self.POP_encoder = POPEmbedder(output_dim, hidden_dim, use_encoder_mask, trend_len, num_trends, gpu_num)
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, decoder_input_type)

        # Decoder
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, 
                                    dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        
        if self.autoregressive: self.pos_encoder = PositionalEncoding(hidden_dim, max_len=12)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not self.autoregressive else 1),
            nn.Dropout(0.2)
        )
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, attrs, temporal_features, pop_signal, images):
        # Encode features and get inputs
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(attrs)
        pop_signal_encoding, wa1 = self.POP_encoder(pop_signal)

        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, dummy_encoding)

        if self.autoregressive:
            # Decode
            tgt = torch.zeros(self.output_len, pop_signal_encoding.shape[1], pop_signal_encoding.shape[-1]).to('cuda:'+str(self.gpu_num))
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(self.output_len)
            memory = pop_signal_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory, tgt_mask)
            forecast = self.decoder_fc(decoder_out)
        else:
            # Decode
            tgt = static_feature_fusion.unsqueeze(0)
            memory = pop_signal_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory)
            forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(),scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
       
        return [optimizer]


    def training_step(self, train_batch, batch_idx):
        item_sales, attrs, temporal_features, vis_trend, images = train_batch 
        forecasted_sales, _ = self.forward(attrs, temporal_features, vis_trend, images)
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log('train_loss', loss)

        return loss

    def validation_step(self, train_batch, batch_idx):
        item_sales, attrs, temporal_features, vis_trend, images = train_batch 
        forecasted_sales, _ = self.forward(attrs, temporal_features, vis_trend, images)
        
        return item_sales.squeeze(), forecasted_sales.squeeze()

    def validation_epoch_end(self, val_step_outputs):
        item_sales, forecasted_sales = [x[0] for x in val_step_outputs], [x[1] for x in val_step_outputs]
        item_sales, forecasted_sales = torch.stack(item_sales), torch.stack(forecasted_sales)
        rescaled_item_sales, rescaled_forecasted_sales = item_sales*1065, forecasted_sales*1065 # 1065 is the normalization factor (max of the sales of the training set)
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        mae = F.l1_loss(rescaled_item_sales, rescaled_forecasted_sales)
        self.log('val_mae', mae)
        self.log('val_loss', loss)

        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])
