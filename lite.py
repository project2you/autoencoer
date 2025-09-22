import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
import random
import os
import sys
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import glob
import argparse
from tqdm import tqdm
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Utility functions
def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
    return result

def multiply(x): 
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    return torch.sum(sim)/(m*(m-1))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def normalize_img(img):
    img_re = copy.copy(img)
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    return img_re

def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    
    if outputs[0].numel() == 0 or imgs[0].numel() == 0:
        return 0.01
    
    outputs_norm = torch.clamp((outputs[0] + 1) / 2, 0, 1)
    imgs_norm = torch.clamp((imgs[0] + 1) / 2, 0, 1)
    
    if torch.isnan(outputs_norm).any():
        outputs_norm = torch.nan_to_num(outputs_norm, nan=0.5)
    if torch.isnan(imgs_norm).any():
        imgs_norm = torch.nan_to_num(imgs_norm, nan=0.5)
    
    error = loss_func_mse(outputs_norm, imgs_norm)
    
    if torch.isnan(error).any():
        return 0.01
    
    error_clamped = torch.clamp(error, 0, 10)
    normal = 1 - torch.exp(-error_clamped)
    
    normal_sum = torch.sum(normal)
    if normal_sum < 1e-8:
        return 0.01
    
    weighted_error = normal * error
    numerator = torch.sum(weighted_error)
    
    if torch.isnan(numerator) or torch.isnan(normal_sum):
        return 0.01
    
    score = (numerator / normal_sum).item()
    
    if math.isnan(score) or math.isinf(score) or score <= 0:
        return 0.01
    
    return max(score, 1e-8)
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
    return list_result

def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

# Data Loader
class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1):
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
        
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
        return frames               
            
    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])
        
        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], 
                                self._resize_height, self._resize_width)
            if self.transform is not None:
                batch.append(self.transform(image))
        return np.concatenate(batch, axis=0)
        
    def __len__(self):
        return len(self.samples)

# Fixed Memory Module
class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        
        # Initialize memory with orthogonal vectors for better diversity
        memory_init = torch.randn((memory_size, key_dim), dtype=torch.float)
        nn.init.orthogonal_(memory_init)
        self.register_buffer('memory', F.normalize(memory_init, dim=1))
        
        # Add projection layer if needed
        if feature_dim != key_dim:
            self.projection = nn.Linear(key_dim, feature_dim, bias=False)
        else:
            self.projection = nn.Identity()
    
    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()
        
        # Normalize for stable attention
        query_norm = F.normalize(query, dim=-1, eps=1e-8)
        mem_norm = F.normalize(mem, dim=-1, eps=1e-8)
        
        score = torch.matmul(query_norm, torch.t(mem_norm))
        score = score.view(bs*h*w, m)
        
        # Add small epsilon to prevent attention collapse
        score_query = F.softmax(score / self.temp_gather + 1e-8, dim=1)
        score_memory = F.softmax(score / self.temp_update + 1e-8, dim=0)
        
        return score_query, score_memory
    
    def forward(self, query, keys, train=True):
        batch_size, dims, h, w = query.size()
        query = F.normalize(query, dim=1, eps=1e-8)
        query = query.permute(0,2,3,1)
        
        if train:
            separateness_loss, compactness_loss = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = self.update(query, keys, train)
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:
            compactness_loss, query_re, top1_keys, keys_ind = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = self.update(query, keys, train)
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, query_re, top1_keys, keys_ind, compactness_loss
    
    def update(self, query, keys, train):
        batch_size, h, w, dims = query.size()
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        # Soft update instead of hard selection
        weighted_sum = torch.matmul(softmax_score_query.t(), query_reshape)
        
        # Momentum update
        momentum = 0.1
        updated_memory = momentum * weighted_sum + (1 - momentum) * keys
        updated_memory = F.normalize(updated_memory, dim=1, eps=1e-8)
        
        return updated_memory.detach()
    
    def gather_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()
        
        if train:
            loss_mse = torch.nn.MSELoss()
            
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size*h*w, dims)
            
            # Use soft attention for gathering
            gathered = torch.matmul(softmax_score_query, keys)
            compactness_loss = loss_mse(query_reshape, gathered)
            
            # Separateness loss using entropy regularization
            entropy = -torch.sum(softmax_score_query * torch.log(softmax_score_query + 1e-8), dim=1)
            separateness_loss = -torch.mean(entropy)  # Encourage diversity
            
            return separateness_loss, compactness_loss
        else:
            loss_mse = torch.nn.MSELoss()
            
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size*h*w, dims)
            
            _, gathering_indices = torch.topk(softmax_score_query, 1, dim=1)
            gathering_indices = torch.clamp(gathering_indices.squeeze(1), 0, keys.size(0)-1)
            
            gathered = keys[gathering_indices]
            compactness_loss = loss_mse(query_reshape, gathered.detach())
            
            return compactness_loss, query_reshape, gathered.detach(), gathering_indices
    
    def read(self, query, updated_memory):
        batch_size, h, w, dims = query.size()
        
        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        # Read from memory using attention
        concat_memory = torch.matmul(softmax_score_query, updated_memory)
        concat_memory = self.projection(concat_memory)
        
        # Concatenate original query with memory read
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)
        
        return updated_query, softmax_score_query, softmax_score_memory

# Encoder with skip connections
class EncoderWithSkip(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(EncoderWithSkip, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        
        self.moduleConv1 = Basic(n_channel*(t_length-1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        
        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

# Decoder with skip connections
class DecoderWithSkip(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(DecoderWithSkip, self).__init__()
        
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
      
        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128, n_channel, 64)
        
    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)
        
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)
        
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)
        
        output = self.moduleDeconv1(cat2)
        
        return output

# Main ConvAE Model
class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = EncoderWithSkip(t_length, n_channel)
        self.decoder = DecoderWithSkip(t_length, n_channel)
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):
        fea, skip1, skip2, skip3 = self.encoder(x)
        
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:
            updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss = self.memory(fea, keys, train)
            output = self.decoder(updated_fea, skip1, skip2, skip3)
            
            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss

def train_model(args):
    # Environment setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    train_folder = args.dataset_path + "/" + args.dataset_type + "/training/frames"
    test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    # Loading dataset
    train_dataset = DataLoader(train_folder, transforms.Compose([
        transforms.ToTensor(),          
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)
    
    test_dataset = DataLoader(test_folder, transforms.Compose([
        transforms.ToTensor(),            
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, 
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    print(f"Training samples: {train_size}")
    print(f"Testing samples: {test_size}")

    # Model setting
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim, args.temp_update, args.temp_gather)
    
    params_encoder = list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params_memory = list(model.memory.parameters())
    params = params_encoder + params_decoder + params_memory
    
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    sys.stdout = f

    loss_func_mse = nn.MSELoss(reduction='none')

    # Initialize memory with diverse patterns
    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda()
    # Add orthogonal initialization
    nn.init.orthogonal_(m_items)
    m_items = F.normalize(m_items, dim=1)
    
    print("="*60)
    print(f"MNAD Training Started - Method: {args.method}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Memory Size: {args.msize}")
    print(f"Feature Dim: {args.fdim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("="*60)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()
        
        epoch_losses = []
        
        pbar = tqdm(enumerate(train_batch), total=len(train_batch), 
                   desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for j, (imgs) in pbar:
            imgs = Variable(imgs).cuda()
            
            if args.method == 'pred':
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs[:, 0:args.c*(args.t_length-1)], m_items, True)
                target_imgs = imgs[:, args.c*(args.t_length-1):]
            else:
                outputs, _, _, m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = model.forward(imgs, m_items, True)
                target_imgs = imgs
            
            optimizer.zero_grad()
            
            loss_pixel = torch.mean(loss_func_mse(outputs, target_imgs))
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pixel': f'{loss_pixel.item():.4f}',
                'Compact': f'{compactness_loss.item():.4f}',
                'Separate': f'{separateness_loss.item():.4f}'
            })
        
        scheduler.step()
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        avg_loss = np.mean(epoch_losses)
        current_lr = get_lr(optimizer)
        
        print(f'Epoch: {epoch+1}/{args.epochs}')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Learning Rate: {current_lr:.2e}')
        print(f'Average Loss: {avg_loss:.6f}')
        print('-'*60)
    
    print('Training is finished')
    
    # Save the model and memory items
    torch.save(model, os.path.join(log_dir, 'model.pth'))
    torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
    
    sys.stdout = orig_stdout
    f.close()
    
    print(f"Training completed! Results saved in: {log_dir}")
    
    # Auto test after training
    print("\nStarting automatic testing...")
    test_args = type('TestArgs', (), {})()
    test_args.gpus = args.gpus
    test_args.batch_size = args.batch_size
    test_args.test_batch_size = args.test_batch_size
    test_args.h = args.h
    test_args.w = args.w
    test_args.c = args.c
    test_args.method = args.method
    test_args.t_length = args.t_length
    test_args.fdim = args.fdim
    test_args.mdim = args.mdim
    test_args.msize = args.msize
    test_args.alpha = 0.6
    test_args.th = 0.01
    test_args.temp_update = args.temp_update
    test_args.temp_gather = args.temp_gather
    test_args.num_workers = args.num_workers
    test_args.num_workers_test = args.num_workers_test
    test_args.dataset_type = args.dataset_type
    test_args.dataset_path = args.dataset_path
    test_args.model_dir = os.path.join(log_dir, 'model.pth')
    test_args.m_items_dir = os.path.join(log_dir, 'keys.pt')
    
    accuracy = test_model(test_args)
    print(f"\nFinal Test AUC: {accuracy*100:.3f}%")
    
    return model, m_items

def test_model(args):
    # Environment setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpus is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        gpus = ""
        for i in range(len(args.gpus)):
            gpus = gpus + args.gpus[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[:-1]

    torch.backends.cudnn.enabled = True

    test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    # Loading dataset
    test_dataset = DataLoader(test_folder, transforms.Compose([
        transforms.ToTensor(),            
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, 
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    # Loading the trained model
    print(f"Loading model from: {args.model_dir}")
    model = torch.load(args.model_dir, map_location='cuda:0', weights_only=False)
    model.cuda()
    model.eval()
    
    print(f"Loading memory items from: {args.m_items_dir}")
    m_items = torch.load(args.m_items_dir, map_location='cuda:0')
    
    # Load ground truth labels
    labels_path = f'./data/frame_labels_{args.dataset_type}.npy'
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        print(f"Warning: Labels file not found at {labels_path}")
        labels = np.zeros((1, test_size))

    # Setup video information
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    # Prepare evaluation metrics
    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}

    print(f'Evaluation of {args.dataset_type}')
    print("="*60)

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args.method == 'pred':
            labels_list = np.append(labels_list, labels[0][4+label_length:videos[video_name]['length']+label_length])
        else:
            labels_list = np.append(labels_list, labels[0][label_length:videos[video_name]['length']+label_length])
        label_length += videos[video_name]['length']
        psnr_list[video_name] = []
        feature_distance_list[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    # Testing with progress bar
    with torch.no_grad():
        pbar = tqdm(enumerate(test_batch), total=len(test_batch), desc="Testing Progress")
        
        for k, (imgs) in pbar:
            if args.method == 'pred':
                if k == label_length-4*(video_num+1):
                    video_num += 1
                    if video_num < len(videos_list):
                        label_length += videos[videos_list[video_num].split('/')[-1]]['length']
            else:
                if k == label_length:
                    video_num += 1
                    if video_num < len(videos_list):
                        label_length += videos[videos_list[video_num].split('/')[-1]]['length']

            imgs = Variable(imgs).cuda()
            
            if args.method == 'pred':
                outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, _, _, _, compactness_loss = model.forward(imgs[:, 0:args.c*(args.t_length-1)], m_items_test, False)
                target_imgs = imgs[:, args.c*(args.t_length-1):]
                point_sc = point_score(outputs, target_imgs)
            else:
                outputs, feas, updated_feas, m_items_test, softmax_score_query, softmax_score_memory, compactness_loss = model.forward(imgs, m_items_test, False)
                target_imgs = imgs
                point_sc = point_score(outputs, target_imgs)
            
            mse_imgs = torch.mean(loss_func_mse((outputs[0]+1)/2, (target_imgs[0]+1)/2)).item()
            mse_feas = compactness_loss.item()
            
            # Update memory if necessary
            if point_sc < args.th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)
                m_items_test = model.memory.update(query, m_items_test, False)

            current_video = videos_list[min(video_num, len(videos_list)-1)].split('/')[-1]
            psnr_val = psnr(mse_imgs)
            psnr_list[current_video].append(psnr_val)
            feature_distance_list[current_video].append(mse_feas)
            
            pbar.set_postfix({
                'Video': f'{video_num+1}/{len(videos_list)}',
                'PSNR': f'{psnr_val:.2f}',
                'Feature': f'{mse_feas:.4f}'
            })

    # Calculate anomaly scores
    anomaly_score_total_list = []
    
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        
        video_psnr_scores = anomaly_score_list(psnr_list[video_name])
        video_feature_scores = anomaly_score_list_inv(feature_distance_list[video_name])
        video_combined_scores = score_sum(video_psnr_scores, video_feature_scores, args.alpha)
        
        anomaly_score_total_list += video_combined_scores

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # Calculate evaluation metrics
    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
    
    threshold = np.median(anomaly_score_total_list)
    predictions = (anomaly_score_total_list > threshold).astype(int)
    true_labels = (1-labels_list).astype(int)
    
    cm = confusion_matrix(true_labels, predictions)
    cr = classification_report(true_labels, predictions, output_dict=True)

    print("\n" + "="*60)
    print("ANOMALY DETECTION RESULTS")
    print("="*60)
    print(f'Dataset: {args.dataset_type}')
    print(f'Method: {args.method}')
    print(f'AUC: {accuracy*100:.3f}%')
    print(f'Precision: {cr["1"]["precision"]:.3f}')
    print(f'Recall: {cr["1"]["recall"]:.3f}')
    print(f'F1-Score: {cr["1"]["f1-score"]:.3f}')
    print(f'Confusion Matrix:')
    print(f'TN: {cm[0,0]}, FP: {cm[0,1]}')
    print(f'FN: {cm[1,0]}, TP: {cm[1,1]}')
    print("="*60)
    
    return accuracy

def get_train_args():
    parser = argparse.ArgumentParser(description="MNAD Training")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--method', type=str, default='pred', choices=['pred', 'recon'], help='The target task for anomaly detection')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--temp_update', type=float, default=0.1, help='temperature for memory update')
    parser.add_argument('--temp_gather', type=float, default=0.1, help='temperature for memory gather')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'], help='type of dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_train_args()
    print("Starting MNAD training...")
    model, memory_items = train_model(args)
    print("Training and testing completed!")
# python main.py --gpus 0 --batch_size 8 --epochs 3 --dataset_type avenue --dataset_path ./data --method pred --exp_dir log
