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
    """Generate random uniform tensor"""
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu
    
def distance(a, b):
    """Calculate euclidean distance between two tensors"""
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)

def distance_batch(a, b):
    """Calculate batch distances"""
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs-1):
        result = torch.cat((result, distance(a[i], b)), 0)
    return result

def multiply(x): 
    """Flatten matrix into a vector"""
    return functools.reduce(lambda x,y: x*y, x, 1)

def flatten(x):
    """Flatten matrix into a vector"""
    count = multiply(x.size())
    return x.resize_(count)

def index(batch_size, x):
    """Create index tensor"""
    idx = torch.arange(0, batch_size).long() 
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)

def MemoryLoss(memory):
    """Calculate memory regularization loss"""
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t))/2 + 1/2
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)
    return torch.sum(sim)/(m*(m-1))

def rmse(predictions, targets):
    """Root Mean Square Error"""
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):
    """Peak Signal-to-Noise Ratio"""
    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    """Get learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def normalize_img(img):
    """Normalize image to [0, 1]"""
    img_re = copy.copy(img)
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    return img_re

def point_score(outputs, imgs):
    """Calculate point-wise anomaly score - Fixed for NaN handling"""
    loss_func_mse = nn.MSELoss(reduction='none')
    
    # Input validation and clipping
    if outputs[0].numel() == 0 or imgs[0].numel() == 0:
        return 0.01
    
    # Normalize to [0,1] range with safety clipping
    outputs_norm = torch.clamp((outputs[0] + 1) / 2, 0, 1)
    imgs_norm = torch.clamp((imgs[0] + 1) / 2, 0, 1)
    
    # Check for NaN in inputs
    if torch.isnan(outputs_norm).any():
        outputs_norm = torch.nan_to_num(outputs_norm, nan=0.5)
    if torch.isnan(imgs_norm).any():
        imgs_norm = torch.nan_to_num(imgs_norm, nan=0.5)
    
    # Calculate error with bounds checking
    error = loss_func_mse(outputs_norm, imgs_norm)
    
    # Check for NaN in error
    if torch.isnan(error).any():
        return 0.01
    
    # Clamp error to prevent overflow in exp
    error_clamped = torch.clamp(error, 0, 10)
    
    # Calculate normal with numerical stability
    normal = 1 - torch.exp(-error_clamped)
    
    # Avoid division by zero
    normal_sum = torch.sum(normal)
    if normal_sum < 1e-8:
        return 0.01
    
    # Calculate weighted error
    weighted_error = normal * error
    numerator = torch.sum(weighted_error)
    
    # Final safety checks
    if torch.isnan(numerator) or torch.isnan(normal_sum):
        return 0.01
    
    score = (numerator / normal_sum).item()
    
    # Validate final result
    if math.isnan(score) or math.isinf(score) or score <= 0:
        return 0.01
    
    return max(score, 1e-8)
    
def anomaly_score(psnr, max_psnr, min_psnr):
    """Calculate anomaly score from PSNR"""
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    """Calculate inverse anomaly score from PSNR"""
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    """Convert PSNR list to anomaly scores"""
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    """Convert PSNR list to inverse anomaly scores"""
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return anomaly_score_list

def AUC(anomal_scores, labels):
    """Calculate Area Under Curve"""
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    """Combine two score lists"""
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
    return list_result

def np_load_frame(filename, resize_height, resize_width):
    """Load and preprocess image frame"""
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

# Enhanced Data Loader with progress tracking
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
        """Setup video dataset with progress tracking"""
        videos = glob.glob(os.path.join(self.dir, '*'))
        print(f"Setting up dataset from {self.dir}")
        print(f"Found {len(videos)} videos")
        
        for video in tqdm(sorted(videos), desc="Processing videos"):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
        
        total_frames = sum([self.videos[v]['length'] for v in self.videos])
        print(f"Total frames loaded: {total_frames}")
        
    def get_all_samples(self):
        """Get all valid frame sequences"""
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

# Fixed Memory Module with proper attention mechanism
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
        
        self.register_buffer('memory_usage', torch.zeros(memory_size))
        self.register_buffer('memory_age', torch.zeros(memory_size))
        
        # Add projection layer if needed - this is crucial fix
        if feature_dim != key_dim:
            self.projection = nn.Linear(key_dim, feature_dim, bias=False)
        else:
            self.projection = nn.Identity()
    
    def update_memory_stats(self, indices):
        """Update memory usage statistics"""
        if indices.numel() > 0:
            indices_flat = indices.flatten()
            valid_mask = (indices_flat >= 0) & (indices_flat < self.memory_size)
            valid_indices = indices_flat[valid_mask]
            
            if valid_indices.numel() > 0:
                unique_indices = torch.unique(valid_indices)
                self.memory_usage[unique_indices] += 1
                self.memory_age[unique_indices] = 0
                self.memory_age += 1
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        return {
            'usage_mean': self.memory_usage.mean().item(),
            'usage_std': self.memory_usage.std().item(),
            'usage_max': self.memory_usage.max().item(),
            'usage_min': self.memory_usage.min().item(),
            'active_memories': (self.memory_usage > 0).sum().item(),
            'total_accesses': self.memory_usage.sum().item()
        }
    
    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        """Update query based on memory attention"""
        m, d = mem.size()
        query_update = torch.zeros((m, d), device=mem.device, dtype=mem.dtype)
        
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i, as_tuple=False)
            if idx.numel() > 0:
                weights = score[idx, i]
                if weights.sum() > 1e-8:
                    weights = weights / (weights.sum() + 1e-8)
                    query_update[i] = torch.sum(weights * query[idx].squeeze(1), dim=0)
        
        return query_update

    def get_score(self, mem, query):
        """Calculate attention scores with fixed collapse prevention"""
        bs, h, w, d = query.size()
        m, d = mem.size()
        
        # Normalize for stable attention
        query_norm = F.normalize(query, dim=-1, eps=1e-8)
        mem_norm = F.normalize(mem, dim=-1, eps=1e-8)
        
        score = torch.matmul(query_norm, torch.t(mem_norm))
        score = score.view(bs*h*w, m)
        
        # Add diversity encouragement and prevent collapse
        diversity_penalty = torch.log(self.memory_usage + 1.0) * 0.01
        score = score - diversity_penalty.unsqueeze(0)
        
        # Temperature scaling with added epsilon to prevent collapse
        score_query = F.softmax(score / self.temp_gather + 1e-4, dim=1)
        score_memory = F.softmax(score / self.temp_update + 1e-4, dim=0)
        
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
            
            self.update_memory_stats(keys_ind)
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, query_re, top1_keys, keys_ind, compactness_loss
    
    def update(self, query, keys, train):
        """Update memory items with enhanced mechanism"""
        batch_size, h, w, dims = query.size()
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        # Use soft attention for update instead of hard selection
        weighted_update = torch.matmul(softmax_score_query.t(), query_reshape)
        
        # Update memory usage statistics
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        self.update_memory_stats(gathering_indices.squeeze(1))
        
        # Enhanced memory update with momentum
        momentum = 0.1 if train else 0.05
        updated_memory = momentum * weighted_update + (1 - momentum) * keys
        updated_memory = F.normalize(updated_memory, dim=1, eps=1e-8)
        
        # Ensure memory diversity with regularization
        similarity = torch.matmul(updated_memory, updated_memory.t())
        identity_mask = torch.eye(self.memory_size, device=updated_memory.device)
        diversity_loss = torch.mean(torch.abs(similarity - identity_mask))
        
        if diversity_loss > 0.9:  # If memories are too similar
            noise = torch.randn_like(updated_memory) * 0.01
            updated_memory = F.normalize(updated_memory + noise, dim=1, eps=1e-8)
        
        return updated_memory.detach()
    
    def gather_loss(self, query, keys, train):
        """Calculate gathering and separateness loss"""
        batch_size, h, w, dims = query.size()
        
        if train:
            loss = torch.nn.TripletMarginLoss(margin=1.0)
            loss_mse = torch.nn.MSELoss()
            
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size*h*w, dims)
            
            # Use soft attention for compactness
            gathered = torch.matmul(softmax_score_query, keys)
            compactness_loss = loss_mse(query_reshape, gathered)
            
            # Entropy-based separateness loss to encourage diversity
            entropy = -torch.sum(softmax_score_query * torch.log(softmax_score_query + 1e-8), dim=1)
            separateness_loss = -torch.mean(entropy)  # Encourage high entropy (diversity)
            
            return separateness_loss, compactness_loss
        else:
            loss_mse = torch.nn.MSELoss()
            
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size*h*w, dims)
            
            _, gathering_indices = torch.topk(softmax_score_query, 1, dim=1)
            gathering_indices = torch.clamp(gathering_indices, 0, keys.size(0)-1)
            
            gathered = keys[gathering_indices.squeeze(1)]
            compactness_loss = loss_mse(query_reshape, gathered.detach())
            
            return compactness_loss, query_reshape, gathered.detach(), gathering_indices[:,0]
    
    def read(self, query, updated_memory):
        """Read from memory and update query with fixed projection"""
        batch_size, h, w, dims = query.size()
        
        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        # Read from memory using attention
        concat_memory = torch.matmul(softmax_score_query, updated_memory)
        
        # Use the learned projection layer (this is the crucial fix)
        concat_memory = self.projection(concat_memory)
        
        # Concatenate original query with memory read
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)
        
        return updated_query, softmax_score_query, softmax_score_memory

# Enhanced Encoder with skip connections
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

# Enhanced Decoder with skip connections
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
            
            return updated_fea, keys, softmax_score_query, softmax_score_memory, query, top1_keys, keys_ind, compactness_loss

# Enhanced Training Statistics Tracker
class TrainingStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.pixel_losses = []
        self.compact_losses = []
        self.separate_losses = []
        self.learning_rates = []
        self.epoch_times = []
        self.memory_stats = []
    
    def update(self, total_loss, pixel_loss, compact_loss, separate_loss, lr, epoch_time, memory_stats=None):
        self.losses.append(total_loss)
        self.pixel_losses.append(pixel_loss)
        self.compact_losses.append(compact_loss) 
        self.separate_losses.append(separate_loss)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        if memory_stats:
            self.memory_stats.append(memory_stats)
    
    def save_stats(self, path):
        stats_dict = {
            'losses': self.losses,
            'pixel_losses': self.pixel_losses,
            'compact_losses': self.compact_losses,
            'separate_losses': self.separate_losses,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'memory_stats': self.memory_stats
        }
        with open(os.path.join(path, 'training_stats.json'), 'w') as f:
            json.dump(stats_dict, f, indent=2)
    
    def plot_training_curves(self, save_path):
        """Plot comprehensive training curves"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0,0].plot(self.losses, label='Total Loss', color='red', linewidth=2)
        axes[0,0].plot(self.pixel_losses, label='Pixel Loss', color='blue', linewidth=2)
        axes[0,0].plot(self.compact_losses, label='Compact Loss', color='green', linewidth=2)
        axes[0,0].plot(self.separate_losses, label='Separate Loss', color='orange', linewidth=2)
        axes[0,0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[0,1].plot(self.learning_rates, color='purple', linewidth=2)
        axes[0,1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Learning Rate')
        axes[0,1].set_yscale('log')
        axes[0,1].grid(True, alpha=0.3)
        
        # Epoch timing
        axes[0,2].plot(self.epoch_times, color='brown', linewidth=2)
        axes[0,2].set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Epoch')
        axes[0,2].set_ylabel('Time (seconds)')
        axes[0,2].grid(True, alpha=0.3)
        
        # Loss components stacked
        axes[1,0].fill_between(range(len(self.pixel_losses)), self.pixel_losses, alpha=0.7, label='Pixel Loss')
        axes[1,0].fill_between(range(len(self.compact_losses)), self.compact_losses, alpha=0.7, label='Compact Loss')
        axes[1,0].fill_between(range(len(self.separate_losses)), self.separate_losses, alpha=0.7, label='Separate Loss')
        axes[1,0].set_title('Loss Components (Stacked)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Memory usage statistics
        if self.memory_stats:
            mem_usage = [stats.get('usage_mean', 0) for stats in self.memory_stats]
            mem_active = [stats.get('active_memories', 0) for stats in self.memory_stats]
            
            ax_mem = axes[1,1]
            ax_mem2 = ax_mem.twinx()
            
            line1 = ax_mem.plot(mem_usage, color='cyan', linewidth=2, label='Avg Usage')
            line2 = ax_mem2.plot(mem_active, color='magenta', linewidth=2, label='Active Memories')
            
            ax_mem.set_title('Memory Statistics', fontsize=14, fontweight='bold')
            ax_mem.set_xlabel('Epoch')
            ax_mem.set_ylabel('Average Usage', color='cyan')
            ax_mem2.set_ylabel('Active Memories', color='magenta')
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_mem.legend(lines, labels, loc='upper left')
            ax_mem.grid(True, alpha=0.3)
        
        # Training efficiency
        if len(self.epoch_times) > 1:
            efficiency = np.array(self.losses[1:]) / np.array(self.epoch_times[1:])
            axes[1,2].plot(efficiency, color='olive', linewidth=2)
            axes[1,2].set_title('Training Efficiency (Loss/Time)', fontsize=14, fontweight='bold')
            axes[1,2].set_xlabel('Epoch')
            axes[1,2].set_ylabel('Loss per Second')
            axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Enhanced Training Function with comprehensive tracking
def train_model(args):
    """Enhanced training with progress tracking and statistics"""
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
    print(f"Batches per epoch: {len(train_batch)}")

    # Model setting
    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim, args.temp_update, args.temp_gather)
    
    params_encoder = list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params_memory = list(model.memory.parameters())
    params = params_encoder + params_decoder + params_memory
    
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    # Initialize statistics tracker
    stats = TrainingStats()

    # Report the training process
    log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    sys.stdout = f

    loss_func_mse = nn.MSELoss(reduction='none')

    # Initialize memory with diverse patterns and orthogonal initialization
    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda()
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

    # Training loop with enhanced progress tracking
    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()
        
        epoch_losses = []
        epoch_pixel_losses = []
        epoch_compact_losses = []
        epoch_separate_losses = []
        
        # Progress bar for batches
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
            
            # Calculate pixel loss
            loss_pixel = torch.mean(loss_func_mse(outputs, target_imgs))
            
            # Total loss with memory regularization
            mem_loss = MemoryLoss(m_items) * 0.01
            loss = loss_pixel + args.loss_compact * compactness_loss + args.loss_separate * separateness_loss + mem_loss
            
            loss.backward(retain_graph=True)
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            epoch_losses.append(loss.item())
            epoch_pixel_losses.append(loss_pixel.item())
            epoch_compact_losses.append(compactness_loss.item())
            epoch_separate_losses.append(separateness_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pixel': f'{loss_pixel.item():.4f}',
                'Compact': f'{compactness_loss.item():.4f}',
                'Separate': f'{separateness_loss.item():.4f}',
                'Mem': f'{mem_loss.item():.4f}'
            })
        
        scheduler.step()
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_pixel = np.mean(epoch_pixel_losses) 
        avg_compact = np.mean(epoch_compact_losses)
        avg_separate = np.mean(epoch_separate_losses)
        current_lr = get_lr(optimizer)
        
        # Get memory statistics
        memory_stats = model.memory.get_memory_stats()
        
        # Update statistics
        stats.update(avg_loss, avg_pixel, avg_compact, avg_separate, current_lr, epoch_time, memory_stats)
        
        print('='*60)
        print(f'Epoch: {epoch+1}/{args.epochs}')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Learning Rate: {current_lr:.2e}')
        if args.method == 'pred':
            print(f'Loss: Prediction {avg_pixel:.6f}/ Compactness {avg_compact:.6f}/ Separateness {avg_separate:.6f}')
        else:
            print(f'Loss: Reconstruction {avg_pixel:.6f}/ Compactness {avg_compact:.6f}/ Separateness {avg_separate:.6f}')
        print(f'Total Loss: {avg_loss:.6f}')
        
        # Print memory statistics
        if memory_stats:
            print(f'Memory Stats: Active={memory_stats["active_memories"]}/{args.msize}, '
                  f'Avg Usage={memory_stats["usage_mean"]:.2f}, '
                  f'Usage Std={memory_stats["usage_std"]:.2f}')
        
        print('='*60)
        
        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for imgs in tqdm(test_batch, desc="Validation"):
                    imgs = Variable(imgs).cuda()
                    
                    if args.method == 'pred':
                        outputs, _, _, _, _, _, _, _, _, val_loss = model.forward(imgs[:, 0:args.c*(args.t_length-1)], m_items, False)
                    else:
                        outputs, _, _, _, _, _, _, val_loss = model.forward(imgs, m_items, False)
                    
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f'Validation Loss: {avg_val_loss:.6f}')
            print('='*60)
    
    print('Training is finished')
    
    # Save the model and memory items
    torch.save(model, os.path.join(log_dir, 'model.pth'))
    torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
    
    # Save training statistics
    stats.save_stats(log_dir)
    
    # Plot training curves
    stats.plot_training_curves(log_dir)
    
    # Generate training report
    generate_training_report(stats, log_dir, args)
    
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
    
    accuracy = test_model_enhanced(test_args)
    print(f"\nFinal Test AUC: {accuracy*100:.3f}%")
    
    return model, m_items, stats

def generate_training_report(stats, log_dir, args):
    """Generate comprehensive training report"""
    report_path = os.path.join(log_dir, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MNAD TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Training configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: {args.dataset_type}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Memory Size: {args.msize}\n")
        f.write(f"Feature Dimension: {args.fdim}\n")
        f.write(f"Loss Weights - Compact: {args.loss_compact}, Separate: {args.loss_separate}\n\n")
        
        # Training statistics
        f.write("TRAINING STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Final Loss: {stats.losses[-1]:.6f}\n")
        f.write(f"Best Loss: {min(stats.losses):.6f} (Epoch {np.argmin(stats.losses)+1})\n")
        f.write(f"Final Pixel Loss: {stats.pixel_losses[-1]:.6f}\n")
        f.write(f"Final Compact Loss: {stats.compact_losses[-1]:.6f}\n")
        f.write(f"Final Separate Loss: {stats.separate_losses[-1]:.6f}\n")
        f.write(f"Average Epoch Time: {np.mean(stats.epoch_times):.2f}s\n")
        f.write(f"Total Training Time: {sum(stats.epoch_times):.2f}s\n\n")
        
        # Memory analysis
        if stats.memory_stats:
            final_mem_stats = stats.memory_stats[-1]
            f.write("MEMORY ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Active Memories: {final_mem_stats.get('active_memories', 0)}/{args.msize}\n")
            f.write(f"Memory Utilization: {final_mem_stats.get('active_memories', 0)/args.msize*100:.1f}%\n")
            f.write(f"Average Usage: {final_mem_stats.get('usage_mean', 0):.2f}\n")
            f.write(f"Usage Std: {final_mem_stats.get('usage_std', 0):.2f}\n")
            f.write(f"Max Usage: {final_mem_stats.get('usage_max', 0):.2f}\n\n")

def test_model_enhanced(args):
    """Enhanced testing with comprehensive evaluation metrics"""
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
    
    # Memory Module Diagnostics
    print("\n" + "="*60)
    print("MEMORY MODULE DIAGNOSTICS")
    print("="*60)
    print(f"Original Memory Shape: {m_items.shape}")
    print(f"Memory Values Range: {m_items.min().item():.4f} to {m_items.max().item():.4f}")
    print(f"Memory Mean: {m_items.mean().item():.4f}")
    print(f"Memory Std: {m_items.std().item():.4f}")
    print(f"Model has memory module: {hasattr(model, 'memory')}")
    if hasattr(model, 'memory'):
        print(f"Memory module type: {type(model.memory)}")
        print(f"Memory size config: {model.memory.memory_size}")
    print("="*60)
    
    # Load ground truth labels
    labels_path = f'./data/frame_labels_{args.dataset_type}.npy'
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        print(f"Warning: Labels file not found at {labels_path}")
        labels = np.zeros((1, test_size))  # Dummy labels

    # Setup video information
    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    
    print(f"Found {len(videos_list)} test videos")
    
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
    reconstruction_errors = []
    anomaly_scores_per_video = {}

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
        anomaly_scores_per_video[video_name] = []

    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    m_items_test = m_items.clone()

    # Testing with progress bar and memory tracking
    test_results = []
    memory_update_count = 0
    memory_usage_history = []
    feature_distances_raw = []
    
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
            
            m_items_before = m_items_test.clone()
            
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
            
            feature_distances_raw.append(mse_feas)
            memory_change = torch.mean(torch.abs(m_items_test - m_items_before)).item()
            
            # Dynamic threshold adjustment
            adaptive_threshold = max(args.th, point_sc * 0.5)
            if point_sc < adaptive_threshold:
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)
                m_items_before_update = m_items_test.clone()
                m_items_test = model.memory.update(query, m_items_test, False)
                memory_change_after_update = torch.mean(torch.abs(m_items_test - m_items_before_update)).item()
                memory_update_count += 1
            
            # Track memory statistics
            if hasattr(model, 'memory') and hasattr(model.memory, 'get_memory_stats'):
                mem_stats = model.memory.get_memory_stats()
                memory_usage_history.append({
                    'step': k,
                    'memory_change': memory_change,
                    'memory_updated': memory_change > 1e-6,
                    'point_score': point_sc,
                    'feature_distance': mse_feas,
                    **mem_stats
                })

            current_video = videos_list[min(video_num, len(videos_list)-1)].split('/')[-1]
            psnr_val = psnr(mse_imgs)
            psnr_list[current_video].append(psnr_val)
            feature_distance_list[current_video].append(mse_feas)
            reconstruction_errors.append(mse_imgs)
            
            pbar.set_postfix({
                'Video': f'{video_num+1}/{len(videos_list)}',
                'PSNR': f'{psnr_val:.2f}',
                'MSE': f'{mse_imgs:.4f}',
                'Feature': f'{mse_feas:.4f}',
                'MemUpdate': memory_update_count,
                'PtScore': f'{point_sc:.4f}'
            })

    # Memory Diagnostics Summary
    print(f"\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)
    print(f"Total memory updates: {memory_update_count}/{len(test_batch)}")
    print(f"Memory update rate: {memory_update_count/len(test_batch)*100:.1f}%")
    print(f"Feature distance range: {min(feature_distances_raw):.6f} - {max(feature_distances_raw):.6f}")
    print(f"Feature distance mean: {np.mean(feature_distances_raw):.6f}")
    print(f"Feature distance std: {np.std(feature_distances_raw):.6f}")
    
    if memory_usage_history:
        avg_mem_change = np.mean([h['memory_change'] for h in memory_usage_history])
        print(f"Average memory change per step: {avg_mem_change:.6f}")
        
        if avg_mem_change < 1e-6:
            print("⚠️  WARNING: Memory changes are very small - check memory update mechanism!")
        else:
            print("✅ Memory module appears to be working correctly")
    
    print("="*60)

    # Calculate final anomaly scores
    anomaly_score_total_list = []
    detailed_results = {}
    
    print("\nCalculating anomaly scores per video...")
    for video in tqdm(sorted(videos_list), desc="Processing videos"):
        video_name = video.split('/')[-1]
        
        video_psnr_scores = anomaly_score_list(psnr_list[video_name])
        video_feature_scores = anomaly_score_list_inv(feature_distance_list[video_name])
        video_combined_scores = score_sum(video_psnr_scores, video_feature_scores, args.alpha)
        
        anomaly_scores_per_video[video_name] = video_combined_scores
        anomaly_score_total_list += video_combined_scores
        
        detailed_results[video_name] = {
            'psnr_scores': video_psnr_scores,
            'feature_scores': video_feature_scores,
            'combined_scores': video_combined_scores,
            'avg_psnr': np.mean(psnr_list[video_name]),
            'std_psnr': np.std(psnr_list[video_name]),
            'avg_feature': np.mean(feature_distance_list[video_name]),
            'std_feature': np.std(feature_distance_list[video_name])
        }

    anomaly_score_total_list = np.asarray(anomaly_score_total_list)

    # Calculate comprehensive evaluation metrics
    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
    
    threshold = np.median(anomaly_score_total_list)
    predictions = (anomaly_score_total_list > threshold).astype(int)
    true_labels = (1-labels_list).astype(int)
    
    cm = confusion_matrix(true_labels, predictions)
    cr = classification_report(true_labels, predictions, output_dict=True)

    # Print comprehensive results
    print("\n" + "="*80)
    print("ANOMALY DETECTION RESULTS")
    print("="*80)
    print(f'Dataset: {args.dataset_type}')
    print(f'Method: {args.method}')
    print(f'AUC: {accuracy*100:.3f}%')
    print(f'Threshold: {threshold:.4f}')
    print(f'Precision: {cr["1"]["precision"]:.3f}')
    print(f'Recall: {cr["1"]["recall"]:.3f}')
    print(f'F1-Score: {cr["1"]["f1-score"]:.3f}')
    
    print(f'\nConfusion Matrix:')
    print(f'TN: {cm[0,0]}, FP: {cm[0,1]}')
    print(f'FN: {cm[1,0]}, TP: {cm[1,1]}')
    
    print(f'\nPer-Video Statistics:')
    print("-"*60)
    for video_name, results in detailed_results.items():
        print(f'{video_name}: PSNR={results["avg_psnr"]:.2f}±{results["std_psnr"]:.2f}, '
              f'Feature={results["avg_feature"]:.4f}±{results["std_feature"]:.4f}')
    
    # Save detailed memory analysis
    save_dir = os.path.join('./results', args.dataset_type, args.method)
    os.makedirs(save_dir, exist_ok=True)
    memory_analysis_path = os.path.join(save_dir, 'memory_analysis.json')
    if memory_usage_history:
        avg_mem_change = np.mean([h['memory_change'] for h in memory_usage_history])
    else:
        avg_mem_change = 0.0
    
    memory_analysis = {
        'total_updates': memory_update_count,
        'update_rate': memory_update_count/len(test_batch)*100,
        'feature_distances': feature_distances_raw,
        'avg_memory_change': avg_mem_change
    }
    
    with open(memory_analysis_path, 'w') as f:
        json.dump(memory_analysis, f, indent=2)
    
    # Enhanced results display
    print("\n" + "="*80)
    print("ENHANCED ANOMALY DETECTION RESULTS")
    print("="*80)
    print(f'Dataset: {args.dataset_type}')
    print(f'Method: {args.method}')
    print(f'AUC: {accuracy*100:.3f}%')
    print(f'Threshold: {threshold:.4f}')
    print(f'Precision: {cr["1"]["precision"]:.3f}')
    print(f'Recall: {cr["1"]["recall"]:.3f}')
    print(f'F1-Score: {cr["1"]["f1-score"]:.3f}')
    
    print(f'\nConfusion Matrix:')
    print(f'TN: {cm[0,0]}, FP: {cm[0,1]}')
    print(f'FN: {cm[1,0]}, TP: {cm[1,1]}')
    
    print(f'\nMemory Module Analysis:')
    print(f'Memory Updates: {memory_update_count}/{len(test_batch)} ({memory_update_count/len(test_batch)*100:.1f}%)')
    print(f'Feature Distance - Mean: {np.mean(feature_distances_raw):.6f}, Std: {np.std(feature_distances_raw):.6f}')
    print(f'Zero Feature Count: {np.sum(np.array(feature_distances_raw) == 0.0)}/{len(feature_distances_raw)}')
    
    if np.sum(np.array(feature_distances_raw) == 0.0) > len(feature_distances_raw) * 0.8:
        print("⚠️  WARNING: >80% of feature distances are zero - Memory module may not be working!")
    elif memory_update_count < len(test_batch) * 0.1:
        print("⚠️  WARNING: <10% memory updates - Check threshold value or memory mechanism!")
    else:
        print("✅ Memory module appears to be functioning normally")
    
    print(f'\nPer-Video Statistics:')
    print("-"*60)
    for video_name, results in detailed_results.items():
        anomaly_frames = np.sum(np.array(results['combined_scores']) > threshold)
        total_frames = len(results['combined_scores'])
        print(f'{video_name}: PSNR={results["avg_psnr"]:.2f}±{results["std_psnr"]:.2f}, '
              f'Feature={results["avg_feature"]:.4f}±{results["std_feature"]:.4f}, '
              f'Anomalies={anomaly_frames}/{total_frames}')
    
    # Generate and save visualization
    plot_evaluation_results(anomaly_score_total_list, labels_list, detailed_results, save_dir, args)
    plot_memory_analysis(memory_usage_history, feature_distances_raw, save_dir)
    
    # Save detailed results
    results_dict = {
        'auc': accuracy,
        'precision': cr["1"]["precision"],
        'recall': cr["1"]["recall"],
        'f1_score': cr["1"]["f1-score"],
        'confusion_matrix': cm.tolist(),
        'per_video_results': detailed_results,
        'anomaly_scores': anomaly_score_total_list.tolist(),
        'ground_truth': labels_list.tolist(),
        'memory_analysis': memory_analysis
    }
    
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nDetailed results saved in: {save_dir}")
    return accuracy

def plot_evaluation_results(anomaly_scores, labels, detailed_results, save_dir, args):
    """Plot comprehensive evaluation results"""
    plt.style.use('seaborn-v0_8')
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    axes[0,0].hist(anomaly_scores[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0,0].hist(anomaly_scores[labels == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
    axes[0,0].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(1-labels, anomaly_scores)
    auc_score = AUC(anomaly_scores, np.expand_dims(1-labels, 0))
    
    axes[0,1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(1-labels, anomaly_scores)
    ap_score = average_precision_score(1-labels, anomaly_scores)
    
    axes[0,2].plot(recall, precision, linewidth=2, label=f'PR (AP = {ap_score:.3f})')
    axes[0,2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    video_names = list(detailed_results.keys())[:10]
    avg_psnr = [detailed_results[v]['avg_psnr'] for v in video_names]
    
    axes[1,0].bar(range(len(video_names)), avg_psnr, color='skyblue')
    axes[1,0].set_title('Average PSNR per Video', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Video Index')
    axes[1,0].set_ylabel('PSNR')
    axes[1,0].set_xticks(range(len(video_names)))
    axes[1,0].set_xticklabels([f'V{i+1}' for i in range(len(video_names))], rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    timeline_length = min(1000, len(anomaly_scores))
    axes[1,1].plot(anomaly_scores[:timeline_length], color='blue', linewidth=1, label='Anomaly Score')
    
    for i in range(timeline_length):
        if labels[i] == 1:
            axes[1,1].axvspan(i-0.5, i+0.5, alpha=0.3, color='red')
    
    axes[1,1].set_title('Anomaly Score Timeline', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Frame Index')
    axes[1,1].set_ylabel('Anomaly Score')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    all_psnr = []
    all_features = []
    for video_results in detailed_results.values():
        all_psnr.extend([video_results['avg_psnr']])
        all_features.extend([video_results['avg_feature']])
    
    axes[1,2].scatter(all_features, all_psnr, alpha=0.6, color='green')
    axes[1,2].set_title('Feature Distance vs PSNR', fontsize=12, fontweight='bold')
    axes[1,2].set_xlabel('Average Feature Distance')
    axes[1,2].set_ylabel('Average PSNR')
    axes[1,2].grid(True, alpha=0.3)
    
    correlation = np.corrcoef(all_features, all_psnr)[0,1]
    axes[1,2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1,2].transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(detailed_results) <= 20:
        fig, ax = plt.subplots(figsize=(15, 8))
        
        max_length = max([len(results['combined_scores']) for results in detailed_results.values()])
        heatmap_data = []
        video_labels = []
        
        for video_name, results in detailed_results.items():
            scores = results['combined_scores']
            if len(scores) < max_length:
                scores.extend([0] * (max_length - len(scores)))
            else:
                scores = scores[:max_length]
            heatmap_data.append(scores)
            video_labels.append(video_name)
        
        sns.heatmap(heatmap_data, xticklabels=False, yticklabels=video_labels, 
                   cmap='viridis', cbar_kws={'label': 'Anomaly Score'})
        ax.set_title('Anomaly Scores Heatmap by Video', fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Video')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'anomaly_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_memory_analysis(memory_history, feature_distances, save_dir):
    """Plot comprehensive memory module analysis"""
    if not memory_history:
        print("No memory history data available for plotting")
        return
        
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data from memory history
    steps = [h['step'] for h in memory_history]
    active_memories = [h['active_memories'] for h in memory_history]
    usage_mean = [h['usage_mean'] for h in memory_history]
    memory_changes = [h['memory_change'] for h in memory_history]
    
    # Plot 1: Active Memories Over Time
    axes[0, 0].plot(steps, active_memories, color='blue', linewidth=2, label='Active Memories')
    axes[0, 0].set_title('Active Memory Items Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Test Batch')
    axes[0, 0].set_ylabel('Number of Active Memories')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Memory Usage
    axes[0, 1].plot(steps, usage_mean, color='green', linewidth=2, label='Average Usage')
    axes[0, 1].set_title('Average Memory Usage Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Test Batch')
    axes[0, 1].set_ylabel('Average Usage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Memory Change Magnitude
    axes[1, 0].plot(steps, memory_changes, color='red', linewidth=2, label='Memory Change')
    axes[1, 0].set_title('Memory Change Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Test Batch')
    axes[1, 0].set_ylabel('Memory Change (L1 Norm)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Feature Distance Distribution
    axes[1, 1].hist(feature_distances, bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title('Feature Distance Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Feature Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add summary statistics as text annotations
    avg_active = np.mean(active_memories)
    avg_change = np.mean(memory_changes)
    avg_usage = np.mean(usage_mean)
    
    axes[0, 0].text(0.05, 0.95, f'Avg Active: {avg_active:.1f}', 
                   transform=axes[0, 0].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[0, 1].text(0.05, 0.95, f'Avg Usage: {avg_usage:.3f}', 
                   transform=axes[0, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    axes[1, 0].text(0.05, 0.95, f'Avg Change: {avg_change:.6f}', 
                   transform=axes[1, 0].transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save memory analysis data
    memory_analysis_summary = {
        'avg_active_memories': float(avg_active),
        'avg_usage': float(avg_usage),
        'avg_memory_change': float(avg_change),
        'feature_distance_mean': float(np.mean(feature_distances)),
        'feature_distance_std': float(np.std(feature_distances)),
        'num_batches': len(steps)
    }
    
    with open(os.path.join(save_dir, 'memory_analysis_summary.json'), 'w') as f:
        json.dump(memory_analysis_summary, f, indent=2)

def get_train_args():
    """Enhanced training argument parser"""
    parser = argparse.ArgumentParser(description="MNAD Enhanced Training")
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

def get_test_args():
    """Enhanced testing argument parser"""  
    parser = argparse.ArgumentParser(description="MNAD Enhanced Testing")
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--method', type=str, default='pred', choices=['pred', 'recon'], help='The target task for anomaly detection')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomaly score')
    parser.add_argument('--th', type=float, default=0.01, help='threshold for test updating')
    parser.add_argument('--temp_update', type=float, default=0.1, help='temperature for memory update')
    parser.add_argument('--temp_gather', type=float, default=0.1, help='temperature for memory gather')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
    parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dataset_type', type=str, default='ped2', choices=['ped2', 'avenue', 'shanghai'], help='type of dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--model_dir', type=str, required=True, help='directory of model')
    parser.add_argument('--m_items_dir', type=str, required=True, help='directory of memory items')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_train_args()
    print("Starting MNAD training...")
    model, memory_items, stats = train_model(args)
    print("Training and testing completed!")


# python main.py --gpus 0 --batch_size 8 --epochs 3 --dataset_type avenue --dataset_path ./data --method pred --exp_dir log
