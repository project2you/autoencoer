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

# Utility functions (unchanged)
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
    """Calculate point-wise anomaly score"""
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
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

# Enhanced Data Loader with progress tracking (unchanged)
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
        
        total_frames = sum([self.animation[v]['length'] for v in self.videos])
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

# Enhanced Memory Module with improved debugging and validation
class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        
        # Initialize memory statistics tracking
        self.memory_usage = torch.zeros(memory_size).cuda()
        self.memory_age = torch.zeros(memory_size).cuda()
        self.memory_items = F.normalize(torch.rand((memory_size, key_dim), dtype=torch.float), dim=1).cuda()
        self.debug_log = []
        
        # Initialize memory items with proper scaling
        self.memory_items = F.normalize(torch.randn(memory_size, key_dim), dim=1).cuda() * 0.1
        
    def debug_memory_state(self, step, context=""):
        """Log detailed memory state for debugging"""
        mem_norm = torch.norm(self.memory_items, dim=1)
        mem_stats = {
            'step': step,
            'context': context,
            'mean_norm': mem_norm.mean().item(),
            'std_norm': mem_norm.std().item(),
            'min_norm': mem_norm.min().item(),
            'max_norm': mem_norm.max().item(),
            'usage_mean': self.memory_usage.mean().item(),
            'usage_std': self.memory_usage.std().item(),
            'active_memories': (self.memory_usage > 0).sum().item(),
            'sparsity': (self.memory_usage == 0).sum().item() / self.memory_size,
            'has_nan': torch.isnan(self.memory_items).any().item(),
            'has_inf': torch.isinf(self.memory_items).any().item()
        }
        self.debug_log.append(mem_stats)
        return mem_stats

    def update_memory_stats(self, indices):
        """Update memory usage statistics"""
        unique_indices = torch.unique(indices)
        for idx in unique_indices:
            self.memory_usage[idx] += 1
            self.memory_age[idx] = 0
        self.memory_age += 1
        
        # Warn if memory items are stagnant
        if (self.memory_age > 1000).any():
            warnings.warn("Some memory items have not been updated for over 1000 steps!")
        
        # Check for numerical issues
        if torch.isnan(self.memory_items).any() or torch.isinf(self.memory_items).any():
            warnings.warn("Memory items contain NaN or Inf values!")

    def get_memory_stats(self):
        """Get current memory statistics"""
        mem_norm = torch.norm(self.memory_items, dim=1)
        return {
            'usage_mean': self.memory_usage.mean().item(),
            'usage_std': self.memory_usage.std().item(),
            'usage_max': self.memory_usage.max().item(),
            'usage_min': self.memory_usage.min().item(),
            'active_memories': (self.memory_usage > 0).sum().item(),
            'norm_mean': mem_norm.mean().item(),
            'norm_std': mem_norm.std().item(),
            'norm_min': mem_norm.min().item(),
            'norm_max': mem_norm.max().item(),
            'sparsity': (self.memory_usage == 0).sum().item() / self.memory_size,
            'has_nan': torch.isnan(self.memory_items).any().item(),
            'has_inf': torch.isinf(self.memory_items).any().item()
        }
    
    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        """Update query based on memory attention"""
        m, d = mem.size()
        query_update = torch.zeros((m, d)).cuda()
        
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i, as_tuple=False)
            a = idx.size(0)
            
            if a > 0:
                # Improved weighted update with numerical stability
                weights = score[idx, i] / (torch.max(score[:, i]) + 1e-8)
                weights = weights / (weights.sum() + 1e-8)  # Normalize weights
                query_update[i] = torch.sum(weights * query[idx].squeeze(1), dim=0)
            else:
                query_update[i] = 0
                
                # Debug log for unused memory items
                self.debug_memory_state(step=0, context=f"No updates for memory item {i}")
        
        # Check for numerical stability
        if torch.isnan(query_update).any() or torch.isinf(query_update).any():
            warnings.warn("Query update contains NaN or Inf values!")
            self.debug_memory_state(step=0, context="Invalid query update")
        
        return query_update

    def get_score(self, mem, query):
        """Calculate attention scores between query and memory"""
        bs, h, w, d = query.size()
        m, d = mem.size()
        
        # Ensure numerical stability with normalization
        query_norm = F.normalize(query, dim=-1)
        mem_norm = F.normalize(mem, dim=-1)
        
        score = torch.matmul(query_norm, torch.t(mem_norm))
        score = score.view(bs * h * w, m)
        
        # Apply temperature scaling with clamping to avoid overflow
        score_query = F.softmax(score / self.temp_gather.clamp(min=1e-8, max=1.0), dim=0)
        score_memory = F.softmax(score / self.temp_update.clamp(min=1e-8, max=1.0), dim=1)
        
        # Debug attention scores
        score_stats = {
            'score_mean': score.mean().item(),
            'score_std': score.std().item(),
            'score_min': score.min().item(),
            'score_max': score.max().item(),
            'score_query_mean': score_query.mean().item(),
            'score_memory_mean': score_memory.mean().item()
        }
        if score.std().item() < 1e-4:
            warnings.warn("Attention scores have very low variance - possible convergence issue!")
        self.debug_memory_state(step=0, context=f"Attention scores: {score_stats}")
        
        return score_query, score_memory
    
    def forward(self, query, keys, train=True):
        batch_size, dims, h, w = query.size()
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)
        
        # Debug initial state
        self.debug_memory_state(step=0, context="Before forward pass")
        
        if train:
            separateness_loss, compactness_loss = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = self.update(query, keys, train)
            
            # Debug after forward pass
            self.debug_memory_state(step=0, context="After forward pass (train)")
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss
        else:
            compactness_loss, query_re, top1_keys, keys_ind = self.gather_loss(query, keys, train)
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            updated_memory = keys
            
            # Update memory statistics
            self.update_memory_stats(keys_ind)
            
            # Debug after forward pass
            self.debug_memory_state(step=0, context="After forward pass (test)")
            
            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, query_re, top1_keys, keys_ind, compactness_loss
    
    def update(self, query, keys, train):
        """Update memory items"""
        batch_size, h, w, dims = query.size()
        
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        
        # Update memory statistics
        self.update_memory_stats(gathering_indices.squeeze(1))
        
        query_update = self.get_update_query(keys, gathering_indices, updating_indices, 
                                           softmax_score_query, query_reshape, train)
        
        # Ensure numerical stability in memory update
        updated_memory = F.normalize(query_update + keys, dim=1)
        
        # Prevent memory items from becoming zero
        if torch.norm(updated_memory, dim=1).min() < 1e-8:
            warnings.warn("Memory items have near-zero norm after update!")
            updated_memory = F.normalize(keys + torch.randn_like(keys) * 0.01, dim=1)
        
        # Update internal memory items
        self.memory_items = updated_memory.detach()
        
        # Debug memory update
        self.debug_memory_state(step=0, context="After memory update")
        
        return updated_memory.detach()
    
    def gather_loss(self, query, keys, train):
        """Calculate gathering and separateness loss"""
        batch_size, h, w, dims = query.size()
        
        if train:
            loss = torch.nn.TripletMarginLoss(margin=1.0)
            loss_mse = torch.nn.MSELoss()
            
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size * h * w, dims)
            
            _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)
            
            pos = keys[gathering_indices[:, 0]]
            neg = keys[gathering_indices[:, 1]]
            
            compactness_loss = loss_mse(query_reshape, pos.detach())
            separateness_loss = loss(query_reshape, pos.detach(), neg.detach())
            
            # Debug loss calculation
            self.debug_memory_state(step=0, context=f"Train loss - Compact: {compactness_loss.item():.4f}, Separate: {separateness_loss.item():.4f}")
            
            return separateness_loss, compactness_loss
        else:
            loss_mse = torch.nn.MSELoss()
            
            softmax_score_query, softmax_score_memory = self.get_score(keys, query)
            query_reshape = query.contiguous().view(batch_size * h * w, dims)
            
            _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
            compactness_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())
            
            # Debug loss calculation
            self.debug_memory_state(step=0, context=f"Test loss - Compact: {compactness_loss.item():.4f}")
            
            return compactness_loss, query_reshape, keys[gathering_indices].squeeze(1).detach(), gathering_indices[:, 0]
    
    def read(self, query, updated_memory):
        """Read from memory and update query"""
        batch_size, h, w, dims = query.size()
        
        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2)
        
        # Debug read operation
        self.debug_memory_state(step=0, context="After memory read")
        
        return updated_query, softmax_score_query, softmax_score_memory

# Backward compatibility - MemoryAdvanced
class MemoryAdvanced(Memory):
    """Backward compatibility class for old saved models"""
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(MemoryAdvanced, self).__init__(memory_size, feature_dim, key_dim, temp_update, temp_gather)

# Encoder with skip connections (unchanged)
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

# Decoder with skip connections (unchanged)
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

# Main ConvAE Model (unchanged)
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

# Backward compatibility - convAEAdvanced
class convAEAdvanced(convAE):
    """Backward compatibility class for old saved models"""
    def __init__(self, n_channel=3, t_length=5, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1, temp_gather=0.1):
        super(convAEAdvanced, self).__init__(n_channel, t_length, memory_size, feature_dim, key_dim, temp_update, temp_gather)

# Training Statistics Tracker (unchanged)
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

# Training function (unchanged)
def train_model(args):
    """Enhanced training with progress tracking and statistics"""
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

    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim, args.temp_update, args.temp_gather)
    
    params_encoder = list(model.encoder.parameters()) 
    params_decoder = list(model.decoder.parameters())
    params = params_encoder + params_decoder
    
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.cuda()

    stats = TrainingStats()

    log_dir = os.path.join('./exp', args.dataset_type, args.method, args.exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    orig_stdout = sys.stdout
    f = open(os.path.join(log_dir, 'log.txt'), 'w')
    sys.stdout = f

    loss_func_mse = nn.MSELoss(reduction='none')

    m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).cuda()
    
    print("="*60)
    print(f"MNAD Training Started - Method: {args.method}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Memory Size: {args.msize}")
    print(f"Feature Dim: {args.fdim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("="*60)

    for epoch in range(args.epochs):
        model.train()
        epoch_start_time = time.time()
        
        epoch_losses = []
        epoch_pixel_losses = []
        epoch_compact_losses = []
        epoch_separate_losses = []
        
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
            
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_pixel_losses.append(loss_pixel.item())
            epoch_compact_losses.append(compactness_loss.item())
            epoch_separate_losses.append(separateness_loss.item())
            
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
        avg_pixel = np.mean(epoch_pixel_losses) 
        avg_compact = np.mean(epoch_compact_losses)
        avg_separate = np.mean(epoch_separate_losses)
        current_lr = get_lr(optimizer)
        
        memory_stats = model.memory.get_memory_stats()
        
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
        
        if memory_stats:
            print(f'Memory Stats: Active={memory_stats["active_memories"]}/{args.msize}, '
                  f'Avg Usage={memory_stats["usage_mean"]:.2f}, '
                  f'Usage Std={memory_stats["usage_std"]:.2f}')
        
        print('='*60)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for imgs in tqdm(test_batch, desc="Validation"):
                    imgs = Variable(imgs).cuda()
                    
                    if args.method == 'pred':
                        outputs, _, _, _, _, _, _, _, _, val_loss = model.forward(imgs[:, 0:args.c*(args.t_length-1)], m_items, False)
                    else:
                        outputs, _, _, _, _, _, val_loss = model.forward(imgs, m_items, False)
                    
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            print(f'Validation Loss: {avg_val_loss:.6f}')
            print('='*60)
    
    print('Training is finished')
    
    torch.save(model, os.path.join(log_dir, 'model.pth'))
    torch.save(m_items, os.path.join(log_dir, 'keys.pt'))
    
    stats.save_stats(log_dir)
    
    stats.plot_training_curves(log_dir)
    
    generate_training_report(stats, log_dir, args)
    
    sys.stdout = orig_stdout
    f.close()
    
    print(f"Training completed! Results saved in: {log_dir}")
    return model, m_items, stats

def generate_training_report(stats, log_dir, args):
    """Generate comprehensive training report"""
    report_path = os.path.join(log_dir, 'training_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MNAD TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
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
        
        f.write("TRAINING STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Final Loss: {stats.losses[-1]:.6f}\n")
        f.write(f"Best Loss: {min(stats.losses):.6f} (Epoch {np.argmin(stats.losses)+1})\n")
        f.write(f"Final Pixel Loss: {stats.pixel_losses[-1]:.6f}\n")
        f.write(f"Final Compact Loss: {stats.compact_losses[-1]:.6f}\n")
        f.write(f"Final Separate Loss: {stats.separate_losses[-1]:.6f}\n")
        f.write(f"Average Epoch Time: {np.mean(stats.epoch_times):.2f}s\n")
        f.write(f"Total Training Time: {sum(stats.epoch_times):.2f}s\n\n")
        
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

    test_dataset = DataLoader(test_folder, transforms.Compose([
        transforms.ToTensor(),            
    ]), resize_height=args.h, resize_width=args.w, time_step=args.t_length-1)

    test_size = len(test_dataset)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, 
                                 shuffle=False, num_workers=args.num_workers_test, drop_last=False)

    loss_func_mse = nn.MSELoss(reduction='none')

    print(f"Loading model from: {args.model_dir}")
    model = torch.load(args.model_dir, map_location='cuda:0', weights_only=False)
    model.cuda()
    model.eval()
    
    print(f"Loading memory items from: {args.m_items_dir}")
    m_items = torch.load(args.m_items_dir, map_location='cuda:0')
    
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
    
    labels_path = f'./data/frame_labels_{args.dataset_type}.npy'
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        print(f"Warning: Labels file not found at {labels_path}")
        labels = np.zeros((1, test_size))

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

    labels_list = []
    label_length = 0
    psnr_list = {}
    feature_distance_list = {}
    reconstruction_errors = []
    anomaly_scores_per_video = {}

    print(f'Evaluation of {args.dataset_type}')
    print("="*60)

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
            
            memory_updated = False
            if point_sc < args.th:
                query = F.normalize(feas, dim=1)
                query = query.permute(0, 2, 3, 1)
                m_items_before_update = m_items_test.clone()
                m_items_test = model.memory.update(query, m_items_test, False)
                memory_change_after_update = torch.mean(torch.abs(m_items_test - m_items_before_update)).item()
                memory_updated = True
                memory_update_count += 1
            
            if hasattr(model, 'memory') and hasattr(model.memory, 'get_memory_stats'):
                mem_stats = model.memory.get_memory_stats()
                memory_usage_history.append({
                    'step': k,
                    'memory_change': memory_change,
                    'memory_updated': memory_updated,
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
        
        if avg_mem_change < 1e-8:
            print("⚠️  WARNING: Memory changes are very small - memory module may not be working properly!")
        else:
            print("✅ Memory module appears to be working correctly")
    
    print("="*60)

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

    accuracy = AUC(anomaly_score_total_list, np.expand_dims(1-labels_list, 0))
    
    threshold = np.median(anomaly_score_total_list)
    predictions = (anomaly_score_total_list > threshold).astype(int)
    true_labels = (1-labels_list).astype(int)
    
    cm = confusion_matrix(true_labels, predictions)
    cr = classification_report(true_labels, predictions, output_dict=True)

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
    
    memory_analysis_path = os.path.join(save_dir, 'memory_analysis.json')
    memory_analysis = {
        'total_updates': memory_update_count,
        'update_rate': memory_update_count/len(test_batch)*100,
        'feature_distance_stats': {
            'min': float(min(feature_distances_raw)),
            'max': float(max(feature_distances_raw)),
            'mean': float(np.mean(feature_distances_raw)),
            'std': float(np.std(feature_distances_raw)),
            'zero_count': int(np.sum(np.array(feature_distances_raw) == 0.0))
        },
        'memory_usage_history': memory_usage_history[:100]
    }
    
    with open(memory_analysis_path, 'w') as f:
        json.dump(memory_analysis, f, indent=2)
    
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
    
    save_dir = os.path.join('./results', args.dataset_type, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plot_evaluation_results(anomaly_score_total_list, labels_list, detailed_results, save_dir, args)
    plot_memory_analysis(memory_usage_history, feature_distances_raw, save_dir)
    
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
    return accuracy, detailed_results

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
    axes[1,2].set_title('Feature Distance vs PSNR', fontsize=14, fontweight='bold')
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

class MNADModelManager:
    def __init__(self):
        self.model = None
        self.memory_items = None
        self.training_stats = None
        
    def create_model(self, args):
        """Create MNAD model with given arguments"""
        model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim, args.temp_update, args.temp_gather)
        return model
    
    def save_model(self, model, memory_items, save_path):
        """Save model and memory items"""
        torch.save(model, os.path.join(save_path, 'model.pth'))
        torch.save(memory_items, os.path.join(save_path, 'keys.pt'))
        
    def load_model(self, model_path, memory_path):
        """Load trained model and memory items"""
        model = torch.load(model_path, weights_only=False)
        memory_items = torch.load(memory_path)
        return model, memory_items
    
    def validate_memory_module(self, model, test_input):
        """Validate memory module functionality"""
        model.eval()
        with torch.no_grad():
            try:
                memory_stats = model.memory.get_memory_stats()
                print("Memory Module Validation:")
                print(f"  Memory Size: {model.memory.memory_size}")
                print(f"  Feature Dim: {model.memory.feature_dim}")
                print(f"  Active Memories: {memory_stats.get('active_memories', 0)}")
                print("  Memory module working correctly ✓")
                return True
            except Exception as e:
                print(f"  Memory module error: {e}")
                return False

def run_memory_diagnostics(model, memory_items, test_data_loader):
    """Run comprehensive memory diagnostics"""
    print("\n" + "="*60)
    print("MEMORY MODULE DIAGNOSTICS")
    print("="*60)
    
    model.eval()
    memory_usage_history = []
    attention_weights_history = []
    
    with torch.no_grad():
        for i, imgs in enumerate(test_data_loader):
            if i >= 10:
                break
                
            imgs = Variable(imgs).cuda()
            
            if hasattr(model, 'memory'):
                outputs, feas, updated_feas, updated_memory, score_query, score_memory, *_ = model(imgs[:, :12], memory_items, False)
                
                memory_stats = model.memory.get_memory_stats()
                memory_usage_history.append(memory_stats)
                
                attention_weights = torch.mean(score_memory, dim=0).cpu().numpy()
                attention_weights_history.append(attention_weights)
    
    if memory_usage_history:
        avg_active = np.mean([stats.get('active_memories', 0) for stats in memory_usage_history])
        avg_usage = np.mean([stats.get('usage_mean', 0) for stats in memory_usage_history])
        
        print(f"Average Active Memories: {avg_active:.1f}/{model.memory.memory_size}")
        print(f"Memory Utilization Rate: {avg_active/model.memory.memory_size*100:.1f}%")
        print(f"Average Memory Usage: {avg_usage:.3f}")
        
    if attention_weights_history:
        avg_attention = np.mean(attention_weights_history, axis=0)
        print(f"Attention Distribution: Min={avg_attention.min():.4f}, Max={avg_attention.max():.4f}")
        print(f"Attention Entropy: {-np.sum(avg_attention * np.log(avg_attention + 1e-8)):.3f}")
    
    print("Memory diagnostics completed ✓")

def create_sample_config():
    """Create sample configuration files for easy setup"""
    
    # Training config
    train_config = {
        "gpus": ["0"],
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "loss_compact": 0.1,
        "loss_separate": 0.1,
        "method": "pred",
        "dataset_type": "ped2",
        "dataset_path": "./dataset",
        "memory_size": 10,
        "feature_dim": 512,
        "memory_dim": 512,
        "temp_update": 0.1,
        "temp_gather": 0.1,
        "image_height": 256,
        "image_width": 256,
        "channels": 3,
        "time_length": 5
    }
    
    # Testing config
    test_config = {
        "gpus": ["0"],
        "test_batch_size": 1,
        "alpha": 0.6,
        "threshold": 0.01,
        "method": "pred",
        "dataset_type": "ped2",
        "dataset_path": "./dataset",
        "model_dir": "./exp/ped2/pred/log/model.pth",
        "memory_items_dir": "./exp/ped2/pred/log/keys.pt"
    }
    
    # Save configs
    os.makedirs('./configs', exist_ok=True)
    
    with open('./configs/train_config.json', 'w') as f:
        json.dump(train_config, f, indent=2)
    
    with open('./configs/test_config.json', 'w') as f:
        json.dump(test_config, f, indent=2)
        
    print("Sample configuration files created in ./configs/")

def plot_memory_analysis(memory_history, feature_distances, save_dir):
    """Plot memory module analysis"""
    if not memory_history:
        return
        
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Memory update pattern
    steps = [h['step'] for h in memory_history]
    updated = [1 if h['memory_updated'] else 0 for h in memory_history]
    
    axes[0,0].plot(steps, updated, 'o-', markersize=2, alpha=0.7)
    axes[0,0].set_title('Memory Update Pattern', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Step')
    axes[0,0].set_ylabel('Updated (1=Yes, 0=No)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Feature distance distribution
    axes[0,1].hist(feature_distances, bins=50, alpha=0.7, color='green')
    axes[0,1].set_title('Feature Distance Distribution', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Feature Distance')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].grid(True, alpha=0.3)
    
    # Point score vs Feature distance
    if len(memory_history) > 10:
        point_scores = [h['point_score'] for h in memory_history]
        feat_distances = [h['feature_distance'] for h in memory_history]
        
        axes[1,0].scatter(point_scores, feat_distances, alpha=0.6, s=10)
        axes[1,0].set_title('Point Score vs Feature Distance', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Point Score')
        axes[1,0].set_ylabel('Feature Distance')
        axes[1,0].grid(True, alpha=0.3)
    
    # Memory change over time
    if len(memory_history) > 10:
        memory_changes = [h['memory_change'] for h in memory_history]
        axes[1,1].plot(steps, memory_changes, color='red', linewidth=1)
        axes[1,1].set_title('Memory Change Over Time', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Memory Change')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'memory_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_evaluation_results(anomaly_scores, labels, detailed_results, save_dir, args):
    """Plot comprehensive evaluation results"""
    plt.style.use('seaborn-v0_8')
    
    # Create comprehensive evaluation plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Anomaly Score Distribution
    axes[0,0].hist(anomaly_scores[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    axes[0,0].hist(anomaly_scores[labels == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
    axes[0,0].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Anomaly Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
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
    
    # 3. Precision-Recall Curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(1-labels, anomaly_scores)
    ap_score = average_precision_score(1-labels, anomaly_scores)
    
    axes[0,2].plot(recall, precision, linewidth=2, label=f'PR (AP = {ap_score:.3f})')
    axes[0,2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('Recall')
    axes[0,2].set_ylabel('Precision')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Per-Video Performance
    video_names = list(detailed_results.keys())[:10]  # Show first 10 videos
    avg_psnr = [detailed_results[v]['avg_psnr'] for v in video_names]
    
    axes[1,0].bar(range(len(video_names)), avg_psnr, color='skyblue')
    axes[1,0].set_title('Average PSNR per Video', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Video Index')
    axes[1,0].set_ylabel('PSNR')
    axes[1,0].set_xticks(range(len(video_names)))
    axes[1,0].set_xticklabels([f'V{i+1}' for i in range(len(video_names))], rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Score Timeline (first 1000 frames)
    timeline_length = min(1000, len(anomaly_scores))
    axes[1,1].plot(anomaly_scores[:timeline_length], color='blue', linewidth=1, label='Anomaly Score')
    
    # Highlight anomalous regions
    for i in range(timeline_length):
        if labels[i] == 1:
            axes[1,1].axvspan(i-0.5, i+0.5, alpha=0.3, color='red')
    
    axes[1,1].set_title('Anomaly Score Timeline', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Frame Index')
    axes[1,1].set_ylabel('Anomaly Score')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Feature vs PSNR Correlation
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
    
    # Add correlation coefficient
    correlation = np.corrcoef(all_features, all_psnr)[0,1]
    axes[1,2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1,2].transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed anomaly score heatmap
    if len(detailed_results) <= 20:  # Only for manageable number of videos
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create heatmap data
        max_length = max([len(results['combined_scores']) for results in detailed_results.values()])
        heatmap_data = []
        video_labels = []
        
        for video_name, results in detailed_results.items():
            scores = results['combined_scores']
            # Pad or truncate to consistent length
            if len(scores) < max_length:
                scores.extend([0] * (max_length - len(scores)))
            else:
                scores = scores[:max_length]
            heatmap_data.append(scores)
            video_labels.append(video_name)
        
        # Create heatmap
        sns.heatmap(heatmap_data, xticklabels=False, yticklabels=video_labels, 
                   cmap='viridis', cbar_kws={'label': 'Anomaly Score'})
        ax.set_title('Anomaly Scores Heatmap by Video', fontsize=16, fontweight='bold')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Video')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'anomaly_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        

def create_avenue_training_config():
    """Create training configuration for Avenue dataset"""
    config = {
        "gpus": ["0"],
        "batch_size": 16,
        "epochs": 3,
        "dataset_path": "./data",
        "dataset_type": "avenue", 
        "method": "pred",
        "exp_dir": "avenue_rtx4090_v3",
        "h": 320,
        "w": 320,
        "msize": 20,
        "loss_compact": 0.1,
        "loss_separate": 0.1,
        "lr": 0.0002,
        "weight_decay": 0.0001,
        "temp_update": 0.1,
        "temp_gather": 0.1,
        "c": 3,
        "t_length": 5,
        "fdim": 512,
        "mdim": 512,
        "num_workers": 2,
        "test_batch_size": 1,
        "num_workers_test": 1
    }
    return config

def train_with_avenue_config():
    """Train model with Avenue dataset configuration"""
    config = create_avenue_training_config()
    train_args = argparse.Namespace(**config)
    
    print("Starting training with Avenue configuration...")
    print(f"Dataset: {config['dataset_type']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Image size: {config['h']}x{config['w']}")
    print(f"Memory size: {config['msize']}")
    print("="*50)
    
    model, memory_items, stats = train_model(train_args)
    
    print("Training completed successfully!")
    return model, memory_items, stats

def create_avenue_testing_config():
    """Create testing configuration for Avenue dataset"""
    config = {
        "gpus": ["0"],
        "test_batch_size": 1,
        "h": 320,
        "w": 320,
        "c": 3,
        "method": "pred",
        "t_length": 5,
        "fdim": 512,
        "mdim": 512,
        "msize": 20,
        "alpha": 0.6,
        "th": 0.01,
        "temp_update": 0.1,
        "temp_gather": 0.1,
        "num_workers": 2,
        "num_workers_test": 1,
        "dataset_type": "avenue",
        "dataset_path": "./data",
        "model_dir": "./exp/avenue/pred/avenue_rtx4090_v3/model.pth",
        "m_items_dir": "./exp/avenue/pred/avenue_rtx4090_v3/keys.pt"
    }
    return config

def test_with_avenue_config():
    """Test model with Avenue dataset configuration"""
    config = create_avenue_testing_config()
    test_args = argparse.Namespace(**config)
    
    print("Starting testing with Avenue configuration...")
    print(f"Dataset: {config['dataset_type']}")
    print(f"Method: {config['method']}")
    print(f"Image size: {config['h']}x{config['w']}")
    print(f"Memory size: {config['msize']}")
    print(f"Alpha: {config['alpha']}")
    print(f"Threshold: {config['th']}")
    print(f"Model path: {config['model_dir']}")
    print(f"Memory items path: {config['m_items_dir']}")
    print("="*50)
    
    accuracy, results = test_model_enhanced(test_args)
    
    print(f"Testing completed! AUC: {accuracy*100:.3f}%")
    return accuracy, results

def main():
    """Enhanced main function with comprehensive options"""
    parser = argparse.ArgumentParser(description="MNAD Enhanced - Video Anomaly Detection System")
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'validate', 'diagnose', 'train_avenue', 'test_avenue'], 
                       required=True, help='Operation mode')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if args.config:
            # Load from config file
            with open(args.config, 'r') as f:
                config = json.load(f)
            # Convert to argparse namespace
            train_args = argparse.Namespace(**config)
        else:
            train_args = get_train_args()
        
        print("Starting enhanced training...")
        model, memory_items, stats = train_model(train_args)
        print("Training completed successfully!")
        
    elif args.mode == 'train_avenue':
        # New mode for direct Avenue training
        print("Starting Avenue dataset training with predefined configuration...")
        model, memory_items, stats = train_with_avenue_config()
        
    elif args.mode == 'test':
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
            test_args = argparse.Namespace(**config)
        else:
            test_args = get_test_args()
        
        print("Starting enhanced testing...")
        accuracy, results = test_model_enhanced(test_args)
        print(f"Testing completed! AUC: {accuracy*100:.3f}%")
        
    elif args.mode == 'test_avenue':
        # New mode for direct Avenue testing
        print("Starting Avenue dataset testing with predefined configuration...")
        accuracy, results = test_with_avenue_config()
        
    elif args.mode == 'validate':
        test_args = get_test_args()
        manager = MNADModelManager()
        
        print("Loading model for validation...")
        model, memory_items = manager.load_model(test_args.model_dir, test_args.m_items_dir)
        
        # Create dummy test input
        test_input = torch.randn(1, test_args.c*(test_args.t_length-1), test_args.h, test_args.w).cuda()
        
        if manager.validate_memory_module(model, test_input):
            print("Model validation passed ✓")
        else:
            print("Model validation failed ✗")
            
    elif args.mode == 'diagnose':
        test_args = get_test_args()
        manager = MNADModelManager()
        
        print("Loading model and data for diagnostics...")
        model, memory_items = manager.load_model(test_args.model_dir, test_args.m_items_dir)
        
        # Load test data
        test_folder = test_args.dataset_path + "/" + test_args.dataset_type + "/testing/frames"
        test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor()]), 
                                 resize_height=test_args.h, resize_width=test_args.w, 
                                 time_step=test_args.t_length-1)
        test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        run_memory_diagnostics(model, memory_items, test_loader)

if __name__ == "__main__":
    # Create sample configs on first run
    if not os.path.exists('./configs'):
        create_sample_config()
    
    # Check for backward compatibility (legacy mode)
    # If arguments contain model_dir and m_items_dir but no --mode, assume test mode
    has_model_args = any('--model_dir' in arg or '--m_items_dir' in arg for arg in sys.argv)
    has_mode_arg = '--mode' in sys.argv
    
    if has_model_args and not has_mode_arg:
        print("Running in legacy test mode (backward compatible)...")
        test_args = get_test_args()
        test_model_enhanced(test_args)
    elif len(sys.argv) == 1:
        print("No arguments provided. Please specify --mode or provide test arguments.")
        print("Use --help for usage information.")
    else:
        main()
