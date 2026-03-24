import os
import argparse
import random
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import torchvision.transforms.functional as TF
import lmdb
import cv2
import datetime

class AugmentedSubset(Dataset):
    """包装Subset，支持数据增强"""
    def __init__(self, subset, augmentation=False):
        self.subset = subset
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        data = self.subset[idx]
        if self.augmentation:
            input_tensor, tgt_tensor, v_mask = data
            if random.random() > 0.5:
                input_tensor = TF.hflip(input_tensor)
                tgt_tensor = TF.hflip(tgt_tensor)
                v_mask = TF.hflip(v_mask)
            if random.random() > 0.5:
                input_tensor = TF.vflip(input_tensor)
                tgt_tensor = TF.vflip(tgt_tensor)
                v_mask = TF.vflip(v_mask)
            return input_tensor, tgt_tensor, v_mask
        return data

# ================= 配置 =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    # ResNet模型较大，显存如果不够请降到 32
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--workers', type=int, default=16,
                        help='数据加载进程数，建议设为CPU核心数，过多可能变慢')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='每个worker预取batch数，减少可节省内存但可能变慢')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.0018)
    
    # 性能优化参数
    parser.add_argument('--compile', action='store_true',
                        help='使用torch.compile加速模型（PyTorch 2.0+）')
    parser.add_argument('--disable_persistent_workers', action='store_true',
                        help='禁用persistent_workers以节省内存（可能稍慢）')
    parser.add_argument('--drop_last', action='store_true',
                        help='丢弃最后一个不完整的batch，保持batch大小一致')
    
    # 路径
    parser.add_argument('--lmdb_path', type=str, default='/home/hjx/multipath-2d/dataset.lmdb')
    # 原始路径用于读取 building_map.png 等静态图
    parser.add_argument('--raw_root', type=str, default='/home/hjx/multipath-2d')
    
    # 验证集切分比例
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                        help='从训练集中切分验证集的比例')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径 (path to checkpoint to resume from)')
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
DEVICE = torch.device("cuda")

# 日志
SAVE_DIR = os.path.join(args.raw_root, "checkpoints_super_fft")
os.makedirs(SAVE_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(SAVE_DIR, f"train_super_{timestamp}.log")

def log(msg):
    print(msg)
    with open(LOG_PATH, 'a+') as f: f.write(msg + '\n')

# 地图划分：合并训练和验证集，从内部按比例分割
TRAIN_VAL_MAP_IDS = set(range(0, 631))  # map 0-630: 训练+验证集（合并）
TEST_MAP_IDS = set(range(631, 701))      # map 631-700: 测试集

# ================= 辅助函数 =================
def get_all_map_ids(lmdb_path):
    """自动检测 LMDB 中所有可用的 map_id"""
    temp_env = lmdb.open(lmdb_path, subdir=True, readonly=True, lock=False, readahead=True, meminit=False)
    total_length = int(temp_env.begin().get("length".encode("ascii")).decode("ascii"))
    
    map_ids = set()
    with temp_env.begin() as txn:
        cursor = txn.cursor()
        for idx in tqdm(range(total_length), desc="检测所有 map_id", leave=False):
            byte_data = cursor.get(f"{idx:08}".encode("ascii"))
            if byte_data is not None:
                try:
                    sample = pickle.loads(byte_data)
                    map_id = sample.get('map_id')
                    if map_id is not None:
                        map_ids.add(map_id)
                except:
                    continue
    
    temp_env.close()
    return sorted(map_ids)

# ================= Dataset (LMDB) =================
class LMDBDataset(Dataset):
    _static_cache = {}

    def __init__(self, lmdb_path, raw_root, valid_map_ids=None, augmentation=False):
        self.lmdb_path = lmdb_path
        self.raw_root = raw_root
        self.valid_map_ids = valid_map_ids  # None 表示加载所有数据
        self.augmentation = augmentation
        self.env = None
        
        # 建立索引：如果 valid_map_ids 为 None，则加载所有数据
        temp_env = lmdb.open(lmdb_path, subdir=True, readonly=True, lock=False, readahead=True, meminit=False)
        total_length = int(temp_env.begin().get("length".encode("ascii")).decode("ascii"))
        
        self.valid_indices = []
        from tqdm import tqdm
        
        # 统计信息
        found_map_ids = set()
        processed_count = 0
        
        # 简化进度条描述：只显示地图数量或范围，不显示所有map_id
        if valid_map_ids is None:
            desc = "建立索引(所有数据)"
        else:
            sorted_ids = sorted(valid_map_ids)
            if len(sorted_ids) <= 10:
                # 如果地图数量少，显示范围
                desc = f"建立索引(map {min(sorted_ids)}-{max(sorted_ids)})"
            else:
                # 如果地图数量多，只显示数量
                desc = f"建立索引({len(valid_map_ids)} maps)"
        
        with temp_env.begin() as txn:
            cursor = txn.cursor()
            pbar = tqdm(range(total_length), desc=desc, leave=True, 
                       unit='样本', unit_scale=True, ncols=120, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for idx in pbar:
                byte_data = cursor.get(f"{idx:08}".encode("ascii"))
                if byte_data is not None:
                    try:
                        sample = pickle.loads(byte_data)
                        map_id = sample.get('map_id')
                        # 如果 valid_map_ids 为 None，或者 map_id 在 valid_map_ids 中，则包含该样本
                        if valid_map_ids is None or map_id in valid_map_ids:
                            self.valid_indices.append(idx)
                            if map_id is not None:
                                found_map_ids.add(map_id)
                        processed_count += 1
                        
                        # 每处理10%的数据，更新一次进度条信息
                        if processed_count % max(1, total_length // 10) == 0:
                            pbar.set_postfix({
                                '有效': f'{len(self.valid_indices):,}',
                                '地图': len(found_map_ids)
                            })
                    except Exception as e:
                        processed_count += 1
                        continue
            
            # 最终更新一次进度条
            pbar.set_postfix({
                '有效样本': f'{len(self.valid_indices):,}',
                '地图数': len(found_map_ids)
            })
            pbar.close()
        
        self.length = len(self.valid_indices)
        temp_env.close()

    def _init_env(self):
        if self.env is None:
            # 优化：启用readahead加速读取，map_async提升并发性能
            self.env = lmdb.open(self.lmdb_path, subdir=True, readonly=True, lock=False, 
                                readahead=True, meminit=False, map_async=True)

    def _get_static(self, map_id):
        if map_id in self._static_cache: return self._static_cache[map_id]
        try:
            p = os.path.join(self.raw_root, f"map_{map_id}", "preprocessed_data")
            # 使用 cv2 读取静态图
            b_img = cv2.imread(os.path.join(p, "building_map.png"), cv2.IMREAD_GRAYSCALE)
            v_img = cv2.imread(os.path.join(p, "vertex_mask.png"), cv2.IMREAD_GRAYSCALE)
            b_ts = torch.from_numpy(b_img).float().unsqueeze(0) / 255.0
            v_ts = torch.from_numpy(v_img).float().unsqueeze(0) / 255.0
            self._static_cache[map_id] = (b_ts, v_ts)
            return b_ts, v_ts
        except:
            z = torch.zeros(1, 256, 256)
            return z, z

    def __len__(self): return self.length

    def __getitem__(self, idx):
        self._init_env()
        # 使用预先建立的索引，直接访问有效样本
        actual_idx = self.valid_indices[idx]
        
        with self.env.begin(write=False) as txn:
            byte_data = txn.get(f"{actual_idx:08}".encode("ascii"))
        
        if byte_data is None:
            raise IndexError(f"索引 {actual_idx} 对应的数据不存在")
        
        sample = pickle.loads(byte_data)
        
        # 解码
        src_img = cv2.imdecode(np.frombuffer(sample['src'], np.uint8), cv2.IMREAD_GRAYSCALE)
        tgt_img = cv2.imdecode(np.frombuffer(sample['tgt'], np.uint8), cv2.IMREAD_COLOR)
        tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

        tx_tensor = torch.from_numpy(src_img).float().unsqueeze(0) / 255.0
        tgt_tensor = torch.from_numpy(tgt_img).permute(2,0,1).float() / 255.0
        
        b_map, v_mask = self._get_static(sample['map_id'])
        input_tensor = torch.cat([b_map, tx_tensor], dim=0)

        if self.augmentation:
            if random.random() > 0.5:
                input_tensor = TF.hflip(input_tensor)
                tgt_tensor = TF.hflip(tgt_tensor)
                v_mask = TF.hflip(v_mask)
            if random.random() > 0.5:
                input_tensor = TF.vflip(input_tensor)
                tgt_tensor = TF.vflip(tgt_tensor)
                v_mask = TF.vflip(v_mask)
        
        return input_tensor, tgt_tensor, v_mask

# ================= SOTA Model Components =================
class GaussianBlurLayer(nn.Module):
    def __init__(self, k=21, s=5.0):
        super().__init__()
        x = torch.arange(k).float()
        grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), -1)
        mean = (k-1)/2.
        kernel = torch.exp(-torch.sum((grid-mean)**2, -1)/(2*s**2))
        kernel /= kernel.sum()
        self.register_buffer('kernel', kernel.view(1,1,k,k))
        self.pad = k//2
    def forward(self, x): return F.conv2d(x, self.kernel, padding=self.pad)

class AddCoords(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        b, _, h, w = x.size()
        xx = torch.linspace(-1,1,w,device=x.device).view(1,1,1,w).repeat(b,1,h,1)
        yy = torch.linspace(-1,1,h,device=x.device).view(1,1,h,1).repeat(b,1,1,w)
        return torch.cat([x,xx,yy], 1)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class AdaptiveFFTBlock(nn.Module):
    """
    [优化版] 自适应频域滤波模块
    结构改进：x + Conv(IFFT(Weight * FFT(x)))
    注意：使用实部和虚部分离的参数，以兼容混合精度训练（AMP）
    """
    def __init__(self, in_channels):
        super().__init__()
        # 学习频域权重的参数（拆分为实部和虚部，避免复数类型与AMP不兼容）
        # 使用实数参数，在forward中组合成复数
        self.weight_real = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        
        # 频域特征处理后的融合层
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 可选：如果你想让残差更强，可以加一个可学习的缩放系数 gamma
        # self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 快速傅里叶变换 (Real -> Complex)
        # rfft2 得到只包含非负频率的部分，节省一半计算量
        fft_x = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 频域门控/滤波 (Spectral Gating)
        # 将实部和虚部组合成复数权重（在计算图中动态组合，避免存储复数参数）
        complex_weight = torch.complex(self.weight_real, self.weight_imag)
        weighted_fft = fft_x * complex_weight
        
        # 3. 傅里叶逆变换 (Complex -> Real)
        x_spectral = torch.fft.irfft2(weighted_fft, s=(H, W), norm='ortho')
        
        # 4. 特征融合
        # 对频域回来的特征做一次卷积整理
        x_spectral = self.spatial_conv(x_spectral)
        
        # 5. [关键] 标准残差连接
        # 原始特征 x (保留局部细节) + 全局频域特征 (补充长距离依赖)
        return x + x_spectral

# ================= 高级组件 (ASPP, CoordAtt, PixelShuffle) =================
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - 多尺度感知"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
        self.atrous_block1 = nn.Conv2d(in_c, out_c, 1)
        self.atrous_block6 = nn.Conv2d(in_c, out_c, 3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_c, out_c, 3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_c, out_c, 3, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_c * 5, out_c, 1)
        self.bn_out = nn.BatchNorm2d(out_c)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.relu(self.bn(image_features))
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        
        net = self.conv_1x1_output(torch.cat([
            image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1))
        return self.relu(self.bn_out(net))

class CoordAtt(nn.Module):
    """Coordinate Attention - 更精准地捕捉 XY 坐标"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class PixelShuffleUp(nn.Module):
    """PixelShuffle 上采样 - 亚像素卷积，杜绝边缘模糊"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ps = nn.PixelShuffle(upscale_factor=2)
    def forward(self, x):
        return self.ps(self.relu(self.bn(self.conv(x))))

class SuperResUNet(nn.Module):
    """
    Super ResUNet - 升级版架构
    Encoder: 保留强力的 ResNet 结构
    Bottleneck: ASPP (多尺度感知) + Adaptive FFT (全局频域感知) 双核驱动
    Decoder: PixelShuffle (亚像素卷积) 替代插值上采样，杜绝边缘模糊；Coordinate Attention 替代 CBAM，更精准地捕捉 XY 坐标
    """
    def __init__(self, inputs=2):
        super().__init__()
        self.blur = GaussianBlurLayer()
        self.coords = AddCoords()
        
        # Encoder (ResNet) - 保留强力的 ResNet 结构
        # Input 4 channels: Building, Heatmap, X, Y
        self.enc0 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.enc1 = ResBlock(32, 64)
        self.pool = nn.MaxPool2d(2) 
        
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        
        # Bottleneck (ASPP + FFT) - 双核驱动
        self.bottleneck_aspp = ASPP(256, 512)  # 多尺度感知
        self.bottleneck_fft = AdaptiveFFTBlock(512)  # 全局频域感知
        
        # Decoder (PixelShuffle + CoordAtt) - 亚像素卷积上采样 + 坐标注意力
        self.ps3 = PixelShuffleUp(512, 256)
        self.dec3 = ResBlock(256 + 256, 256)
        self.att3 = CoordAtt(256, 256)
        
        self.ps2 = PixelShuffleUp(256, 128)
        self.dec2 = ResBlock(128 + 128, 128)
        self.att2 = CoordAtt(128, 128)
        
        self.ps1 = PixelShuffleUp(128, 64)
        self.dec1 = ResBlock(64 + 64, 64)
        self.att1 = CoordAtt(64, 64)
        
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        b, tx = x[:,0:1], x[:,1:2]
        tx = self.blur(tx); tx = tx / (tx.amax((2,3), True) + 1e-6)
        x = self.coords(torch.cat([b, tx], 1))
        
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck: ASPP + FFT 双核驱动
        b_feat = self.pool(e3)
        b = self.bottleneck_aspp(b_feat)  # 多尺度感知
        b = self.bottleneck_fft(b)  # 全局频域感知
        
        # Decode: PixelShuffle 上采样 + CoordAtt 注意力
        d3 = self.att3(self.dec3(torch.cat([self.ps3(b), e3], dim=1)))
        d2 = self.att2(self.dec2(torch.cat([self.ps2(d3), e2], dim=1)))
        d1 = self.att1(self.dec1(torch.cat([self.ps1(d2), e1], dim=1)))
        
        return self.final(d1)

class AdvancedLoss(nn.Module):
    def __init__(self, coord_w=20.0, dice_w=1.0):
        super().__init__()
        self.coord_w = coord_w; self.dice_w = dice_w
    def forward(self, pred, tgt, mask):
        num = mask.sum() + 1e-6
        # Focal-like BCE
        bce = F.binary_cross_entropy_with_logits(pred[:,0:1], tgt[:,0:1], reduction='none')
        pt = torch.exp(-bce)
        pt = torch.exp(-bce)
        focal = ((1-pt)**2 * bce * mask).sum() / num
        
        # Dice Loss
        p_prob = torch.sigmoid(pred[:,0:1])
        inter = (p_prob * tgt[:,0:1] * mask).sum()
        union = (p_prob * mask).sum() + (tgt[:,0:1] * mask).sum()
        dice = 1 - (2.*inter + 1e-6)/(union + 1e-6)
        
        # L1
        p_coord = torch.sigmoid(pred[:,1:3])
        valid = mask * tgt[:,0:1]
        l1 = (F.l1_loss(p_coord, tgt[:,1:3], reduction='none').sum(1,True)*valid).sum()/(valid.sum()+1e-6)
        
        return focal + self.dice_w*dice + self.coord_w*l1

# ================= Training =================
def train_epoch(model, loader, crit, opt, scaler):
    model.train()
    loss_sum = 0
    # 优化：减少进度条更新频率
    pbar = tqdm(loader, desc="Train", leave=False, mininterval=1.0)
    for i, (x, y, m) in enumerate(pbar):
        x, y, m = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        with autocast(device_type='cuda'):
            loss = crit(model(x), y, m)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        loss_val = loss.item()
        loss_sum += loss_val
        # 优化：每100个batch更新一次进度条，减少开销
        if i % 100 == 0:
            pbar.set_postfix({'L': f"{loss_val:.4f}"})
    return loss_sum / len(loader)

def validate(model, loader, crit):
    model.eval()
    ls, er, n = 0, 0, 0
    total_valid_pixels = 0
    total_error_sum = 0.0
    total_correct = 0.0
    total_vertices = 0.0
    with torch.no_grad():
        # 优化：减少进度条更新频率
        pbar = tqdm(loader, desc="Val", leave=False, mininterval=1.0)
        for x, y, m in pbar:
            x, y, m = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
            with autocast(device_type='cuda'):
                logits = model(x)
                ls += crit(logits, y, m).item()
            pred = torch.sigmoid(logits)
            # 全局统计准确率：累加正确数量和总顶点数量
            pred_binary = (pred[:,0:1] > 0.5).float()
            correct = ((pred_binary == y[:,0:1]) * m).sum().item()
            total_correct += correct
            total_vertices += m.sum().item()
            
            valid = m * y[:,0:1]
            if valid.sum()>0:
                # 改进的计算方式：全局平均而不是按batch平均
                coord_diff = (pred[:,1:3] - y[:,1:3]) * 256
                pixel_errors = torch.sqrt((coord_diff ** 2).sum(dim=1, keepdim=True))
                valid_errors = (pixel_errors * valid).sum().item()
                valid_count = valid.sum().item()
                total_error_sum += valid_errors
                total_valid_pixels += valid_count
                # 保留旧方式用于兼容
                er += valid_errors / valid_count
            n+=1
    # 使用全局平均（更准确）
    avg_error = total_error_sum / total_valid_pixels if total_valid_pixels > 0 else 0.0
    avg_acc = total_correct / total_vertices if total_vertices > 0 else 0.0
    return ls/n, avg_acc, avg_error

def main():
    # 设置全局随机种子，确保可复现性
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了完全可复现，禁用 cudnn 的 benchmark 和 deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    log(f"🚀 SOTA Training on {DEVICE}")
    log(f"📊 从训练集中切分验证集 (比例={args.val_split_ratio:.1%})")
    log(f"📂 LMDB Path: {args.lmdb_path}")
    log(f"💾 Checkpoint Dir: {SAVE_DIR}")
    log(f"🎲 随机种子已设置为: {seed} (确保可复现性)")
    log("="*60)
    
    # 合并训练和验证集，然后按比例分割
    full_train_dataset = LMDBDataset(args.lmdb_path, args.raw_root, TRAIN_VAL_MAP_IDS, False)
    total_size = len(full_train_dataset)
    val_size = int(total_size * args.val_split_ratio)
    train_size = total_size - val_size
    
    # 使用固定随机种子确保可复现
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 包装训练集以支持数据增强
    train_dataset = AugmentedSubset(train_subset, augmentation=True)
    val_dataset = val_subset  # 验证集不需要增强
    
    log(f"✅ 训练集: {train_size:,} 样本, 验证集: {val_size:,} 样本 (来自 {len(TRAIN_VAL_MAP_IDS)} 个map)")
    
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                         num_workers=args.workers, pin_memory=True, 
                         persistent_workers=not args.disable_persistent_workers,
                         prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
                         drop_last=args.drop_last)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                       num_workers=args.workers, pin_memory=True,
                       prefetch_factor=args.prefetch_factor if args.workers > 0 else None)
    
    model = SuperResUNet(inputs=2).to(DEVICE)

    # Resume from checkpoint if provided
    if args.resume:
        if os.path.isfile(args.resume):
            log(f"🔄 从检查点恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            
            # 处理 state_dict (支持 resume 只有 state_dict 的情况)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 处理 compile 带来的 _orig_mod 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            
            try:
                model.load_state_dict(new_state_dict, strict=True)
                log("✅ 成功加载模型权重")
            except Exception as e:
                log(f"⚠️  加载权重部分失败 (strict=False): {e}")
                model.load_state_dict(new_state_dict, strict=False)
        else:
            log(f"❌ 检查点文件不存在: {args.resume}")
    
    # 使用torch.compile加速（PyTorch 2.0+）
    if args.compile:
        try:
            log("⚡ 启用 torch.compile 加速...")
            model = torch.compile(model, mode='reduce-overhead')
            log("✅ torch.compile 启用成功")
        except Exception as e:
            log(f"⚠️  torch.compile 失败，继续使用普通模式: {e}")
    
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    crit = AdvancedLoss()
    scaler = GradScaler()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    
    # 记录优化设置
    log(f"⚙️  性能设置: batch_size={args.batch_size}, workers={args.workers}, "
        f"prefetch={args.prefetch_factor}, compile={args.compile}, "
        f"drop_last={args.drop_last}")
    
    best_loss = float('inf')
    best_path = os.path.join(SAVE_DIR, "super_model_fft_thin.pth")
    
    log(f"🎯 Training for {args.epochs} epochs with batch_size={args.batch_size}, lr={args.lr}")
    log("="*60)
    
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tl = train_epoch(model, train_dl, crit, opt, scaler)
        vl, va, ve = validate(model, val_dl, crit)
        sched.step()
        
        msg = f"Ep {ep:03d} [{time.time()-t0:.0f}s] T_L:{tl:.4f} | V_L:{vl:.4f} Acc:{va:.2%} Err:{ve:.2f}px"
        if vl < best_loss:
            best_loss = vl
            # 如果使用了 torch.compile，需要获取原始模型的 state_dict
            if args.compile and hasattr(model, '_orig_mod'):
                state_dict = model._orig_mod.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, best_path)
            msg += " ★"
        log(msg)
    
    # ================= 测试集评估 =================
    log("="*60)
    log("🧪 训练完成，开始在测试集上评估最佳模型...")
    
    # 创建测试集 DataLoader
    test_dataset = LMDBDataset(args.lmdb_path, args.raw_root, TEST_MAP_IDS, False)
    test_dl = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None
    )
    log(f"📊 测试集: {len(TEST_MAP_IDS)} maps, {len(test_dl.dataset)} 样本")
    
    # 加载最佳模型权重
    if os.path.exists(best_path):
        # 创建原始模型（未编译版本）来加载权重
        original_model = SuperResUNet(inputs=2).to(DEVICE)
        state_dict = torch.load(best_path, map_location=DEVICE)
        
        # 处理键名：如果保存的权重包含 _orig_mod. 前缀，需要去掉
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # 去掉 _orig_mod. 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[10:]] = v  # 去掉 '_orig_mod.' 前缀（10个字符）
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            log("✅ 已处理权重键名（去掉 _orig_mod. 前缀）")
        
        # 加载权重
        try:
            original_model.load_state_dict(state_dict, strict=True)
            log("✅ 已加载最佳模型权重")
        except Exception as e:
            log(f"⚠️  加载权重时出错: {e}")
            log("   尝试使用 strict=False 加载...")
            original_model.load_state_dict(state_dict, strict=False)
            log("✅ 已加载最佳模型权重（部分键名不匹配，已忽略）")
        
        # 如果使用了 torch.compile，重新编译模型
        if args.compile:
            try:
                model = torch.compile(original_model, mode='reduce-overhead')
                log("✅ 模型已重新编译")
            except Exception as e:
                model = original_model
                log(f"⚠️  模型编译失败，使用普通模式: {e}")
        else:
            model = original_model
    else:
        log(f"⚠️  最佳模型文件不存在: {best_path}，使用当前模型进行评估")
    
    # 在测试集上评估
    test_loss, test_acc, test_err = validate(model, test_dl, crit)
    
    log("="*60)
    log(f"🎯 测试集最终结果:")
    log(f"   Loss: {test_loss:.4f}")
    log(f"   Accuracy: {test_acc:.2%}")
    log(f"   Error: {test_err:.2f}px")
    log("="*60)

if __name__ == "__main__":
    main()
