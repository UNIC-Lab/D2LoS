import os
import argparse
import json
import subprocess
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 模型组件 (与 train_fft_thin_super.py 保持一致)
# =============================================================================

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
    def __init__(self, in_channels):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(1, in_channels, 1, 1) * 0.02)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        fft_x = torch.fft.rfft2(x, norm='ortho')
        complex_weight = torch.complex(self.weight_real, self.weight_imag)
        weighted_fft = fft_x * complex_weight
        x_spectral = torch.fft.irfft2(weighted_fft, s=(H, W), norm='ortho')
        x_spectral = self.spatial_conv(x_spectral)
        return x + x_spectral

class ASPP(nn.Module):
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
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1))
        return self.relu(self.bn_out(net))

class CoordAtt(nn.Module):
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
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c * 4, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c * 4)
        self.relu = nn.ReLU(inplace=True)
        self.ps = nn.PixelShuffle(upscale_factor=2)
    def forward(self, x):
        return self.ps(self.relu(self.bn(self.conv(x))))

class SuperResUNet(nn.Module):
    def __init__(self, inputs=2):
        super().__init__()
        self.blur = GaussianBlurLayer()
        self.coords = AddCoords()
        self.enc0 = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.enc1 = ResBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.bottleneck_aspp = ASPP(256, 512)
        self.bottleneck_fft = AdaptiveFFTBlock(512)
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
        b_in, tx_pos = x[:,0:1], x[:,1:2]
        tx_blur = self.blur(tx_pos)
        tx_blur = tx_blur / (tx_blur.amax((2,3), True) + 1e-6)
        x_in = self.coords(torch.cat([b_in, tx_blur], 1))
        e0 = self.enc0(x_in)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b_feat = self.pool(e3)
        bn = self.bottleneck_aspp(b_feat)
        bn = self.bottleneck_fft(bn)
        d3 = self.att3(self.dec3(torch.cat([self.ps3(bn), e3], dim=1)))
        d2 = self.att2(self.dec2(torch.cat([self.ps2(d3), e2], dim=1)))
        d1 = self.att1(self.dec1(torch.cat([self.ps1(d2), e1], dim=1)))
        return self.final(d1)

# =============================================================================
# 辅助函数
# =============================================================================

def get_visibility_polygon_cv2(tx_xy, proj_points, grid_size=257):
    """使用 cv2 快速填充多边形（无几何修正版本直接使用此函数构建 street mask）"""
    if len(proj_points) < 3:
        return np.zeros((grid_size, grid_size), dtype=np.int8)

    def get_angle(pt):
        return np.atan2(pt[1] - tx_xy[1], pt[0] - tx_xy[0])

    sorted_pts = sorted(proj_points, key=get_angle)
    pts = np.array(sorted_pts, dtype=np.int32).reshape((-1, 1, 2))

    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(np.int8)

def compute_2d_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# =============================================================================
# 批量推理入口
# =============================================================================

def resolve_map_ids(args, parser):
    if args.map_ids:
        return list(dict.fromkeys(args.map_ids))
    if args.map_id_start is not None or args.map_id_end is not None:
        if args.map_id_start is None or args.map_id_end is None:
            parser.error("--map-id-start and --map-id-end must be used together")
        if args.map_id_end < args.map_id_start:
            parser.error("--map-id-end must be >= --map-id-start")
        return list(range(args.map_id_start, args.map_id_end + 1))
    if args.map_id is not None:
        return [args.map_id]
    parser.error("one of --map-id, --map-ids, or --map-id-start/--map-id-end is required")


def run_batch_inference(args, map_ids):
    failed = []
    total = len(map_ids)
    script_path = os.path.abspath(__file__)

    for idx, map_id in enumerate(map_ids, start=1):
        cmd = [
            sys.executable,
            script_path,
            "--checkpoint", args.checkpoint,
            "--data-root", args.data_root,
            "--map-id", str(map_id),
            "--vertex-height", str(args.vertex_height),
            "--street-height", str(args.street_height),
            "--gpu-id", str(args.gpu_id),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
        ]
        if args.output_root is not None:
            cmd.extend(["--output-root", args.output_root])

        print(f"\n[{idx}/{total}] map_id={map_id}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failed.append(map_id)

    if failed:
        print(f"\nBatch inference finished with failures: {failed}")
        raise SystemExit(1)

    print(f"\nBatch inference finished successfully: {total} maps")


def main():
    parser = argparse.ArgumentParser(
        description="消融实验：去掉几何修正的推理脚本（直接使用模型输出，不调用 proj_geometry）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    paths = parser.add_argument_group("路径参数 (Paths)")
    paths.add_argument("--checkpoint", type=str, required=True, metavar="PATH")
    paths.add_argument("--data-root", type=str, required=True, metavar="DIR")
    paths.add_argument("--output-root", type=str, default=None, metavar="DIR")

    geo = parser.add_argument_group("几何参数 (Geometry)")
    geo.add_argument("--map-id", type=int, default=None)
    geo.add_argument("--map-ids", type=int, nargs="+", default=None)
    geo.add_argument("--map-id-start", type=int, default=None)
    geo.add_argument("--map-id-end", type=int, default=None)
    geo.add_argument("--vertex-height", type=float, default=20.0, metavar="M")
    geo.add_argument("--street-height", type=float, default=1.5, metavar="M")

    run = parser.add_argument_group("运行参数 (Runtime)")
    run.add_argument("--gpu-id", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    map_ids = resolve_map_ids(args, parser)
    if len(map_ids) > 1:
        run_batch_inference(args, map_ids)
        return
    args.map_id = map_ids[0]

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("[消融模式] 已禁用几何修正，直接使用模型输出")

    if args.output_root is None:
        args.output_root = args.data_root

    map_dir = os.path.join(args.data_root, f"map_{args.map_id}")
    out_dir = os.path.join(args.output_root, f"map_{args.map_id}")
    fund_dir = os.path.join(out_dir, "fundamental")
    os.makedirs(fund_dir, exist_ok=True)

    # 1. 加载模型
    model = SuperResUNet(inputs=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        raw_sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and not any(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:1]):
        raw_sd = next((v for v in ckpt.values() if isinstance(v, dict)), ckpt)
    else:
        raw_sd = ckpt
    new_state_dict = {}
    for k, v in raw_sd.items():
        name = k
        if name.startswith('_orig_mod.'): name = name[10:]
        if name.startswith('module.'): name = name[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 2. 加载静态数据
    b_img = cv2.imread(
        os.path.join(args.data_root, "png_buildings_complete", f"{args.map_id}.png"),
        cv2.IMREAD_GRAYSCALE
    )
    if b_img is None:
        print(f"Error: png_buildings_complete/{args.map_id}.png not found in {args.data_root}")
        return
    b_ts = torch.from_numpy(b_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # 建筑掩码（用于排除建筑内部像素）
    building_mask_256 = (b_img > 127).astype(np.uint8)
    building_mask_257 = cv2.resize(building_mask_256, (257, 257), interpolation=cv2.INTER_NEAREST)
    building_mask_257 = building_mask_257[::-1].copy()

    # 加载顶点坐标
    vertex_points_path = os.path.join(map_dir, f"vertex_points_{args.map_id}.npy")
    if not os.path.exists(vertex_points_path):
        print(f"Error: {vertex_points_path} not found.")
        return
    vertex_points = np.load(vertex_points_path)
    num_vertices = len(vertex_points)
    print(f"Loaded {num_vertices} vertices.")

    # 3. 加载 Selected Sources
    sel_src_path = os.path.join(map_dir, f"selected_source_points_{args.map_id}.npy")
    if not os.path.exists(sel_src_path):
        print(f"Warning: {sel_src_path} not found.")
        return
    selected_sources = np.load(sel_src_path)
    num_selected = len(selected_sources)
    print(f"Processing {num_selected} selected sources.")

    BS = args.batch_size

    # =========================================================================
    # Selected 推理循环（无几何修正）
    # =========================================================================
    print("Inferring Selected Sources (no geometry correction)...")
    sel_vis_vertex = np.full((num_selected, 257, 257), -1, dtype=np.int8)
    sel_vis_street = np.zeros((num_selected, 257, 257), dtype=np.int8)
    sel_proj_points = np.full((num_selected, 257, 257, 2), -1.0, dtype=np.float32)

    # 阶段1: batch 模型推理
    sel_preds = np.empty((num_selected, 3, 256, 256), dtype=np.float32)
    for start in tqdm(range(0, num_selected, BS), desc="Selected inference"):
        end = min(start + BS, num_selected)
        batch_tx = torch.zeros(end - start, 1, 256, 256)
        for j, idx in enumerate(range(start, end)):
            sx, sy = selected_sources[idx]
            ix, iy = int(np.clip(sx, 0, 255)), int(np.clip(sy, 0, 255))
            batch_tx[j, 0, iy, ix] = 1.0
        batch_b = b_ts.expand(end - start, -1, -1, -1)
        batch_input = torch.cat([batch_b, batch_tx.to(device)], 1)
        with torch.no_grad():
            batch_pred = torch.sigmoid(model(batch_input)).cpu().numpy()
        sel_preds[start:end] = batch_pred

    # 阶段2: 逐 source 后处理（直接使用模型输出，无几何修正）
    for i in tqdm(range(num_selected), desc="Selected postproc (no geom)"):
        sx, sy = selected_sources[i]
        pred = sel_preds[i]  # [3, 256, 256]

        vis_map = pred[0]
        pred_proj_ch = pred[1:3]

        # TX 自身位置
        sel_vis_vertex[i, int(np.clip(sy, 0, 256)), int(np.clip(sx, 0, 256))] = 0

        # 遍历顶点：直接用模型可见性输出，不做几何验证
        proj_points_for_street = []
        for v_idx in range(num_vertices):
            vx, vy = vertex_points[v_idx]
            pix_x, pix_y = int(np.clip(vx, 0, 255)), int(np.clip(vy, 0, 255))
            if vis_map[pix_y, pix_x] > 0.5:
                gx = int(np.clip(vx, 0, 256))
                gy = int(np.clip(vy, 0, 256))
                sel_vis_vertex[i, gy, gx] = 1

                # 直接使用模型预测的投影点（无修正）
                pred_px = float(pred_proj_ch[0, pix_y, pix_x]) * 256.0
                pred_py = 255.0 - float(pred_proj_ch[1, pix_y, pix_x]) * 256.0
                pred_px = float(np.clip(pred_px, 0, 256))
                pred_py = float(np.clip(pred_py, 0, 256))

                sel_proj_points[i, gy, gx, 0] = pred_px
                sel_proj_points[i, gy, gx, 1] = pred_py
                proj_points_for_street.append((int(round(pred_px)), int(round(pred_py))))

        # 直接用投影点列表构建 street mask（无 sector-cell 算法）
        street_mask = get_visibility_polygon_cv2((sx, sy), proj_points_for_street, grid_size=257)
        street_mask[building_mask_257 == 1] = 0
        sel_vis_street[i] = street_mask.astype(np.int8)

    # 保存 Selected 数据
    np.save(os.path.join(out_dir, f"selected_visibility_vertex_mask_{args.map_id}.npy"), sel_vis_vertex)
    np.save(os.path.join(out_dir, f"selected_visibility_street_mask_{args.map_id}.npy"), sel_vis_street)
    np.save(os.path.join(out_dir, f"selected_proj_points_{args.map_id}.npy"), sel_proj_points)

    # =========================================================================
    # Fundamental 推理循环（无几何修正）
    # =========================================================================
    print("Inferring Fundamental (Per-Vertex, no geometry correction)...")
    fund_vertex_vis = np.full((num_vertices, 257, 257), -1, dtype=np.int8)
    fund_street_vis = np.full((num_vertices, 257, 257), -1, dtype=np.int8)
    fund_vertex_dist = np.full((num_vertices, 257, 257), -1.0, dtype=np.float32)
    fund_vertex_angle = np.full((num_vertices, 257, 257), -1.0, dtype=np.float32)
    fund_street_dist = np.full((num_vertices, 257, 257), -1.0, dtype=np.float32)
    fund_street_angle = np.full((num_vertices, 257, 257, 2), -1.0, dtype=np.float32)
    fund_proj_points = np.full((num_vertices, 257, 257, 2), -1.0, dtype=np.float32)

    dz_street = args.street_height - args.vertex_height

    # 阶段1: batch 模型推理
    fund_preds = np.empty((num_vertices, 3, 256, 256), dtype=np.float32)
    for start in tqdm(range(0, num_vertices, BS), desc="Fundamental inference"):
        end = min(start + BS, num_vertices)
        batch_tx = torch.zeros(end - start, 1, 256, 256)
        for j, idx in enumerate(range(start, end)):
            tx_x, tx_y = vertex_points[idx]
            ix, iy = int(np.clip(tx_x, 0, 255)), int(np.clip(tx_y, 0, 255))
            batch_tx[j, 0, iy, ix] = 1.0
        batch_b = b_ts.expand(end - start, -1, -1, -1)
        batch_input = torch.cat([batch_b, batch_tx.to(device)], 1)
        with torch.no_grad():
            batch_pred = torch.sigmoid(model(batch_input)).cpu().numpy()
        fund_preds[start:end] = batch_pred

    # 阶段2: 逐 vertex 后处理（无几何修正）
    for k in tqdm(range(num_vertices), desc="Fundamental postproc (no geom)"):
        tx_x, tx_y = vertex_points[k]
        pred = fund_preds[k]  # [3, 256, 256]

        vis_map = pred[0]
        proj_ch = pred[1:3]

        # 自身位置 [X, Y]
        gx_self, gy_self = int(np.clip(tx_x, 0, 256)), int(np.clip(tx_y, 0, 256))
        fund_vertex_vis[k, gx_self, gy_self] = 0
        fund_street_vis[k, gx_self, gy_self] = 0
        fund_vertex_dist[k, gx_self, gy_self] = 0.0
        fund_vertex_angle[k, gx_self, gy_self] = 0.0
        fund_street_dist[k, gx_self, gy_self] = 0.0
        fund_street_angle[k, gx_self, gy_self, 0] = 0.0
        fund_street_angle[k, gx_self, gy_self, 1] = 0.0

        proj_points_for_street = []
        for v_idx in range(num_vertices):
            if v_idx == k:
                continue
            vx, vy = vertex_points[v_idx]
            pix_x, pix_y = int(np.clip(vx, 0, 255)), int(np.clip(vy, 0, 255))

            if vis_map[pix_y, pix_x] > 0.5:
                gx, gy = int(np.clip(vx, 0, 256)), int(np.clip(vy, 0, 256))
                # 直接使用模型可见性，不做几何验证
                fund_vertex_vis[k, gx, gy] = 1
                fund_vertex_dist[k, gx, gy] = compute_2d_distance(tx_x, tx_y, vx, vy)
                fund_vertex_angle[k, gx, gy] = np.atan2(vy - tx_y, vx - tx_x)

                # 直接使用模型预测的投影点（无修正）
                pred_px = float(proj_ch[0, pix_y, pix_x]) * 256.0
                pred_py = 255.0 - float(proj_ch[1, pix_y, pix_x]) * 256.0
                pred_px = float(np.clip(pred_px, 0, 256))
                pred_py = float(np.clip(pred_py, 0, 256))

                fund_proj_points[k, gx, gy, 0] = pred_px
                fund_proj_points[k, gx, gy, 1] = pred_py
                proj_points_for_street.append((int(round(pred_px)), int(round(pred_py))))

        # 直接用投影点列表构建 street mask（无 sector-cell 算法）
        street_poly_xy = get_visibility_polygon_cv2((tx_x, tx_y), proj_points_for_street, grid_size=257)
        street_poly_xy[building_mask_257 == 1] = 0

        # 向量化填充到 fundamental 数组 [k, X, Y]
        y_indices, x_indices = np.where(street_poly_xy == 1)
        if len(y_indices) > 0:
            not_self = (x_indices != gx_self) | (y_indices != gy_self)
            valid_xs = x_indices[not_self]
            valid_ys = y_indices[not_self]

            fund_street_vis[k, valid_xs, valid_ys] = 1

            dx = valid_xs - tx_x
            dy = valid_ys - tx_y
            d2d = np.sqrt(dx**2 + dy**2)

            fund_street_dist[k, valid_xs, valid_ys] = np.sqrt(d2d**2 + dz_street**2)
            fund_street_angle[k, valid_xs, valid_ys, 0] = np.arctan2(dy, dx)
            fund_street_angle[k, valid_xs, valid_ys, 1] = np.arctan2(dz_street, d2d)

    # 保存 Fundamental 数据
    files_to_save = {
        os.path.join(fund_dir, f"vertex_visibility_mask_{args.map_id}.npy"):  fund_vertex_vis,
        os.path.join(fund_dir, f"street_visibility_mask_{args.map_id}.npy"):  fund_street_vis,
        os.path.join(fund_dir, f"vertex_distance_mask_{args.map_id}.npy"):    fund_vertex_dist,
        os.path.join(fund_dir, f"vertex_angle_mask_{args.map_id}.npy"):       fund_vertex_angle,
        os.path.join(fund_dir, f"street_distance_mask_{args.map_id}.npy"):    fund_street_dist,
        os.path.join(fund_dir, f"street_angle_mask_{args.map_id}.npy"):       fund_street_angle,
        os.path.join(fund_dir, f"proj_points_{args.map_id}.npy"):             fund_proj_points,
    }
    for path, arr in files_to_save.items():
        np.save(path, arr)

    print("\n── 输出文件 ──────────────────────────────────")
    for path, arr in {
        os.path.join(out_dir, f"selected_visibility_vertex_mask_{args.map_id}.npy"): sel_vis_vertex,
        os.path.join(out_dir, f"selected_visibility_street_mask_{args.map_id}.npy"): sel_vis_street,
        **files_to_save
    }.items():
        print(f"  {path}")
        print(f"    shape={arr.shape}  dtype={arr.dtype}")
    print(f"\n✓ 推理完成（无几何修正）  map_id={args.map_id}  output_root={args.output_root}")

if __name__ == "__main__":
    main()
