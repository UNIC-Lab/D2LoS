import csv
import os
import argparse
import json
import subprocess
import sys
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
# from shapely.geometry import Polygon, Point # Removed dependency
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
import proj_geometry  # 投影点几何修正模块

# =============================================================================
# 模型组件 (直接从 train_fft_thin_super.py 复制，确保一致性)
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
        # 归一化热图
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

# Removed get_visibility_polygon_sampling to eliminate shapely dependency

def get_visibility_polygon_cv2(tx_xy, proj_points, grid_size=257):
    """使用 cv2 快速填充多边形"""
    if len(proj_points) < 3:
        return np.zeros((grid_size, grid_size), dtype=np.int8)
    
    def get_angle(pt):
        return np.atan2(pt[1] - tx_xy[1], pt[0] - tx_xy[0])
    
    sorted_pts = sorted(proj_points, key=get_angle)
    pts = np.array(sorted_pts, dtype=np.int32).reshape((-1, 1, 2))
    
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(np.int8)

def compute_3d_distance(x1, y1, h1, x2, y2, h2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (h2-h1)**2)

def compute_2d_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# =============================================================================
# 推理流程
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


TIMING_FIELDNAMES = [
    "map_id",
    "t_model_load_s", "t_data_load_s",
    "t_sel_infer_s", "t_sel_postproc_s",
    "t_fund_infer_s", "t_fund_postproc_s",
    "t_save_s", "t_total_map_s",
]


def _append_timing_row(csv_path: str, row: dict) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TIMING_FIELDNAMES})


def run_batch_inference(args, map_ids):
    failed = []
    total = len(map_ids)
    script_path = os.path.abspath(__file__)
    timing_csv = args.timing_csv

    map_wall_times = {}
    batch_start = time.perf_counter()

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
            "--timing-csv", timing_csv,
        ]
        if args.output_root is not None:
            cmd.extend(["--output-root", args.output_root])

        print(f"\n[{idx}/{total}] map_id={map_id}")
        t_map_start = time.perf_counter()
        result = subprocess.run(cmd)
        map_wall_times[map_id] = time.perf_counter() - t_map_start

        if result.returncode != 0:
            failed.append(map_id)
        else:
            print(f"  wall time: {map_wall_times[map_id]:.1f}s")

    batch_elapsed = time.perf_counter() - batch_start
    successful = [m for m in map_ids if m not in failed]

    # Write summary rows
    if successful:
        avg_wall = sum(map_wall_times[m] for m in successful) / len(successful)
        _append_timing_row(timing_csv, {
            "map_id": "TOTAL",
            "t_total_map_s": f"{batch_elapsed:.3f}",
        })
        _append_timing_row(timing_csv, {
            "map_id": "AVERAGE",
            "t_total_map_s": f"{avg_wall:.3f}",
        })
        print(f"\nTotal wall time : {batch_elapsed:.1f}s")
        print(f"Avg per map     : {avg_wall:.1f}s  ({len(successful)} maps)")
        print(f"Timing CSV      : {timing_csv}")

    if failed:
        print(f"\nBatch inference finished with failures: {failed}")
        raise SystemExit(1)

    print(f"\nBatch inference finished successfully: {total} maps")


def main():
    parser = argparse.ArgumentParser(
        description="SuperResUNet 推理脚本：生成与 New_C_Plus_los 兼容的 .npy 文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── 路径参数 ──────────────────────────────────────────────────
    paths = parser.add_argument_group("路径参数 (Paths)")
    paths.add_argument(
        "--checkpoint", type=str, required=True,
        metavar="PATH",
        help="训练好的模型权重路径 (.pth)"
    )
    paths.add_argument(
        "--data-root", type=str, required=True,
        metavar="DIR",
        help="输入数据根目录，需包含 map_{ID}/ 子目录"
    )
    paths.add_argument(
        "--output-root", type=str, default=None,
        metavar="DIR",
        help="输出根目录（默认与 data-root 相同）; BFS 的 --data-root 指向此处"
    )

    # ── 几何/物理参数 ─────────────────────────────────────────────
    geo = parser.add_argument_group("几何参数 (Geometry)")
    geo.add_argument("--map-id", type=int, default=None,
                     help="要处理的地图 ID")
    geo.add_argument("--map-ids", type=int, nargs="+", default=None,
                     help="批量处理多个 map_id，例如 --map-ids 630 631 632")
    geo.add_argument("--map-id-start", type=int, default=None,
                     help="批量处理起始 map_id（需和 --map-id-end 一起用）")
    geo.add_argument("--map-id-end", type=int, default=None,
                     help="批量处理结束 map_id（含，需和 --map-id-start 一起用）")
    geo.add_argument("--vertex-height", type=float, default=20.0,
                     metavar="M", help="顶点（建筑角点）高度（米）")
    geo.add_argument("--street-height", type=float, default=1.5,
                     metavar="M", help="街道接收点高度（米）")

    # ── 运行参数 ──────────────────────────────────────────────────
    run = parser.add_argument_group("运行参数 (Runtime)")
    run.add_argument("--gpu-id", type=int, default=0,
                     help="使用的 GPU 编号，例如 0、1、2")
    run.add_argument("--batch-size", type=int, default=64,
                     help="模型推理 batch size（同时推理的 source 数量）")
    run.add_argument("--seed", type=int, default=42,
                     help="随机种子（保留，暂未使用）")
    run.add_argument("--timing-csv", type=str, default="timing_report.csv",
                     metavar="PATH", help="时间统计输出 CSV 路径")

    args = parser.parse_args()
    map_ids = resolve_map_ids(args, parser)
    if len(map_ids) > 1:
        run_batch_inference(args, map_ids)
        return
    args.map_id = map_ids[0]

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.output_root is None:
        args.output_root = args.data_root

    map_dir = os.path.join(args.data_root, f"map_{args.map_id}")
    out_dir = os.path.join(args.output_root, f"map_{args.map_id}")
    fund_dir = os.path.join(out_dir, "fundamental")
    os.makedirs(fund_dir, exist_ok=True)

    _t_map_start = time.perf_counter()

    # 1. 加载模型
    _t0 = time.perf_counter()
    model = SuperResUNet(inputs=2).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # 兼容两种 checkpoint 格式（对齐训练脚本的实际保存逻辑）：
    #   A) torch.save(state_dict, path) 直接保存 state_dict（最常见）
    #   B) {'state_dict': ..., ...} 格式（resume 保存的完整 checkpoint）
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        raw_sd = ckpt['state_dict']
    elif isinstance(ckpt, dict) and not any(isinstance(v, torch.Tensor) for v in list(ckpt.values())[:1]):
        # 字典中第一个 value 不是 Tensor，可能是包装过的
        raw_sd = next((v for v in ckpt.values() if isinstance(v, dict)), ckpt)
    else:
        raw_sd = ckpt  # 直接是 state_dict
    # 去掉 torch.compile (_orig_mod.) 和 DataParallel (module.) 的前缀
    new_state_dict = {}
    for k, v in raw_sd.items():
        name = k
        if name.startswith('_orig_mod.'): name = name[10:]
        if name.startswith('module.'): name = name[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    _t_model_load = time.perf_counter() - _t0

    # 2. 加载静态数据
    _t0 = time.perf_counter()
    b_img = cv2.imread(
        os.path.join(args.data_root, "png_buildings_complete", f"{args.map_id}.png"),
        cv2.IMREAD_GRAYSCALE
    )
    if b_img is None:
        print(f"Error: png_buildings_complete/{args.map_id}.png not found in {args.data_root}")
        return
    b_ts = torch.from_numpy(b_img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0 # [1, 1, 256, 256]

    # 构建 257×257 建筑掩码，用于排除建筑物内部像素（街道点不含建筑物内部）
    building_mask_256 = (b_img > 127).astype(np.uint8)  # 建筑=1, 街道=0
    building_mask_257 = cv2.resize(building_mask_256, (257, 257), interpolation=cv2.INTER_NEAREST)
    building_mask_257 = building_mask_257[::-1].copy()  # PNG Y轴翻转，对齐 npy 坐标系

    # 加载 C++ 生成的 vertex_points，确保索引完全对齐
    vertex_points_path = os.path.join(map_dir, f"vertex_points_{args.map_id}.npy")
    if not os.path.exists(vertex_points_path):
        print(f"Error: {vertex_points_path} not found. Please run C++ code first to generate base geometry.")
        return
    vertex_points = np.load(vertex_points_path) # [M, 2]
    num_vertices = len(vertex_points)
    print(f"Loaded {num_vertices} vertices.")
    vertex_coord_to_indices = {}
    for idx, (vx, vy) in enumerate(vertex_points):
        key = (round(float(vx), 6), round(float(vy), 6))
        vertex_coord_to_indices.setdefault(key, []).append(idx)

    # 构建几何修正用的边集（从 buildings JSON 加载建筑棱 + 地图边界）
    buildings_json_path = os.path.join(args.data_root, "buildings_complete", f"{args.map_id}.json")
    _buildings = proj_geometry.load_buildings(buildings_json_path)
    _edge_set = proj_geometry.build_edge_set(_buildings)
    _adjacency = {}  # Legacy adjacency path is no longer used by sector-cell rasterization.
    print(f"Built edge set: {len(_edge_set)} edges (from {len(_buildings)} buildings + boundary).")

    # 3. 确定 Selected Sources
    sel_src_path = os.path.join(map_dir, f"selected_source_points_{args.map_id}.npy")
    if os.path.exists(sel_src_path):
        selected_sources = np.load(sel_src_path)
    else:
        print(f"Warning: {sel_src_path} not found.")
        return
    num_selected = len(selected_sources)
    print(f"Processing {num_selected} selected sources.")
    _t_data_load = time.perf_counter() - _t0

    # =========================================================================
    # Selected 推理循环 [S, 257, 257] (Y, X)
    # =========================================================================
    print("Inferring Selected Sources...")
    sel_vis_vertex = np.full((num_selected, 257, 257), -1, dtype=np.int8)
    sel_vis_street = np.zeros((num_selected, 257, 257), dtype=np.int8)
    # 投影点 [S, Y, X, 2]，MAP 坐标（与 buildings JSON 一致）；-1 表示无效
    sel_proj_points = np.full((num_selected, 257, 257, 2), -1.0, dtype=np.float32)

    BS = args.batch_size

    # ---- 阶段1: batch 模型推理，缓存全部预测 ----
    _t0 = time.perf_counter()
    sel_preds = np.empty((num_selected, 3, 256, 256), dtype=np.float32)
    for start in tqdm(range(0, num_selected, BS), desc="Selected inference"):
        end = min(start + BS, num_selected)
        batch_tx = torch.zeros(end - start, 1, 256, 256)
        for j, idx in enumerate(range(start, end)):
            sx, sy = selected_sources[idx]
            ix, iy = int(np.clip(sx, 0, 255)), int(np.clip(sy, 0, 255))
            batch_tx[j, 0, iy, ix] = 1.0
        batch_b = b_ts.expand(end - start, -1, -1, -1)  # [B, 1, 256, 256]
        batch_input = torch.cat([batch_b, batch_tx.to(device)], 1)  # [B, 2, 256, 256]
        with torch.no_grad():
            batch_pred = torch.sigmoid(model(batch_input)).cpu().numpy()  # [B, 3, 256, 256]
        sel_preds[start:end] = batch_pred

    _t_sel_infer = time.perf_counter() - _t0

    # ---- 阶段2: 逐 source 后处理（几何修正 + 可视多边形） ----
    _t0 = time.perf_counter()
    for i in tqdm(range(num_selected), desc="Selected postproc"):
        sx, sy = selected_sources[i]
        pred = sel_preds[i]  # [3, 256, 256]

        vis_map = pred[0]
        pred_proj_ch = pred[1:3]

        # TX 自身位置
        sel_vis_vertex[i, int(np.clip(sy, 0, 256)), int(np.clip(sx, 0, 256))] = 0

        # 遍历顶点：可见性 + 投影点修正
        proj_entries_sel = []
        for v_idx in range(num_vertices):
            vx, vy = vertex_points[v_idx]
            pix_x, pix_y = int(np.clip(vx, 0, 255)), int(np.clip(vy, 0, 255))
            if vis_map[pix_y, pix_x] > 0.5:
                gx = int(np.clip(vx, 0, 256))
                gy = int(np.clip(vy, 0, 256))
                # 几何验证：source → vertex 是否被遮挡
                if not proj_geometry.is_visible((sx, sy), (vx, vy), _edge_set):
                    continue
                sel_vis_vertex[i, gy, gx] = 1
                pred_px = float(pred_proj_ch[0, pix_y, pix_x]) * 256.0
                pred_py = 255.0 - float(pred_proj_ch[1, pix_y, pix_x]) * 256.0
                (cx, cy), edge = proj_geometry.correct_proj_point_with_edge(
                    (sx, sy), (vx, vy), _edge_set, pred_proj_xy=(pred_px, pred_py),
                    buildings=_buildings,
                )
                proj_entries_sel.append((gx, gy, (vx, vy), (cx, cy), edge))

        proj_data_sel = []
        for gx, gy, vxy, pxy, edge in proj_entries_sel:
            (ax, ay), audited_edge = proj_geometry.audit_near_vertex_projection_with_edge(
                (sx, sy),
                vxy,
                pxy,
                edge,
                _edge_set,
                buildings=_buildings,
            )
            sel_proj_points[i, gy, gx, 0] = ax
            sel_proj_points[i, gy, gx, 1] = ay
            proj_data_sel.append((vxy, (ax, ay), audited_edge))

        street_poly = proj_geometry.build_visibility_mask(
            (sx, sy), proj_data_sel, _adjacency, _buildings,
            edge_set=_edge_set,
        )
        street_poly[building_mask_257 == 1] = 0
        sel_vis_street[i] = street_poly.astype(np.int8)

    _t_sel_postproc = time.perf_counter() - _t0

    # 保存 Selected 数据
    np.save(os.path.join(out_dir, f"selected_visibility_vertex_mask_{args.map_id}.npy"), sel_vis_vertex)
    np.save(os.path.join(out_dir, f"selected_visibility_street_mask_{args.map_id}.npy"), sel_vis_street)
    np.save(os.path.join(out_dir, f"selected_proj_points_{args.map_id}.npy"), sel_proj_points)

    # =========================================================================
    # Fundamental 推理循环 [M, 257, 257] (X, Y 转置!)
    # =========================================================================
    print("Inferring Fundamental (Per-Vertex)...")
    fund_vertex_vis = np.full((num_vertices, 257, 257), -1, dtype=np.int8)
    fund_street_vis = np.full((num_vertices, 257, 257), -1, dtype=np.int8)
    fund_vertex_dist = np.full((num_vertices, 257, 257), -1.0, dtype=np.float32)
    fund_vertex_angle = np.full((num_vertices, 257, 257), -1.0, dtype=np.float32)
    fund_street_dist = np.full((num_vertices, 257, 257), -1.0, dtype=np.float32)
    fund_street_angle = np.full((num_vertices, 257, 257, 2), -1.0, dtype=np.float32)
    # 投影点坐标 [M, 257, 257, 2]：对可见顶点，存储穿过该顶点后射线的落点坐标 (proj_x, proj_y)
    # 坐标轴顺序与 fundamental 系列一致（X 在前）; -1 表示无效
    fund_proj_points = np.full((num_vertices, 257, 257, 2), -1.0, dtype=np.float32)

    dz_street = args.street_height - args.vertex_height

    # ---- 阶段1: batch 模型推理，缓存全部预测 ----
    _t0 = time.perf_counter()
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

    _t_fund_infer = time.perf_counter() - _t0

    # ---- 阶段2: 逐 vertex 后处理 ----
    _t0 = time.perf_counter()
    for k in tqdm(range(num_vertices), desc="Fundamental postproc"):
        tx_x, tx_y = vertex_points[k]
        pred = fund_preds[k]  # [3, 256, 256]

        vis_map = pred[0]
        proj_ch = pred[1:3]
        forced_visible_indices = set()

        # Highest-priority supplement for vertex-mode:
        # if the source vertex lies on a building, its topological building-neighbors
        # should be re-checked geometrically even when the model misses them.
        for nxy in proj_geometry.get_building_neighbor_vertices((tx_x, tx_y), _buildings):
            key = (round(float(nxy[0]), 6), round(float(nxy[1]), 6))
            for v_idx in vertex_coord_to_indices.get(key, []):
                if v_idx == k:
                    continue
                vx_n, vy_n = vertex_points[v_idx]
                if proj_geometry.is_visible((tx_x, tx_y), (vx_n, vy_n), _edge_set):
                    forced_visible_indices.add(v_idx)

        # 自身位置 [X, Y]
        gx_self, gy_self = int(np.clip(tx_x, 0, 256)), int(np.clip(tx_y, 0, 256))
        fund_vertex_vis[k, gx_self, gy_self] = 0
        fund_street_vis[k, gx_self, gy_self] = 0
        fund_vertex_dist[k, gx_self, gy_self] = 0.0
        fund_vertex_angle[k, gx_self, gy_self] = 0.0
        fund_street_dist[k, gx_self, gy_self] = 0.0
        fund_street_angle[k, gx_self, gy_self, 0] = 0.0
        fund_street_angle[k, gx_self, gy_self, 1] = 0.0

        proj_entries_fund = []
        for v_idx in range(num_vertices):
            if v_idx == k:
                continue
            vx, vy = vertex_points[v_idx]
            pix_x, pix_y = int(np.clip(vx, 0, 255)), int(np.clip(vy, 0, 255))

            if vis_map[pix_y, pix_x] > 0.5 or v_idx in forced_visible_indices:
                gx, gy = int(np.clip(vx, 0, 256)), int(np.clip(vy, 0, 256))
                # 几何验证：source → vertex 是否被遮挡
                if not proj_geometry.is_visible((tx_x, tx_y), (vx, vy), _edge_set):
                    continue
                fund_vertex_vis[k, gx, gy] = 1
                fund_vertex_dist[k, gx, gy] = compute_2d_distance(tx_x, tx_y, vx, vy)
                fund_vertex_angle[k, gx, gy] = np.atan2(vy - tx_y, vx - tx_x)
                pred_proj_xy = None
                if v_idx not in forced_visible_indices:
                    pred_px = float(proj_ch[0, pix_y, pix_x]) * 256.0
                    pred_py = 255.0 - float(proj_ch[1, pix_y, pix_x]) * 256.0
                    pred_proj_xy = (pred_px, pred_py)
                (cx, cy), edge = proj_geometry.correct_proj_point_with_edge(
                    (tx_x, tx_y), (vx, vy), _edge_set, pred_proj_xy=pred_proj_xy,
                    buildings=_buildings,
                )
                proj_entries_fund.append((gx, gy, (vx, vy), (cx, cy), edge))

        proj_data_fund = []
        for gx, gy, vxy, pxy, edge in proj_entries_fund:
            (ax, ay), audited_edge = proj_geometry.audit_near_vertex_projection_with_edge(
                (tx_x, tx_y),
                vxy,
                pxy,
                edge,
                _edge_set,
                buildings=_buildings,
            )
            fund_proj_points[k, gx, gy, 0] = ax
            fund_proj_points[k, gx, gy, 1] = ay
            proj_data_fund.append((vxy, (ax, ay), audited_edge))

        # 构建 street mask：用投影点 + 建筑物边 + 地图边界构建可视多边形
        street_poly_xy = proj_geometry.build_visibility_mask(
            (tx_x, tx_y), proj_data_fund, _adjacency, _buildings,
            edge_set=_edge_set,
        )
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

    _t_fund_postproc = time.perf_counter() - _t0

    # 保存 Fundamental 数据
    _t0 = time.perf_counter()
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
    _t_save = time.perf_counter() - _t0
    _t_total = time.perf_counter() - _t_map_start

    timing_row = {
        "map_id":           args.map_id,
        "t_model_load_s":   f"{_t_model_load:.3f}",
        "t_data_load_s":    f"{_t_data_load:.3f}",
        "t_sel_infer_s":    f"{_t_sel_infer:.3f}",
        "t_sel_postproc_s": f"{_t_sel_postproc:.3f}",
        "t_fund_infer_s":   f"{_t_fund_infer:.3f}",
        "t_fund_postproc_s":f"{_t_fund_postproc:.3f}",
        "t_save_s":         f"{_t_save:.3f}",
        "t_total_map_s":    f"{_t_total:.3f}",
    }
    _append_timing_row(args.timing_csv, timing_row)

    print(f"\n── 时间统计 ──────────────────────────────────")
    print(f"  模型加载      : {_t_model_load:7.1f}s")
    print(f"  静态数据加载  : {_t_data_load:7.1f}s")
    print(f"  Selected推理  : {_t_sel_infer:7.1f}s")
    print(f"  Selected后处理: {_t_sel_postproc:7.1f}s")
    print(f"  Fund推理      : {_t_fund_infer:7.1f}s")
    print(f"  Fund后处理    : {_t_fund_postproc:7.1f}s")
    print(f"  保存文件      : {_t_save:7.1f}s")
    print(f"  总计          : {_t_total:7.1f}s")
    print(f"  写入CSV       : {args.timing_csv}")
    print(f"\n✓ 推理完成  map_id={args.map_id}  output_root={args.output_root}")

if __name__ == "__main__":
    main()
