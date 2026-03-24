# D^2LoS
The code of D^2LoS

## Repository Structure

```
├── training/
│   └── train.py                    # Model training (SuperResUNet)
├── inference/
│   ├── infer_with_geom.py          # Inference with geometric correction
│   └── infer_without_geom.py       # Inference without geometric correction
├── conversion/
│   ├── propbin_to_aps_pdp.py       # Convert propbin to APS/PDP numpy arrays
│   └── apply_sinc_beam.py          # Apply sinc antenna beam pattern to propbin
├── visualization/
│   ├── visualize_rss.py            # Visualize a single RSS heatmap
│   └── visualize_aps_pdp.py        # Visualize a single APS/PDP curve
└── utils/
    ├── proj_geometry.py            # Geometric correction module
    ├── propbin_reader.py           # Read .propbin / .propbin.gz files
    └── propbin_writer.py           # Write .propbin v2 files
```

## Requirements

```
torch >= 2.0
numpy
opencv-python
tqdm
lmdb
Pillow
matplotlib
```

## Usage

### 1. Training

```bash
python training/train.py \
    --lmdb_path /path/to/dataset.lmdb \
    --raw_root  /path/to/data_root \
    --gpu 0 \
    --batch_size 64 \
    --epochs 120 \
    --lr 0.0002
```

Checkpoints are saved to `./ckpt/` by default.

### 2. Inference

**With geometric correction** (full D²LoS pipeline):

```bash
python inference/infer_with_geom.py \
    --checkpoint /path/to/best.pth \
    --data-root  /path/to/data_root \
    --output-root /path/to/output \
    --map-id-start 0 --map-id-end 99 \
    --gpu-id 0
```

**Without geometric correction** (ablation):

```bash
python inference/infer_without_geom.py \
    --checkpoint /path/to/best.pth \
    --data-root  /path/to/data_root \
    --output-root /path/to/output \
    --map-id-start 0 --map-id-end 99 \
    --gpu-id 0
```

### 3. APS / PDP Conversion

Convert propagation binary files to Angular Power Spectrum (APS) and Power Delay Profile (PDP) numpy arrays:

```bash
python conversion/propbin_to_aps_pdp.py
```

Configuration (grid resolution, input/output paths) is set inside the script.

### 4. Antenna Beam Pattern

Apply a sinc-shaped antenna beam pattern to propbin files:

```bash
python conversion/apply_sinc_beam.py \
    --input-root /path/to/propbin_dir \
    --output-root /path/to/beamed_output \
    --map-id-start 0 --map-id-end 99 \
    --tx-boresight-az 0.0 \
    --tx-boresight-el 0.0 \
    --az-mainlobe-width 30.0 \
    --el-mainlobe-width 30.0
```

### 5. Visualization

**RSS heatmap** from a propbin file:

```bash
python visualization/visualize_rss.py /path/to/source_0.propbin.gz --map-id 0
```

**APS / PDP curves** from numpy files:

```bash
python visualization/visualize_aps_pdp.py \
    --root /path/to/aps_pdp_dir \
    --name "aps_0_100_200_150_180"
```

## Citation

```bibtex
@article{d2los2026,
  title   = {D$^2$LoS: ...},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

## License

This project is released under the MIT License.
