# Toy Field II Dataset

This folder contains a small, reproducible ultrasound simulation dataset for validating the initial pipeline:

simulation -> channel RF data -> save `.mat` -> load later for DAS or model experiments.

It uses Field II and SI units throughout: meters, seconds, and Hz. Coordinates follow `[x, y, z]`, where `x` is lateral, `y` is elevation, and `z` is depth.

## Samples

`simulate_toy_dataset.m` generates 5 samples:

1. `speckle`
2. `points`
3. `cyst_center`
4. `cyst_lateral`
5. `cyst_small`

Each sample uses deterministic seeds where random scatterers are generated.

## Run

From MATLAB:

```matlab
cd simulations/fieldii
simulate_toy_dataset
```

Field II must be available on the MATLAB path. The script uses `FIELDII_PATH` if defined; otherwise it looks for `../../Field_II_ver_3_30_linux` relative to the repo root.

On the cluster, submit:

```bash
sbatch scripts/slurm/run_toy_dataset.sh
```

## Output

Files are written under the repo root:

```text
data/simulated/toy_fieldii/
```

The generated files are:

```text
sample_001_speckle.mat
sample_002_points.mat
sample_003_cyst_center.mat
sample_004_cyst_lateral.mat
sample_005_cyst_small.mat
```

Each `.mat` contains one variable:

```matlab
sample
```

with:

```matlab
sample.phantom.positions      % [N x 3], columns [x, y, z] in meters
sample.phantom.amplitudes     % [N x 1]
sample.channel_data.rf        % [num_samples x num_elements]
sample.channel_data.t0        % Field II RF start time [s]
sample.cfg
sample.metadata
```

This toy dataset does not include DAS, IQ, training targets, or model outputs. It is only intended to validate the first simulation and data-loading pipeline.

## Inspect

After generating the dataset:

```matlab
cd simulations/fieldii
inspect_toy_sample
```

This loads `sample_002_points.mat` by default, prints metadata and RF dimensions, and saves quick-look PNGs for RF and scatterer positions.
