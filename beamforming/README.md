# Beamforming

This module implements a minimal classical delay-and-sum (DAS) beamformer for the toy Field II dataset.

It is intended to close the first reproducible pipeline:

```text
Field II channel RF -> DAS image -> envelope -> log compression -> inspection
```

This is not the deep learning model and not the final robustness experiment.

## Input

The expected input is the simulated toy Field II dataset:

```text
data/simulated/toy_fieldii/
```

Each `.mat` file should contain one `sample` structure with `sample.channel_data.rf`, `sample.channel_data.t0`, element positions, `sample.phantom`, and `sample.cfg`.

## Run DAS

From MATLAB:

```matlab
cd beamforming
beamform_toy_dataset
```

Outputs are written to:

```text
data/beamformed/toy_fieldii/
```

The generated files are named like:

```text
sample_001_speckle_das.mat
sample_002_points_das.mat
```

Each beamformed `.mat` contains one variable:

```matlab
sample
```

with:

```matlab
sample.channel_data.rf      % [num_time_samples x num_elements]
sample.phantom
sample.das.rf_image         % [Nz x Nx]
sample.das.envelope         % [Nz x Nx]
sample.das.bmode_db         % [Nz x Nx]
sample.das.x_axis           % lateral axis [m]
sample.das.z_axis           % depth axis [m]
sample.metadata
```

## Inspect

After beamforming:

```matlab
cd beamforming
inspect_das_sample
```

By default, this loads the first `*_das.mat` file. To inspect a specific sample:

```matlab
sample_file = 'sample_002_points_das.mat';
inspect_das_sample
```

Figures are written to:

```text
outputs/figures/toy_fieldii_das/
```

The inspector saves:

```text
*_bmode.png
*_scatterers.png
*_rf.png
```

## Notes

The current DAS implementation assumes a linear array and a single plane-wave transmit at 0 degrees. It uses SI units internally. There is no dynamic aperture, no learned model, and no robustness experiment yet.

## Aperture Modes

`das_beamform` supports two receive aperture modes through `cfg.das`:

```matlab
cfg.das.aperture_mode = 'full';      % all elements contribute
cfg.das.aperture_mode = 'dynamic';   % aperture_size = z / f_number
cfg.das.f_number = 1.5;
cfg.das.apodization = 'hanning';     % or 'rectangular'
```

`beamform_toy_dataset` uses dynamic aperture by default:

```matlab
cfg.das.aperture_mode = 'dynamic';
cfg.das.f_number = 1.5;
cfg.das.apodization = 'hanning';
cfg.das.dynamic_range_db = 60;
```

The output format is unchanged: `sample.das.rf_image`, `sample.das.envelope`, `sample.das.bmode_db`, axes, grid, and metadata are still saved in each beamformed `.mat` file.

## Compare Full vs Dynamic Aperture

To generate a side-by-side comparison on the same sample and grid:

```matlab
cd beamforming
compare_das_apertures
```

By default this uses `sample_002_points.mat` if available. You can choose another simulated sample before running:

```matlab
sample_file = 'sample_004_cyst_lateral.mat';
compare_das_apertures
```

The comparison uses:

```matlab
cfg_full.das.aperture_mode = 'full';
cfg_full.das.apodization = 'hanning';

cfg_dynamic.das.aperture_mode = 'dynamic';
cfg_dynamic.das.f_number = 1.5;
cfg_dynamic.das.apodization = 'hanning';
```

Figures are saved under:

```text
outputs/figures/toy_fieldii_das/
```
