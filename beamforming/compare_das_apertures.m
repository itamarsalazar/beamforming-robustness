%COMPARE_DAS_APERTURES Compare full and dynamic aperture DAS on one sample.
%   If sample_file is not defined before running, this script prefers the
%   point-target toy sample. It does not modify simulated or beamformed data.

clearvars -except sample_file;
clc;

% sample_file = 'Q:\isalazar\codes\beamforming-robustness\data\simulated\toy_fieldii\sample_001_speckle.mat'
% sample_file = 'Q:\isalazar\codes\beamforming-robustness\data\simulated\toy_fieldii\sample_002_points.mat'
% sample_file = 'Q:\isalazar\codes\beamforming-robustness\data\simulated\toy_fieldii\sample_003_cyst_center.mat'
% sample_file = 'Q:\isalazar\codes\beamforming-robustness\data\simulated\toy_fieldii\sample_004_cyst_lateral.mat'
% sample_file = 'Q:\isalazar\codes\beamforming-robustness\data\simulated\toy_fieldii\sample_005_cyst_small.mat'


script_dir = fileparts(mfilename('fullpath'));
repo_root = normalize_path(fullfile(script_dir, '..'));

addpath(script_dir);
addpath(fullfile(script_dir, 'utils'));

input_dir = fullfile(repo_root, 'data', 'simulated', 'toy_fieldii');
figure_dir = fullfile(repo_root, 'outputs', 'figures', 'toy_fieldii_das');

if ~exist(input_dir, 'dir')
    error('compare_das_apertures:MissingInput', ...
        'Input directory not found: %s. Run simulate_toy_dataset first.', input_dir);
end

if ~exist('sample_file', 'var') || isempty(sample_file)
    preferred = fullfile(input_dir, 'sample_002_points.mat');
    if exist(preferred, 'file')
        sample_file = preferred;
    else
        files = dir(fullfile(input_dir, 'sample_*.mat'));
        if isempty(files)
            error('compare_das_apertures:MissingInput', ...
                'No sample_*.mat files found in %s.', input_dir);
        end
        sample_file = fullfile(files(1).folder, files(1).name);
    end
elseif isfolder(sample_file)
    files = dir(fullfile(sample_file, 'sample_*.mat'));
    if isempty(files)
        error('compare_das_apertures:MissingInput', ...
            'No sample_*.mat files found in %s.', sample_file);
    end
    sample_file = fullfile(files(1).folder, files(1).name);
elseif ~exist(sample_file, 'file')
    candidate = fullfile(input_dir, sample_file);
    if exist(candidate, 'file')
        sample_file = candidate;
    else
        error('compare_das_apertures:MissingInput', ...
            'Sample file not found: %s.', sample_file);
    end
end

if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

loaded = load(sample_file);
if ~isfield(loaded, 'sample')
    error('compare_das_apertures:InvalidSample', ...
        'File does not contain variable sample: %s', sample_file);
end

sample = loaded.sample;
cfg = sample.cfg;
grid = make_default_grid(cfg);

cfg_full = cfg;
cfg_full.das.aperture_mode = 'full';
cfg_full.das.apodization = 'hanning';
cfg_full.das.dynamic_range_db = 60;

das_full = das_beamform(sample.channel_data, cfg_full, grid);

cfg_dynamic = cfg;
cfg_dynamic.das.aperture_mode = 'dynamic';
cfg_dynamic.das.f_number = 1;
cfg_dynamic.das.apodization = 'hanning';
cfg_dynamic.das.dynamic_range_db = 60;

das_dynamic = das_beamform(sample.channel_data, cfg_dynamic, grid);

rf_size = size(sample.channel_data.rf);
image_size = size(das_full.bmode_db);

[~, base_name] = fileparts(sample_file);
figure_name = sprintf('compare_aperture_%s.png', base_name);
figure_path = fullfile(figure_dir, figure_name);

x_mm = grid.x_axis * 1e3;
z_mm = grid.z_axis * 1e3;
dynamic_range_db = 60;

fig = figure('Visible', 'off');

subplot(1, 2, 1);
imagesc(x_mm, z_mm, das_full.bmode_db);
axis image;
set(gca, 'YDir', 'reverse');
colormap(gray);
caxis([-dynamic_range_db, 0]);
colorbar;
xlabel('x lateral [mm]');
ylabel('z depth [mm]');
title('Full aperture');

subplot(1, 2, 2);
imagesc(x_mm, z_mm, das_dynamic.bmode_db);
axis image;
set(gca, 'YDir', 'reverse');
colormap(gray);
caxis([-dynamic_range_db, 0]);
colorbar;
xlabel('x lateral [mm]');
ylabel('z depth [mm]');
%title('Dynamic aperture, F-number = 1.5');
title(sprintf('Dynamic aperture, F-number = %.1f', das_dynamic.metadata.f_number));

sgtitle(sprintf('DAS aperture comparison: %s', sample.name), 'Interpreter', 'none');
saveas(fig, figure_path);
close(fig);

fprintf('Sample: %s\n', sample.name);
fprintf('Sample file: %s\n', sample_file);
fprintf('RF size: [%d x %d]\n', rf_size(1), rf_size(2));
fprintf('Image size: [%d x %d]\n', image_size(1), image_size(2));
fprintf('Full aperture mode: %s\n', das_full.metadata.aperture_mode);
fprintf('Dynamic aperture mode: %s\n', das_dynamic.metadata.aperture_mode);
fprintf('Dynamic F-number: %.2f\n', das_dynamic.metadata.f_number);
fprintf('Saved figure: %s\n', figure_path);

function grid = make_default_grid(cfg)
grid = struct();
grid.x_axis = linspace(cfg.x_limits(1), cfg.x_limits(2), 256);
grid.z_axis = linspace(cfg.z_limits(1), cfg.z_limits(2), 512);
grid.metadata = struct();
grid.metadata.x_axis_units = 'm';
grid.metadata.z_axis_units = 'm';
grid.metadata.dimensions = '[Nz x Nx]';
end

function normalized_path = normalize_path(path_value)
current_dir = pwd;
cd(path_value);
normalized_path = pwd;
cd(current_dir);
end
