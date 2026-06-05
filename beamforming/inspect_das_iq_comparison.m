%INSPECT_DAS_IQ_COMPARISON Compare real DAS and DAS-IQ B-mode images.
%   Loads all data/beamformed/toy_fieldii/sample_*_das.mat files and saves
%   side-by-side PNGs in data/beamformed/toy_fieldii/inspection_iq_comparison.
%   This script does not modify the .mat files or rerun beamforming.

clear;
clc;

script_dir = fileparts(mfilename('fullpath'));
repo_root = normalize_path(fullfile(script_dir, '..'));

input_dir = fullfile(repo_root, 'data', 'beamformed', 'toy_fieldii');
figure_dir = fullfile(input_dir, 'inspection_iq_comparison');

if ~exist(input_dir, 'dir')
    error('inspect_das_iq_comparison:MissingInput', ...
        'Input directory not found: %s. Run beamform_toy_dataset first.', input_dir);
end

if ~exist(figure_dir, 'dir')
    mkdir(figure_dir);
end

files = dir(fullfile(input_dir, 'sample_*_das.mat'));
if isempty(files)
    error('inspect_das_iq_comparison:MissingInput', ...
        'No sample_*_das.mat files found in %s.', input_dir);
end

fprintf('Comparing DAS real and DAS-IQ B-mode for %d samples.\n', numel(files));
fprintf('Output directory: %s\n\n', figure_dir);
fprintf(['sample | corr(bmode,bmode_iq) | mean_abs_diff_db | ', ...
    'max_abs_diff_db | bmode_range_db | bmode_iq_range_db | figure\n']);

for idx = 1:numel(files)
    sample_path = fullfile(files(idx).folder, files(idx).name);
    loaded = load(sample_path);

    if ~isfield(loaded, 'sample')
        error('inspect_das_iq_comparison:InvalidSample', ...
            'File does not contain variable sample: %s', sample_path);
    end

    sample = loaded.sample;
    validate_sample(sample, sample_path);

    bmode_db = sample.das.bmode_db;
    bmode_iq_db = sample.das.bmode_iq_db;
    abs_diff_db = abs(bmode_db - bmode_iq_db);

    corr_value = corr(bmode_db(:), bmode_iq_db(:));
    mean_abs_diff_db = mean(abs_diff_db(:));
    max_abs_diff_db = max(abs_diff_db(:));
    bmode_range = [min(bmode_db(:)), max(bmode_db(:))];
    bmode_iq_range = [min(bmode_iq_db(:)), max(bmode_iq_db(:))];

    x_mm = sample.das.x_axis * 1e3;
    z_mm = sample.das.z_axis * 1e3;
    dynamic_range_db = get_dynamic_range(sample.das);

    [~, base_name] = fileparts(files(idx).name);
    base_name = regexprep(base_name, '_das$', '');
    figure_path = fullfile(figure_dir, ...
        sprintf('%s_bmode_vs_bmode_iq.png', base_name));

    fig = figure('Visible', 'off', 'Position', [100, 100, 1500, 450]);

    subplot(1, 3, 1);
    imagesc(x_mm, z_mm, bmode_db);
    axis image;
    set(gca, 'YDir', 'reverse');
    colormap(gca, gray);
    colorbar;
    caxis([-dynamic_range_db, 0]);
    xlabel('x lateral [mm]');
    ylabel('z depth [mm]');
    title(sprintf('DAS real: %s', sample.name), 'Interpreter', 'none');

    subplot(1, 3, 2);
    imagesc(x_mm, z_mm, bmode_iq_db);
    axis image;
    set(gca, 'YDir', 'reverse');
    colormap(gca, gray);
    colorbar;
    caxis([-dynamic_range_db, 0]);
    xlabel('x lateral [mm]');
    ylabel('z depth [mm]');
    title('DAS-IQ');

    subplot(1, 3, 3);
    imagesc(x_mm, z_mm, abs_diff_db);
    axis image;
    set(gca, 'YDir', 'reverse');
    colormap(gca, parula);
    colorbar;
    caxis([0, max(1, prctile(abs_diff_db(:), 99))]);
    xlabel('x lateral [mm]');
    ylabel('z depth [mm]');
    title('|DAS - DAS-IQ| [dB]');

    saveas(fig, figure_path);
    close(fig);

    fprintf('%s | %.4f | %.3f | %.3f | [%.1f, %.1f] | [%.1f, %.1f] | %s\n', ...
        base_name, corr_value, mean_abs_diff_db, max_abs_diff_db, ...
        bmode_range(1), bmode_range(2), bmode_iq_range(1), bmode_iq_range(2), ...
        figure_path);
end

function validate_sample(sample, sample_path)
required_top = {'das', 'metadata'};
for idx = 1:numel(required_top)
    if ~isfield(sample, required_top{idx})
        error('inspect_das_iq_comparison:InvalidSample', ...
            'sample.%s is required in %s.', required_top{idx}, sample_path);
    end
end

required_das = {'bmode_db', 'bmode_iq_db', 'x_axis', 'z_axis'};
for idx = 1:numel(required_das)
    if ~isfield(sample.das, required_das{idx})
        error('inspect_das_iq_comparison:InvalidSample', ...
            'sample.das.%s is required in %s.', required_das{idx}, sample_path);
    end
end

if ~isequal(size(sample.das.bmode_db), size(sample.das.bmode_iq_db))
    error('inspect_das_iq_comparison:InvalidSample', ...
        'sample.das.bmode_db and sample.das.bmode_iq_db must have the same size in %s.', ...
        sample_path);
end

if any(~isfinite(sample.das.bmode_db(:))) || any(~isfinite(sample.das.bmode_iq_db(:)))
    error('inspect_das_iq_comparison:InvalidSample', ...
        'B-mode images must not contain NaN or Inf values in %s.', sample_path);
end
end

function dynamic_range_db = get_dynamic_range(das)
dynamic_range_db = 60;
if isfield(das, 'metadata') && isfield(das.metadata, 'dynamic_range_db') && ...
        ~isempty(das.metadata.dynamic_range_db)
    dynamic_range_db = das.metadata.dynamic_range_db;
end
end

function normalized_path = normalize_path(path_value)
current_dir = pwd;
cd(path_value);
normalized_path = pwd;
cd(current_dir);
end
