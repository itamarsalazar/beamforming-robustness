function sample_path = save_sample(sample, output_dir, sample_id)
%SAVE_SAMPLE Save one simulated sample structure to a MAT file.
%   Only the variable "sample" is written to disk.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

sample_name_tag = char(sample.name);
sample_name = sprintf('sample_%03d_%s.mat', sample_id, sample_name_tag);
sample_path = fullfile(output_dir, sample_name);

save(sample_path, 'sample');
end
