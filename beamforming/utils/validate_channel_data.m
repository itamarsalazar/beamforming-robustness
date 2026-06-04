function channel_data = validate_channel_data(channel_data)
%VALIDATE_CHANNEL_DATA Validate and normalize channel RF dimensions.
%   RF is returned as [num_time_samples x num_elements]. If it appears
%   transposed relative to element_positions, it is transposed with warning.

required_fields = {'rf', 'fs', 'c', 't0', 'element_positions'};
for idx = 1:numel(required_fields)
    field_name = required_fields{idx};
    if ~isfield(channel_data, field_name)
        error('validate_channel_data:MissingField', ...
            'channel_data.%s is required.', field_name);
    end
end

if ~isnumeric(channel_data.rf) || ndims(channel_data.rf) ~= 2
    error('validate_channel_data:InvalidRF', ...
        'channel_data.rf must be a numeric 2-D matrix.');
end

if any(isnan(channel_data.rf(:)))
    error('validate_channel_data:InvalidRF', ...
        'channel_data.rf must not contain NaNs.');
end

if ~is_positive_scalar(channel_data.fs) || ~is_positive_scalar(channel_data.c)
    error('validate_channel_data:InvalidScalar', ...
        'channel_data.fs and channel_data.c must be positive scalars.');
end

if ~isnumeric(channel_data.t0) || ~isscalar(channel_data.t0) || isnan(channel_data.t0)
    error('validate_channel_data:InvalidScalar', ...
        'channel_data.t0 must be a numeric scalar.');
end

element_positions = channel_data.element_positions;
if ~isnumeric(element_positions) || size(element_positions, 2) ~= 3
    error('validate_channel_data:InvalidElements', ...
        'channel_data.element_positions must be [num_elements x 3].');
end

if any(isnan(element_positions(:)))
    error('validate_channel_data:InvalidElements', ...
        'channel_data.element_positions must not contain NaNs.');
end

num_elements = size(element_positions, 1);
rf_size = size(channel_data.rf);

if rf_size(2) == num_elements
    return;
end

if rf_size(1) == num_elements
    warning('validate_channel_data:TransposedRF', ...
        ['channel_data.rf appears transposed. Converting from ', ...
         '[num_elements x num_time_samples] to [num_time_samples x num_elements].']);
    channel_data.rf = channel_data.rf.';
    return;
end

error('validate_channel_data:DimensionMismatch', ...
    ['channel_data.rf must be [num_time_samples x num_elements]. ', ...
     'Found RF [%d x %d] and %d element positions.'], ...
    rf_size(1), rf_size(2), num_elements);
end

function tf = is_positive_scalar(value)
tf = isnumeric(value) && isscalar(value) && value > 0 && ~isnan(value);
end
