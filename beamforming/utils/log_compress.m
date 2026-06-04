function bmode_db = log_compress(envelope, dynamic_range_db)
%LOG_COMPRESS Normalize envelope data and convert to clipped dB.
%   bmode_db is clipped to [-dynamic_range_db, 0].

if nargin < 2 || isempty(dynamic_range_db)
    dynamic_range_db = 60;
end

if dynamic_range_db <= 0 || isnan(dynamic_range_db)
    error('log_compress:InvalidDynamicRange', ...
        'dynamic_range_db must be a positive scalar.');
end

if any(isnan(envelope(:)))
    error('log_compress:InvalidEnvelope', ...
        'envelope must not contain NaNs.');
end

max_value = max(envelope(:));
if max_value <= 0
    envelope_norm = zeros(size(envelope));
else
    envelope_norm = envelope ./ max_value;
end

bmode_db = 20 * log10(envelope_norm + eps);
bmode_db = max(bmode_db, -dynamic_range_db);
bmode_db = min(bmode_db, 0);
end
