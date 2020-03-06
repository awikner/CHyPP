function u = GetDataChunk(dataindices,filehandle,field,chunk_size,...
    locality,chunk_begin,chunk_end,rear_overlap,forward_overlap)
% GetDataChunk - function for obtaining the chunk of time series data to be
%                used by a single worker during training
% Inputs:
%   dataindices - time indices of data to be loaded in
%
%   filehandle - full file path to the .mat data file
%
%   field - field of the data file to be accessed
%
%   chunk_size - spatial size of the data to be predicted by this worker
%
%   locality - local overlap on either side of the data chunk
%
%   chunk_begin - index denoting where in full data the chunk begins
%
%   chunk_end - index denoting where in full data the chunk ends
%
%   rear_overlap - indices for rear overlap data
%
%   forward_overlap - indices for forward overlap data
%
% Outputs
%   u - data chunk of size chunk_size+2*locality by length(dataindices)

u = zeros(chunk_size + 2*locality, numel(dataindices)); % this will be populated by the input data to the reservoir

%If locality is nonzero, first load in overlap data
if locality > 0
    if rear_overlap(end)<rear_overlap(1)
        u(1:locality,:) = [filehandle.(field)(dataindices, ...
            rear_overlap(rear_overlap>rear_overlap(end))),...
            filehandle.(field)(dataindices, ...
            rear_overlap(rear_overlap<=rear_overlap(end)))]';
    else
        u(1:locality,:) = filehandle.(field)(dataindices,rear_overlap)';
    end

    if forward_overlap(end) < forward_overlap(1)
        u(locality+chunk_size+1:2*locality+chunk_size,:) = ...
            [filehandle.(field)(dataindices,...
            forward_overlap(forward_overlap>forward_overlap(end))),...
            filehandle.(field)(dataindices,...
            forward_overlap(forward_overlap<=forward_overlap(end)))]';
    else
        u(locality+chunk_size+1:2*locality+chunk_size,:) = ...
            filehandle.(field)(dataindices,forward_overlap)';
    end
end

%Load in data in the prediction chunk
u(locality+1:locality+chunk_size,:) = ...
    filehandle.(field)(dataindices, chunk_begin:chunk_end)';