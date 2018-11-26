function [tss, labels] = importTimeSeries(dataset, sfx, path)
%dataset: e.g. 'MALLAT'
%sfx: 'TRAIN' or 'TEST'
%path: the dataset path

if(~exist('path', 'var'));
    path = dataset;
end

fileName = fullfile(path, [dataset, '_', sfx]);
tss = importdata(fileName);
labels = tss(:, 1);
tss = tss(:, 2 : end);
