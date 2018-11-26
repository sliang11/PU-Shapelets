%Generate initial positive labeled examples
function seeds = generateSeeds(dataset, numSeeds, path)
%dataset: e.g. 'MALLAT'
%numSeeds: number of seeds to generate, default 10
%path: dataset path

if ~exist('numSeeds', 'var')
    numSeeds = 10;
end

if(~exist('path', 'var'));
    path = dataset;
end


[~, labels] = importTimeSeries(dataset, 'TRAIN', path);
pInds = (find(labels == 1))';
numP = length(pInds);

numLabeled = max(round(numP * 0.1), 1);
seeds = zeros(numSeeds, numLabeled);
for i = 1 : numSeeds
    seed = pInds(sort(random('unid', numP, 1, numLabeled)));
	while(ismember(seed, seeds, 'rows') || length(unique(seed)) < numLabeled)
		seed = pInds(sort(random('unid', numP, 1, numLabeled)));
	end
	seeds(i, :) = seed;
end

fName = fullfile(path, ['seeds_', dataset, '.txt']);
fid = fopen(fName, 'w');
for i = 1 : numSeeds
    for j = 1 : numLabeled
        fprintf(fid, '%d ', seeds(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);