addpath(genpath('./data'));
addpath(genpath('./src'));
addpath(genpath('./test'));
addpath(genpath('./toolbox'));

% Enable local parallel computation (leverage multicore CPU)
parpool;