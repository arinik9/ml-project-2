% --- PERSON DETECTION PREDICITON CHECK ----
%
% Store your predictions in a mat file named 'personPred.mat'. 
% This mat file should contain a vector 'Ytest_score'
% which contains the prediction _score_ for each test sample.
% The size of Ytest_score must be 8743x1.
% One way to create these matrices is shown below:
%
%   -- assign Ytest_score first according to your classifier --
%   - and then:
%   save('personPred', 'Ytest_score');
%
% Once you have the file, running this file will check if the sizes are correct or not.

clear all;
load personPred.mat;

% exists?
if ~exist('Ytest_score', 'var')
  error('No Ytest_score vector found');
end

% check size
[nU,nA] = size(Ytest_score);
if (nU ~= 8743) && (nA ~= 1)
  error('size of Ytest_score is incorrect');
end

% make sure it contains scores and not labels
nUniqueVals = length(unique(Ytest_score(:)));
if nUniqueVals < 4
    error('Ytest_score should be scores and not a binary prediction');
end

fprintf('\nSuccessful, you can submit!\n\n');
