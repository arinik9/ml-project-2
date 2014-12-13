 %Store your predictions in a mat file named 'songPred.mat'. 
% This mat file should contain two matrices 'Ytest_weak_pred'
% and 'Ytest_strong_pred'. It is clear what names mean.
% Ytest_weak_pred should be a sparse matrix of size 1774x15082
% Ytest_stong_pred should be a sparse matrix of size 93x15082
% One way to create these matrices is shown below:
%
%   load songTestPairs;
%   Ytest_strong_pred = Ytest_strong_pairs;
%   Ytest_weak_pred = Ytest_weak_pairs;
%   save('songPred', 'Ytest_strong_pred', 'Ytest_weak_pred');
%
% You can then fill in your predictions in the non-zero entries of the two matrices.
% Once you have the file, running this file will check if the sizes are correct or not.

clear all;
load songPred; % your file
load songTestPairs; % test data

% for  Ytest_weak_pred
% exists?
if ~exist('Ytest_weak_pred', 'var')
  error('no Ytest_weak_pred matrix');
end
% check size
[nU,nA] = size(Ytest_weak_pred);
if nU ~= 1774 & nA ~= 15082
  error('size of the matrix is not correct');
end
% check num of non-zero entries
if nnz(Ytest_weak_pred) ~= 15924 
  error('number of non-zero entries is incorrect');
end

% for Ytest_strong_pred
% exists?
if ~exist('Ytest_strong_pred', 'var')
  error('no Ytest_strong_pred matrix');
end
% check size
[nU,nA] = size(Ytest_strong_pred);
if nU ~= 93 & nA ~= 15082
  error('size of the matrix is not correct');
end
% check num of non-zero entries
if nnz(Ytest_strong_pred) ~= 4440
  error('number of non-zero entries is incorrect');
end

fprintf('successful, you can submit\n');

