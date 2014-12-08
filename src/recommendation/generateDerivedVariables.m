function [userDV, artistDV] = generateDerivedVariables(Y)
% GENERATEDERIVEDVARIABLES
% Derived Variables (DV) used follow the DV proposed by:
%   Park, Yoon-Joo, and Alexander Tuzhilin.
%   "The long tail of recommender systems and how to leverage it."
%
% We define "popularity" of an item as its number
% of observed listening counts.
%
% INPUT
%   Y: The (n x d) listening counts matrix
% OUTPUT
%   userDV: (n x 8) The 8 following DVs:
%     1. Average rating given by that user
%     2. Number of ratings given by that user
%     3. Average popularity of items rated by that user
%     4. Average of average ratings of items rated by that user
%     5. Average popularity of items which this user "liked"
%        (i.e. rated higher than his average rating)
%     6. Average of average ratings of "liked" items
%     7. Average popularity of items which this user "disliked"
%        (i.e. rated lower than his average rating)
%     8. Average of average ratings of "disliked" items
%
%  artistDV: (n x 3) The 3 following DVs:
%     9. Average listening count for this artist
%     10. Popularity (i.e. number of observed counts) for this artist
%     11. Measure of "likability": average difference between the rating
%         received and the rating's user's average

  [N, D] = size(Y);
  [uIdx, aIdx] = find(Y);
  usersIdx = unique(uIdx);
  artistsIdx = unique(aIdx);

  userDV = zeros(N, 8);
  artistDV = zeros(D, 3);

  % For each artist
  for j = 1:length(artistsIdx)
    artist = artistsIdx(j);

    artistDV(artist, 1) = mean(nonzeros(Y(:, artist)));
    artistDV(artist, 2) = nnz(Y(:, artist));
  end;

  % For each user
  for i = 1:length(usersIdx)
    user = usersIdx(i);
    artistsRated = aIdx(uIdx == user);
    
    counts = full(Y(user, artistsRated));
    nnzCounts = nonzeros(counts);
    assert(nnz(counts) == length(artistsRated));
    
    averageCount = mean(nnzCounts);

    userDV(user, 1) = averageCount;
    userDV(user, 2) = length(artistsRated);
    userDV(user, 3) = mean(artistDV(artistsRated, 2));
    userDV(user, 4) = mean(artistDV(artistsRated, 1));

    likedLogical = counts >= averageCount;
    liked = find(likedLogical);
    userDV(user, 5) = mean(artistDV(liked, 2));
    userDV(user, 6) = mean(artistDV(liked, 1));

    disliked = find(~likedLogical);
    userDV(user, 7) = mean(artistDV(disliked, 2));
    userDV(user, 8) = mean(artistDV(disliked, 1));
  end;

  % For each artist, compute the "likability"
  for j = 1:length(artistsIdx)
    artist = artistsIdx(j);
    ratedBy = uIdx(aIdx == artist);

    artistDV(artist, 3) = mean( Y(ratedBy, artist) - userDV(ratedBy, 1) );
  end;
end
