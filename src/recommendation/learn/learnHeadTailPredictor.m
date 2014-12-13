function predictor = learnHeadTailPredictor(Y, Ytest, userDV, artistDV, headThreshold)
% LEARNHEADTAILPREDICTOR This predictor separates the dataset into "head" and "tail" artist
% Head contains artists for which we have many observations,
% while Tail contains artists which we do not know well.
% Both parts are handled separately.
%
% INPUT
%   headThreshold: Minimum number of observations required for an artist
%                  to be part of the head
%
% SEE ALSO
%   Park, Yoon-Joo, and Alexander Tuzhilin.
%   "The long tail of recommender systems and how to leverage it."

    % TODO: take the social network in input!
    G = sparse(size(Y, 1), size(Y, 1));

    headPredictor = learnHeadPredictor(Y, G, Ytest, userDV, artistDV, headThreshold);
    tailPredictor = learnTailPredictor(Y, G, Ytest, userDV, artistDV, headThreshold);
    predictor = @(user, artist) predict(user, artist, artistDV(:, 2), headThreshold, headPredictor, tailPredictor);
end

function prediction = predict(user, artist, popularity, headThreshold, head, tail)
% At prediction time, choose either head or tail predictor
    if(popularity(artist) >= headThreshold)
        prediction = head(user, artist);
    else
        prediction = tail(user, artist);
        %fprintf('Tail predictor for artist %d --> %f\n', artist, prediction);
    end;
end

function predictor = learnHeadPredictor(Y, G, Ytest, userDV, artistDV, headThreshold)
    % Each prediction uses a model learnt from all users having listened to the artist
    % TODO: `modelEachArtist` shouldn't be aware of `headThreshold`
    betas = modelEachArtist(Y, G, headThreshold, userDV, artistDV);
    getFeatures = @(user, artist) generateFeatures(artist, user, G, userDV, artistDV);

    predictor = @(user, artist) getFeatures(user, artist) * betas(:, artist);
end

function predictor = learnTailPredictor(Y, G, Ytest, userDV, artistDV, headThreshold)
% Since artists in the tail have very little information available,
% we cluster them together and learn a model per cluster instead of one per artist.
% Clustering is done in the Derived Variables space.
% SEE ALSO
%   Park, Yoon-Joo, and Alexander Tuzhilin.
%   "The long tail of recommender systems and how to leverage it."

    nArtists = size(artistDV, 1);

    % TODO: what happens to artists having 0 counts? Maybe we should handle
    % them explicitely?
    tailArtists = find((artistDV(:, 2) < headThreshold));
    tailSpace = artistDV(tailArtists, :);

    % Clustering
    % TODO: tweak the number of clusters
    % TODO: use more info to cluster on?
    % TODO: use GMM (soft clustering)?
    K = 20;
    [tailClusters, ~] = kmeans2(tailSpace, K);

    % Convert cluster assignments back to "all artists" indexing
    clusters = zeros(nArtists, 1) - 1;
    clusters(tailArtists) = tailClusters;

    fprintf('The tail has %d artists (cutoff at %d).\n', length(tailArtists), headThreshold);

    % We now learn a common model per cluster
    betas = cell(K, 1);
    for k = 1:K
        artistsInThisCluster = find(clusters == k);
        [uIdx, ~] = find(Y(:, artistsInThisCluster));
        correspondingUsers = unique(uIdx);

        % TODO: more subtle target for the learning?
        Ysub = Y(:, artistsInThisCluster);
        y = zeros(length(correspondingUsers), 1);
        for i = 1:length(correspondingUsers)
            y(i) = mean(nonzeros(Ysub(correspondingUsers(i), :)));
        end
        tX = getFeaturesForCluster(artistsInThisCluster, correspondingUsers, userDV, artistDV);

        nnz(y)
        
        % TODO: choose lambda by cross-validation
        % TODO: do better than a simple ridge regression
        lambda = 0.1;
        betas{k} = ridgeRegression(y, tX, lambda);
        
        fprintf('Cluster %d has %d artists\n', k, length(artistsInThisCluster));
    end;


    % predictor = learnAveragePerArtistPredictor(Y, Ytest, userDV, artistDV);
    % predictor = @(user, artist) predictTailFromCluster(user, artist, Y, userDV, artistDV, clusters, betas);
    predictor = @(user, artist) ...
        getFeaturesForCluster(find(clusters == clusters(artist)), user, userDV, artistDV) ...
        * betas{clusters(artist)};

    fprintf('Learnt the tail predictor successfully!\n');
end

function tX = getFeaturesForCluster(artists, users, userDV, artistDV)
    m = length(users);
    dUser = size(userDV, 2);
    dArtist = size(artistDV, 2) * length(artists);
    dSocial = 0;
    tX = zeros(m, 1 + dUser + dArtist + dSocial);

    % TODO: use more data than this?
    tX(:, 1) = 1;
    % Features corresponding to each user having listened to these artists
    tX(:, 2:(dUser+1)) = userDV(users, :);
    % Features corresponding to the artists are repeated on each line
    relevantArtistsDV = artistDV(artists, :);
    tX(:, (dUser+2):(dUser+dArtist+1)) = repmat(relevantArtistsDV(:)', [m 1]);
end
