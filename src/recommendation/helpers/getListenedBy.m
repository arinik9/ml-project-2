function listenedBy = getListenedBy(Y)
% GETLISTENEDBY
%
% INPUT
%   Y:          (users x artists)
% OUTPUT
%   listenedBy: Cell array (one entry per user) giving a list of the
%               of the artists this user listened to at least once

    u = size(Y, 1);
    listenedBy = cell(u, 1);
    for i = 1:u
        listenedBy{i} = find(Y(i, :));
    end;
end
