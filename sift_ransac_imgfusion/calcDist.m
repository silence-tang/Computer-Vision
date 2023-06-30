function d = calcDist(H, pts1, pts2)

% 用单应矩阵将pst1投影到pst3
% 再计算pst2和pst3之间的距离
n = size(pts1,2);
pts3 = H * [pts1;ones(1,n)];
pts3 = pts3(1:2,:) ./ repmat(pts3(3,:),2,1);
d = sum((pts2-pts3).^2,1);

end