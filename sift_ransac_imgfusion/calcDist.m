function d = calcDist(H, pts1, pts2)

% �õ�Ӧ����pst1ͶӰ��pst3
% �ټ���pst2��pst3֮��ľ���
n = size(pts1,2);
pts3 = H * [pts1;ones(1,n)];
pts3 = pts3(1:2,:) ./ repmat(pts3(3,:),2,1);
d = sum((pts2-pts3).^2,1);

end