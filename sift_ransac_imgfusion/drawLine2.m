function drawLine2(img1, img2, matchLoc1, matchLoc2, corrPtIdx)

% ����һ��ͼ��������ԭʼͼ�������һ����ʾ�������������������ƥ����
img3 = appendimages(img1,img2);
% ����RANSAC�õ��ľ�������ƥ����
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
disp = size(img1,2);
for i = 1: size(matchLoc1,1)
    % ����ǰ�±�i��corrPtIdx�У�����ǰ����ƥ������RANSAC�ҵ��ľ�ƥ����
    if ismember(i,corrPtIdx)
        % ��������ƥ���Ծ��󣬶�ÿ�Ծ�ƥ������㹹��һ��ֱ��
        line([matchLoc1(i,1) matchLoc2(i,1)+disp],[matchLoc1(i,2) matchLoc2(i,2)], 'Color', 'c');
        plot(matchLoc1(i,1),matchLoc1(i,2),'g*');
        plot(matchLoc2(i,1)+disp,matchLoc2(i,2),'g*');
    end
end
hold off;

end