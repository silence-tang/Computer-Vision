function [des1, loc1, des2, loc2] = drawKeypoints(img1, img2)

% 用SIFT算法计算每幅图像的特征点描述子和特征点坐标
[des1, loc1] = sift(img1);
[des2, loc2] = sift(img2);
[x1,~] = size(loc1(:,1));
[x2,~] = size(loc2(:,1));
fprintf('img1有 %d 个关键点\n',x1);
fprintf('img2有 %d 个关键点\n',x2);

% 构建一幅图，将两幅原始图像简单排在一起显示，方便后续画出特征点匹配线
img3 = appendimages(img1,img2);
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
% 绘制img2的角点时需要加上一个偏移量（即img1的宽）
disp = size(img1,2);
% 开始绘制img1的特征点（角点）
for i = 1 : size(loc1(:,1))
    % loc的第一列是角点坐标的x，第二列是y，但Matlab绘图时默认横向是y周纵向是x轴
    plot(loc1(i,2), loc1(i,1),'co');
end
% 开始绘制img2的特征点（角点）
for i = 1 : size(loc2(:,1))
    plot(loc2(i,2)+disp, loc2(i,1),'bo');
end
hold off;

end