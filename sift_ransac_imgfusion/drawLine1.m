function drawLine1(img1, img2, matchLoc1, matchLoc2)

% 构建一幅图，将两幅原始图像简单排在一起显示，方便后续画出特征点匹配线
img3 = appendimages(img1,img2);
% 绘制符合粗筛选条件的特征点匹配线
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
disp = size(img1,2);
for i = 1 : size(matchLoc1,1)
    % 遍历整个匹配点对矩阵，对每对粗特征匹配点两点构造一条直线
    line([matchLoc1(i,1) matchLoc2(i,1)+disp],[matchLoc1(i,2) matchLoc2(i,2)], 'Color', [rand(),rand(),1.0]);
    plot(matchLoc1(i,1),matchLoc1(i,2),'g*');
    plot(matchLoc2(i,1)+disp,matchLoc2(i,2),'g*');
end
hold off;

end