function drawLine2(img1, img2, matchLoc1, matchLoc2, corrPtIdx)

% 构建一幅图，将两幅原始图像简单排在一起显示，方便后续画出特征点匹配线
img3 = appendimages(img1,img2);
% 绘制RANSAC得到的精特征点匹配线
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
disp = size(img1,2);
for i = 1: size(matchLoc1,1)
    % 若当前下标i在corrPtIdx中，代表当前特征匹配点对是RANSAC找到的精匹配点对
    if ismember(i,corrPtIdx)
        % 遍历整个匹配点对矩阵，对每对精匹配点两点构造一条直线
        line([matchLoc1(i,1) matchLoc2(i,1)+disp],[matchLoc1(i,2) matchLoc2(i,2)], 'Color', 'c');
        plot(matchLoc1(i,1),matchLoc1(i,2),'g*');
        plot(matchLoc2(i,1)+disp,matchLoc2(i,2),'g*');
    end
end
hold off;

end