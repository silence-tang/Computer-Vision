function drawLine1(img1, img2, matchLoc1, matchLoc2)

% ����һ��ͼ��������ԭʼͼ�������һ����ʾ�������������������ƥ����
img3 = appendimages(img1,img2);
% ���Ʒ��ϴ�ɸѡ������������ƥ����
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
disp = size(img1,2);
for i = 1 : size(matchLoc1,1)
    % ��������ƥ���Ծ��󣬶�ÿ�Դ�����ƥ������㹹��һ��ֱ��
    line([matchLoc1(i,1) matchLoc2(i,1)+disp],[matchLoc1(i,2) matchLoc2(i,2)], 'Color', [rand(),rand(),1.0]);
    plot(matchLoc1(i,1),matchLoc1(i,2),'g*');
    plot(matchLoc2(i,1)+disp,matchLoc2(i,2),'g*');
end
hold off;

end