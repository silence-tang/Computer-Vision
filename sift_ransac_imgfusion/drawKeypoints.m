function [des1, loc1, des2, loc2] = drawKeypoints(img1, img2)

% ��SIFT�㷨����ÿ��ͼ��������������Ӻ�����������
[des1, loc1] = sift(img1);
[des2, loc2] = sift(img2);
[x1,~] = size(loc1(:,1));
[x2,~] = size(loc2(:,1));
fprintf('img1�� %d ���ؼ���\n',x1);
fprintf('img2�� %d ���ؼ���\n',x2);

% ����һ��ͼ��������ԭʼͼ�������һ����ʾ�������������������ƥ����
img3 = appendimages(img1,img2);
figure('Position', [100 100 size(img3,2) size(img3,1)]);
colormap('gray');
imagesc(img3);
hold on;
% ����img2�Ľǵ�ʱ��Ҫ����һ��ƫ��������img1�Ŀ�
disp = size(img1,2);
% ��ʼ����img1�������㣨�ǵ㣩
for i = 1 : size(loc1(:,1))
    % loc�ĵ�һ���ǽǵ������x���ڶ�����y����Matlab��ͼʱĬ�Ϻ�����y��������x��
    plot(loc1(i,2), loc1(i,1),'co');
end
% ��ʼ����img2�������㣨�ǵ㣩
for i = 1 : size(loc2(:,1))
    plot(loc2(i,2)+disp, loc2(i,1),'bo');
end
hold off;

end