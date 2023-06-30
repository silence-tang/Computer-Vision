clear;
close all;
img1 = imread('test3.png');
img2 = imread('test4.png');

% ����SIFT�㷨�ҵ�ƥ���������Բ����ӻ�
[des1, loc1, des2, loc2] = drawKeypoints(img1, img2);

% ������ϴ�ɸѡ����������ƥ���Բ����ӻ�
[matchLoc1, matchLoc2] = siftMatch(des1, loc1, des2, loc2);
% ������е��������
disp([matchLoc1, matchLoc2]);

% ������SIFT�����õ��Ĵ�����ƥ�������
drawLine1(img1, img2, matchLoc1, matchLoc2);

% ����RANSAC�㷨�������Ӧ����H�Լ����յ��������±�corrPtIdx
[H, corrPtIdx] = CalcH(matchLoc2' ,matchLoc1');
% ��ʾ��Ӧ����
fprintf('��Ӧ����H���£�\n');
disp(H);
% ��ʾ��RANSAC�ҵ�������ƥ������matchLoc������±�
fprintf('��RANSAC�ҵ�������ƥ������matchLoc������±����£�\n');
disp(corrPtIdx);
% ������RANSAC���յõ��ľ�����ƥ�������
drawLine2(img1, img2, matchLoc1, matchLoc2, corrPtIdx)

% ���õ�Ӧ����img2ͶӰ��ʹimg2��img1��ͬһ����ϵ�£����������ƴ��
tform = projective2d(H');
img2_adjusted = imwarp(img2, tform); 
% ��ʾһ������ԭʼͼ��
figure;
imshow(img1);
figure;
imshow(img2_adjusted);

% ����ͼ�������ƴ�����ں�
final_img = mosaicFusion(img1, img2, img2_adjusted, H);

% ��ʾ���յõ���ͼ��
figure;
imshow(final_img);
