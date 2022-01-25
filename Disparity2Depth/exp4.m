clear;
close all;
% �ȶ�ȡ�������õ���ͼƬ
img1 = imread('C:\Users\HP\Desktop\CV���Ĵ��ϻ�\left.png');
img2 = imread('C:\Users\HP\Desktop\CV���Ĵ��ϻ�\right.png');
% ��ͼ��Ŀ�͸�
[h,w,~] = size(img1);
% ������ͼƬ������ʾһ��
figure;
imshowpair(img1, img2, 'montage');
title('ԭʼͼ��');
% ���ú������Ӳ�ͼ ���룺����ͼ��ĻҶ�ͼ�����������ͼ����Ӳ�ͼ
% ����Ҷ�ͼ
img1_gray = rgb2gray(img1);
img2_gray = rgb2gray(img2);
% ������ʾ����ͼ�ĻҶ�ͼ
figure;
imshowpair(img1_gray, img2_gray, 'montage');
title('ԭʼͼ��Ҷ�ͼ');
% ���Ӳ�ͼ
disp_img = disparity(img1_gray, img2_gray, 'Method', 'BlockMatching',...
    'BlockSize', 255, 'DisparityRange', [0,80], 'ContrastThreshold', 1,...
    'UniquenessThreshold', 55, 'DistanceThreshold',400);
% ������ʾ�Ӳ�ͼ
figure;
imshow(disp_img, [0, 80]);
title('�Ӳ�ͼ');
colormap jet
colorbar
% �����ͼ
% �Ӳ�ĵ�λ�����أ�pixel������ȵĵ�λ�Ǻ��ף�mm��
% depth = ( f * baseline ) / disp
% depth��ʾ���ͼ��f��ʾ��һ���Ľ��࣬Ҳ�����ڲ��е�fx
% baseline�ǻ��߾���,��λ�Ǻ��ף�mm����disp���Ӳ�ֵ
b = 500;                    % ������߾���
f_x = 7 / (6.4 * 1e-3);     % ���役��
% ��ʼ�����ͼ����
depth_img = zeros(h,w);
for i = 1 : h
    for j = 1 : w
        if disp_img(i,j) == 0
            continue;
        else
            depth_img(i,j) = 65535 - uint16( ( f_x * b ) / disp_img(i,j) ) ;
        end
    end
end
% ��ʾ���ͼ
depth_img = uint16(depth_img);
figure;
imshow(depth_img);
title('���ͼ')
% ����ͼ��
imwrite(depth_img,'���ͼ.png','png','bitdepth',16);