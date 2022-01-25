clear;
close all;
% 先读取相机拍摄得到的图片
img1 = imread('C:\Users\HP\Desktop\CV第四次上机\left.png');
img2 = imread('C:\Users\HP\Desktop\CV第四次上机\right.png');
% 求图像的宽和高
[h,w,~] = size(img1);
% 将两张图片并排显示一下
figure;
imshowpair(img1, img2, 'montage');
title('原始图像');
% 调用函数求视差图 输入：左右图像的灰度图，输出：两幅图像的视差图
% 先求灰度图
img1_gray = rgb2gray(img1);
img2_gray = rgb2gray(img2);
% 并排显示两幅图的灰度图
figure;
imshowpair(img1_gray, img2_gray, 'montage');
title('原始图像灰度图');
% 求视差图
disp_img = disparity(img1_gray, img2_gray, 'Method', 'BlockMatching',...
    'BlockSize', 255, 'DisparityRange', [0,80], 'ContrastThreshold', 1,...
    'UniquenessThreshold', 55, 'DistanceThreshold',400);
% 单独显示视差图
figure;
imshow(disp_img, [0, 80]);
title('视差图');
colormap jet
colorbar
% 求深度图
% 视差的单位是像素（pixel），深度的单位是毫米（mm）
% depth = ( f * baseline ) / disp
% depth表示深度图，f表示归一化的焦距，也就是内参中的fx
% baseline是基线距离,单位是毫米（mm），disp是视差值
b = 500;                    % 定义基线距离
f_x = 7 / (6.4 * 1e-3);     % 定义焦距
% 初始化深度图矩阵
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
% 显示深度图
depth_img = uint16(depth_img);
figure;
imshow(depth_img);
title('深度图')
% 保存图像
imwrite(depth_img,'深度图.png','png','bitdepth',16);