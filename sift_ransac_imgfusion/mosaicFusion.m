function [result_img] = mosaicFusion(img1, img2, img2_adjusted, H)

% 利用几何变换进行图像的最终拼接与融合
[m1, n1, ~] = size(img1);
[m2, n2, ~] = size(img2);

T = zeros(3,4);
T(:, 1) = H * [1; 1; 1];
T(:, 2) = H * [n2; 1; 1];
T(:, 3) = H * [n2; m2; 1];
T(:, 4) = H * [1; m2; 1];
x2 = T(1,:) ./ T(3,:);
y2 = T(2,:) ./ T(3,:);

up = round(min(y2));
ydisp = 0;
if up <= 0
	ydisp = -up+1;
	up = 1;
end

left = round(min(x2));
xdisp = 0;
if left <= 0
	xdisp = -left+1;
	left = 1;
end

[m3, n3, ~] = size(img2_adjusted);
result_img(up:up+m3-1, left:left+n3-1, :) = img2_adjusted;
result_img(ydisp+1:ydisp+m1, xdisp+1:xdisp+n1, :) = img1;

end