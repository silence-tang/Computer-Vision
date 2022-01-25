function gray=img2gray(im)
if length(size(im))==3
    gray=rgb2gray(im);
else
    gray=im;
end
gray=uint8(medfilt2(double(gray)));