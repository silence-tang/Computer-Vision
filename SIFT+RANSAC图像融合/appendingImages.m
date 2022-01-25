function im=appendingImages(im1,im2)
if length(size(im1))==3
    [row1,col1,~]=size(im1);
    [row2,col2,~]=size(im2);
    if row1<=row2
        im1=[im1;zeros(row2-row1,col1,3)];
    else
        im2=[im2;zeros(row1-row2,col2,3)];
    end
else
    [row1,col1]=size(im1);
    [row2,col2]=size(im2);
    if row1<=row2
        im1=[im1;zeros(row2-row1,col1)];
    else
        im2=[im2;zeros(row1-row2,col2)];
    end
end
im=[im1,im2];
