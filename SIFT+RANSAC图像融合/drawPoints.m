function drawPoints(im1,loc1,im2,loc2)
im=appendingImages(im1,im2);
imshow(im);
hold on
set(gcf,'Color','w');
plot(loc1(:,2),loc1(:,1),'r*',loc2(:,2)+size(im1,2),loc2(:,1),'b*');
hold off
