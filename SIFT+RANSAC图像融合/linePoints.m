function linePoints(im1,loc1,im2,loc2)
im=appendingImages(im1,im2);
imshow(im);
hold on
set(gcf,'Color','w');
plot(loc1(:,2),loc1(:,1),'r*',loc2(:,2)+size(im1,2),loc2(:,1),'b*');
for k=1:size(loc1,1)
	text(loc1(k,2)-10,loc1(k,1),num2str(k),'Color','y','FontSize',12);
	text(loc2(k,2)+size(im1,2)+5,loc2(k,1),num2str(k),'Color','y','FontSize',12);
    line([loc1(k,2) loc2(k,2)+size(im1,2)],[loc1(k,1) loc2(k,1)],'Color','g');
end
hold off
