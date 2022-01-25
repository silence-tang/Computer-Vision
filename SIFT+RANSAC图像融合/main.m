close all;
clear;
clc;

im1=imread('test3.png');
im2=imread('test4.png');
 
gray1=img2gray(im1);
gray2=img2gray(im2);
 
[des1,loc1]=sift(gray1);
[des2,loc2]=sift(gray2);
 
figure;
drawPoints(im1,loc1,im2,loc2);
 
Num=3;
Thresh=0.85;

match=featureMatch(des1,des2,Num,Thresh);

loc1=loc1(match(:,1),:);
loc2=loc2(match(:,2),:);
 
figure;
linePoints(im1,loc1,im2,loc2);
 
agl=getRotAgl(loc1,loc2);
 
figure;
drawRotAglHist(agl);

opt=optIndex(agl);
loc1=loc1(opt,:);
loc2=loc2(opt,:);
 
figure;
linePoints(im1,loc1,im2,loc2);
 
T=getTransMat(gray1,loc1,gray2,loc2);
im=imRegist(im1,im2,T);
 
figure;
imshow(im);
