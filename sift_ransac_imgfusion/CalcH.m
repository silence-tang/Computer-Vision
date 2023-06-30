function [H, corrPtIdx] = CalcH(pts1, pts2)

coef.minPtNum = 4;
coef.iterNum = 50;
coef.thDist = 4;
coef.thInlrRatio = 0.1;
% 使用RANSAC算法求解最优的H
[H, corrPtIdx] = ransac(pts1, pts2, coef, @CalcHDetail, @calcDist);

end