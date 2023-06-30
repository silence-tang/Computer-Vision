function [f, inlierIdx] = ransac(x, y, ransacCoef, funcFindF, funcDist)

% MINPTNUM是用于找到合适拟合模型的最小点数，对于单应性矩阵而言，MINPTNUM=4。
minPtNum = ransacCoef.minPtNum;
iterNum = ransacCoef.iterNum;
thInlrRatio = ransacCoef.thInlrRatio;
thDist = ransacCoef.thDist;
ptNum = size(x,2);
thInlr = round(thInlrRatio*ptNum);

inlrNum = zeros(1,iterNum);
fLib = cell(1,iterNum);

for p = 1 : iterNum
	% 每轮迭代中用随机点进行计算
	sampleIdx = createRandidx(ptNum,minPtNum);
	f1 = funcFindF(x(:,sampleIdx),y(:,sampleIdx));
	% 统计内点个数，若大于当前值则更新，否则进行下一次迭代
    % 计算距离
	dist = funcDist(f1,x,y);
	inlier1 = find(dist < thDist);
	inlrNum(p) = length(inlier1);
	if length(inlier1) < thInlr
        continue;
	end
	fLib{p} = funcFindF(x(:,inlier1),y(:,inlier1));
end

% 选择具有最多内点的f作为最终返回的H
[~,idx] = max(inlrNum);
f = fLib{idx};
dist = funcDist(f,x,y);
inlierIdx = find(dist < thDist);

end