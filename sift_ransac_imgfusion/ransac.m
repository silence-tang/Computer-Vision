function [f, inlierIdx] = ransac(x, y, ransacCoef, funcFindF, funcDist)

% MINPTNUM�������ҵ��������ģ�͵���С���������ڵ�Ӧ�Ծ�����ԣ�MINPTNUM=4��
minPtNum = ransacCoef.minPtNum;
iterNum = ransacCoef.iterNum;
thInlrRatio = ransacCoef.thInlrRatio;
thDist = ransacCoef.thDist;
ptNum = size(x,2);
thInlr = round(thInlrRatio*ptNum);

inlrNum = zeros(1,iterNum);
fLib = cell(1,iterNum);

for p = 1 : iterNum
	% ÿ�ֵ��������������м���
	sampleIdx = createRandidx(ptNum,minPtNum);
	f1 = funcFindF(x(:,sampleIdx),y(:,sampleIdx));
	% ͳ���ڵ�����������ڵ�ǰֵ����£����������һ�ε���
    % �������
	dist = funcDist(f1,x,y);
	inlier1 = find(dist < thDist);
	inlrNum(p) = length(inlier1);
	if length(inlier1) < thInlr
        continue;
	end
	fLib{p} = funcFindF(x(:,inlier1),y(:,inlier1));
end

% ѡ���������ڵ��f��Ϊ���շ��ص�H
[~,idx] = max(inlrNum);
f = fLib{idx};
dist = funcDist(f,x,y);
inlierIdx = find(dist < thDist);

end