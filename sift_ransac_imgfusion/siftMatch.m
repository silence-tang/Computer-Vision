function [matchLoc1, matchLoc2] = siftMatch(des1, loc1, des2, loc2)

% 仅保留从最近邻到第二最近邻的矢量角度比小于distRatio的特征匹配点
distRatio = 0.8;   

% 对于img1中的每个特征点描述子，找到它在img2中的匹配点
des2t = des2';
matchTable = zeros(1,size(des1,1));
for i = 1 : size(des1,1)
   dotprods = des1(i,:) * des2t;
   % 取反余弦并对结果进行排序
   [vals,idx] = sort(acos(dotprods));
   % 检查最近邻的角度是否小于distRatio乘以第二近邻的角度
   if (vals(1) < distRatio * vals(2))
      matchTable(i) = idx(1);
   else
      matchTable(i) = 0;
   end
end

% 计算特征匹配点对的数目
num = sum(matchTable > 0);
fprintf('共有 %d 对特征匹配点：\n', num);
idx1 = find(matchTable);
idx2 = matchTable(idx1);
x1 = loc1(idx1,2);
x2 = loc2(idx2,2);
y1 = loc1(idx1,1);
y2 = loc2(idx2,1);
matchLoc1 = [x1,y1];
matchLoc2 = [x2,y2];

end