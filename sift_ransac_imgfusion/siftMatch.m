function [matchLoc1, matchLoc2] = siftMatch(des1, loc1, des2, loc2)

% ������������ڵ��ڶ�����ڵ�ʸ���Ƕȱ�С��distRatio������ƥ���
distRatio = 0.8;   

% ����img1�е�ÿ�������������ӣ��ҵ�����img2�е�ƥ���
des2t = des2';
matchTable = zeros(1,size(des1,1));
for i = 1 : size(des1,1)
   dotprods = des1(i,:) * des2t;
   % ȡ�����Ҳ��Խ����������
   [vals,idx] = sort(acos(dotprods));
   % �������ڵĽǶ��Ƿ�С��distRatio���Եڶ����ڵĽǶ�
   if (vals(1) < distRatio * vals(2))
      matchTable(i) = idx(1);
   else
      matchTable(i) = 0;
   end
end

% ��������ƥ���Ե���Ŀ
num = sum(matchTable > 0);
fprintf('���� %d ������ƥ��㣺\n', num);
idx1 = find(matchTable);
idx2 = matchTable(idx1);
x1 = loc1(idx1,2);
x2 = loc2(idx2,2);
y1 = loc1(idx1,1);
y2 = loc2(idx2,1);
matchLoc1 = [x1,y1];
matchLoc2 = [x2,y2];

end