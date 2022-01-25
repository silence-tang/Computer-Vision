function match=featureMatch(des1,des2,Num,Thresh)
X=sum(des1.^2,2);
Y=sum(des2.^2,2);
XY=des1*des2';
% zoo_BidirectionalMatch
corr=XY./sqrt(X*Y');
 
[corr1,ix1]=sort(corr,2,'descend');
corr1=corr1(:,1:Num);
ix1=ix1(:,1:Num);
[row1,col1]=find(corr1>Thresh);
match12=zeros(length(row1),2);
match12(:,1)=row1;
match12(:,2)=ix1(size(corr1,1)*(col1-1)+row1);
clear corr1 ix1 row1 col1
 
[corr2,ix2]=sort(corr,1,'descend');
corr2=corr2(1:Num,:);
ix2=ix2(1:Num,:);
[row2,col2]=find(corr2>Thresh);
match21=zeros(length(col2),2);
match21(:,1)=ix2(Num*(col2-1)+row2);
match21(:,2)=col2;
clear corr2 ix2 row2 col2
 
m1=match12(:,1)*10000+match12(:,2);
m2=match21(:,1)*10000+match21(:,2);
 
clear match12
 
match=[];
for k=1:length(m1)
    re=m1(k)-m2;
    idx=find(re==0);
    if ~isempty(idx)
        match =[match;match21(idx,:)];
    end
end
