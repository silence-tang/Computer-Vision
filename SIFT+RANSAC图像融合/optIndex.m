function opt=optIndex(agl)
[n,xout]=hist(agl,180);
alpha=0.75;
[~,IX]=find(n>alpha*max(n));
n=n(IX);
xout=xout(IX);
theta=sum(xout.*n)/sum(n);
rg=[theta-1,theta+1];
opt=[];
for k=1:length(agl)
    if agl(k)>=rg(1) && agl(k)<=rg(2)
        opt=[opt,k];
    end
    if length(opt)>=16
        break
    end
end
