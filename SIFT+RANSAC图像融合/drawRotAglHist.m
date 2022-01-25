function drawRotAglHist(agl)
agl=agl(agl>-180);
agl=agl(agl<180);
hist(agl,180);
hold on
set(gcf,'Color','w');
xlabel('Rotated Angle(??)');
ylabel('Number of Feature Point');
hold off
