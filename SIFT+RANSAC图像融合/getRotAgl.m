function agl=getRotAgl(loc1,loc2)
ori1=loc1(:,4);
ori2=loc2(:,4);
agl=ori2-ori1;
agl=agl*180/pi;