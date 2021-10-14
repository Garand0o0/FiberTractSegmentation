function [vtx,fiberNum,fiber] = fiberreading(fname)
fid = fopen(fname,'rt');
s = fscanf(fid,'%s',1);
while(~strcmp(s,'POINTS'))
   s = fscanf(fid,'%s',1);  
end
vtxNum = fscanf(fid,'%d',1);
s = fscanf(fid,'%s',1);
vtx = fscanf(fid,'%f',[3,vtxNum]);
s = fscanf(fid,'%s',1);
fiberNum = fscanf(fid,'%d',1);
s = fscanf(fid,'%d',1);
for i = 1:fiberNum
    s = fscanf(fid,'%d',1);
    tem = fscanf(fid,'%d',s);
    tem = (tem+1)';
%     fiber{i} = tem+1;
    fiber{i} = tem;%for matlab index
end
fclose(fid);