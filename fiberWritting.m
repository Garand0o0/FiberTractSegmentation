function []= fiberWritting(fname,vtx,fiberNum,fiber)

fp=fopen(fname,'w');
fprintf(fp,'# vtk DataFile Version 3.0');
fprintf(fp,'\n');
fprintf(fp,'Fiber point_L_C');
fprintf(fp,'\n');
fprintf(fp,'ASCII');
fprintf(fp,'\n');
fprintf(fp,'DATASET POLYDATA');
fprintf(fp,'\n');
fprintf(fp,'%s %d %s','POINTS',size(vtx,2),'float');
fprintf(fp,'\n');

for i=1:size(vtx,2)
    fprintf(fp,'%f %f %f\n',vtx(:,i));
end
tempv=0;
for i=1:fiberNum
    tempv=tempv+length(fiber{i});
end
fprintf(fp,'%s %d %d\n','LINES',fiberNum,fiberNum+tempv);


for i = 1:fiberNum
    vtxNum2=length(fiber{i});
    fprintf(fp,'%d',vtxNum2);
    for j=fiber{i}(1):fiber{i}(vtxNum2)     
        fprintf(fp,' %d',j-1);
    end
    fprintf(fp,'\n');
end
    
fclose(fp);