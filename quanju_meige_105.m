% clear;
% clc;
% close all;
% 
% p1 = '3DSliceAtlas/global_txt';
% fileFolder1=fullfile(p1);
% dirOutput1=dir(fullfile(fileFolder1));
% fileNames1={dirOutput1.name};  %105个文件夹
% for c=1:2  %删除隐藏文件
%     fileNames1(1)=[];
% end
% fileNames1
% 
% for z = 12:12
%     z
%     f1 = ['3DSliceAtlas/global_txt/',char(fileNames1(z))]
%     fid = fopen(f1);
%     data=textscan(fid,'%f');
%     fclose(fid);
%     data_train1 = data{1,1};
%     data_train1 = reshape(data_train1,108,(size(data_train1,1))/108);
%     data_train1 = data_train1'; %n*108
%     data_train1 = reshape(data_train1,size(data_train1,1),36,3); %n*36*3
%     %data_train1 = permute(data_train1,[1 3 2]); %n*3*36
%     size(data_train1,1)
%     size(data_train1,2)
%     size(data_train1,3)
% 
%     f2 = ['3DSliceAtlas/local_txt/',char(fileNames1(z))]
%     fid = fopen(f2);
%     data=textscan(fid,'%f');
%     fclose(fid);
%     data_train2 = data{1,1};
%     data_train2 = reshape(data_train2,108,(size(data_train2,1))/108);
%     data_train2 = data_train2'; %n*108
%     data_train2 = reshape(data_train2,size(data_train2,1),36,3); %n*36*3
%     %data_train2 = permute(data_train2,[1 3 2]); %n*3*36
%     size(data_train2,1)
%     size(data_train2,2)
%     size(data_train2,3)
% 
%     data_train = [data_train1,data_train2]; % n*72*3
%     size(data_train,1)
%     size(data_train,2)
%     size(data_train,3)
% 
%     %data_train = permute(data_train,[1 3 2]); %n*36*6
%     data_train = reshape(data_train,size(data_train,1),216); %n*216
%     p3 = ['3DSliceAtlas/global_local_txt/',char(fileNames1(z))];                              
%     if  exist(['3DSliceAtlas/global_local_txt/'])==0 
%        mkdir(['3DSliceAtlas/global_local_txt/']);
%     end
%     dlmwrite(p3, data_train, 'delimiter',' ');   
% end

%全脑纤维，全局+每个
clear;
clc;
close all;

p1 = 'txt/Center_txt';
fileFolder1=fullfile(p1);
dirOutput1=dir(fullfile(fileFolder1));
fileNames1={dirOutput1.name};  
for c=1:2  %删除隐藏文件
    fileNames1(1)=[];
end
fileNames1

for z = 1:size(fileNames1,2)
    z
    f1 = ['txt/Center_txt/',char(fileNames1(z))]
    fid = fopen(f1);
    data=textscan(fid,'%f');
    fclose(fid);
    data_train1 = data{1,1};
    data_train1 = reshape(data_train1,108,(size(data_train1,1))/108);
    data_train1 = data_train1'; %n*108
    data_train1 = reshape(data_train1,size(data_train1,1),36,3); %n*36*3
    %data_train1 = permute(data_train1,[1 3 2]); %n*3*36
    size(data_train1,1)
    size(data_train1,2)
    size(data_train1,3)

    f2 = ['txt/local_txt/',char(fileNames1(z))]
    fid = fopen(f2);
    data=textscan(fid,'%f');
    fclose(fid);
    data_train2 = data{1,1};
    data_train2 = reshape(data_train2,108,(size(data_train2,1))/108);
    data_train2 = data_train2'; %n*108
    data_train2 = reshape(data_train2,size(data_train2,1),36,3); %n*36*3
    %data_train2 = permute(data_train2,[1 3 2]); %n*3*36
    size(data_train2,1)
    size(data_train2,2)
    size(data_train2,3)

    data_train = [data_train1,data_train2]; % n*72*3
    size(data_train,1)
    size(data_train,2)
    size(data_train,3)

    %data_train = permute(data_train,[1 3 2]); %n*36*6
    data_train = reshape(data_train,size(data_train,1),216); %n*216
    p3 = ['txt/Center_local_txt/',char(fileNames1(z))];                              
    if  exist(['txt/Center_local_txt/'])==0 
       mkdir(['txt/Center_local_txt/']);
    end
    dlmwrite(p3, data_train, 'delimiter',' ');   
end