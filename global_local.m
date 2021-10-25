function info = global_local()
    p1 = 'data/global';
    fileFolder1=fullfile(p1);
    dirOutput1=dir(fullfile(fileFolder1));
    fileNames1={dirOutput1.name};  
    for c=1:2  
        fileNames1(1)=[];
    end

    for z = 1:length(fileNames1)
        f1 = ['data/global/',char(fileNames1(z))];
        fid = fopen(f1);
        data=textscan(fid,'%f');
        fclose(fid);
        data_train1 = data{1,1};
        data_train1 = reshape(data_train1,108,(size(data_train1,1))/108);
        data_train1 = data_train1'; %n*108
        data_train1 = reshape(data_train1,size(data_train1,1),36,3); %n*36*3

        f2 = ['data/local/',char(fileNames1(z))];
        fid = fopen(f2);
        data=textscan(fid,'%f');
        fclose(fid);
        data_train2 = data{1,1};
        data_train2 = reshape(data_train2,108,(size(data_train2,1))/108);
        data_train2 = data_train2'; %n*108
        data_train2 = reshape(data_train2,size(data_train2,1),36,3); %n*36*3

        data_train = [data_train1,data_train2]; % n*72*3
        data_train = reshape(data_train,size(data_train,1),216); %n*216
        p3 = ['data/global_local/',char(fileNames1(z))];                              
        if  exist(['data/global_local/'])==0 
           mkdir(['data/global_local/']);
        end
        dlmwrite(p3, data_train, 'delimiter',' ');   
    end
    info = 'FiberTractSegmentation......';
end
