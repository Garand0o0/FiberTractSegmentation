function info = c_m_c40(fname)
    [vtx,fiberNum,fiber] = fiberReading(fname);
    Centerpoint = mean(vtx'); 


    vtx(1,:)= vtx(1,:)-Centerpoint(1,1);
    vtx(2,:)= vtx(2,:)-Centerpoint(1,2);
    vtx(3,:)= vtx(3,:)-Centerpoint(1,3);

    r = (vtx(1,:).^2 + vtx(2,:).^2 + vtx(3,:).^2).^(1/2); %ÿ���������ĵ�ľ���?
    zangle = acosd(vtx(3,:)./r)+eps;    %�춥�� arccos  ����ԭ�����ߺ� z ��ļн�?
    Azimuth  = atan2d(vtx(2,:),vtx(1,:))+eps;  %��λ��  ���ԭ��������? xy ����ɵ�ƽ���ϵ�ͶӰ��? x ��ļн�? 
    r = mapminmax(r,0,1)+eps;  %��һ��r

    x1=0:1/36:1;  %���ڽ��뾶�ȷֳ�36��
    x2=0:5:180;  %���ڽ�����36�ȷ�
    x3=-180:10:180; %����λ�ǵȷ�36��

    batch = 1:100000:fiberNum;
    batch(1) = batch(1) - 1;
    if batch(end) ~= fiberNum
        batch = [batch,fiberNum];
    end
    for pici=1:length(batch)-1
        j = 1;
        for i=batch(pici)+1:batch(pici+1)
            qvtx(1,:)=r(fiber{i});
            qvtx(2,:)=zangle(fiber{i});
            qvtx(3,:)=Azimuth(fiber{i});

            for i1=1:36     
                 A(i1,1)=size(find(qvtx(1,:)>x1(i1)&qvtx(1,:)<=x1(i1+1)),2);
            end
            for i2=1:36    
                 A(i2,2)=size(find(qvtx(2,:)>x2(i2)&qvtx(2,:)<=x2(i2+1)),2);
            end
            for i3=1:36
                 A(i3,3)=size(find(qvtx(3,:)>x3(i3)&qvtx(3,:)<=x3(i3+1)),2);
            end

           A = single(A);
           coordinate(j,:)=A(:)/size(fiber{j},2);
           j=j+1;
           qvtx=[];
           A=[];
        end
        coordinate = single(coordinate);
        p3 = ['data/global/output_',num2str(pici),'.txt'];                             
        if  exist(['data/global/'])==0 
            mkdir(['data/global/']);
        end
        dlmwrite(p3, coordinate, 'delimiter',' ');      
        coordinate=[];
    end

    [vtx,fiberNum,fiber] = fiberReading(fname);
    x1=0:1/36:1;  %���ڽ��뾶�ȷֳ�36��
    x2=0:5:180;  %���ڽ�����36�ȷ�
    x3=-180:10:180; %����λ�ǵȷ�36��

    for pici=1:length(batch)-1
        j = 1;
        for i=batch(pici)+1:batch(pici+1)
            vtx1 = vtx(:,fiber{i}(1):fiber{i}(size(fiber{i},2)));
            Centerpoint = mean(vtx1'); %���ĵ�

            vtx1(1,:)= vtx1(1,:)-Centerpoint(1,1);
            vtx1(2,:)= vtx1(2,:)-Centerpoint(1,2);
            vtx1(3,:)= vtx1(3,:)-Centerpoint(1,3);

            r = (vtx1(1,:).^2 + vtx1(2,:).^2 + vtx1(3,:).^2).^(1/2); %ÿ���������ĵ�ľ���?
            zangle = acosd(vtx1(3,:)./r)+eps;    %�춥�� arccos  ����ԭ�����ߺ� z ��ļн�?
            Azimuth  = atan2d(vtx1(2,:),vtx1(1,:))+eps;  %��λ��  ���ԭ��������? xy ����ɵ�ƽ���ϵ�ͶӰ��? x ��ļн�? 
            r = mapminmax(r,0,1)+eps;  %��һ��r

            qvtx(1,:)=r;
            qvtx(2,:)=zangle;
            qvtx(3,:)=Azimuth;

            for i1=1:36     
                 A(i1,1)=size(find(qvtx(1,:)>x1(i1)&qvtx(1,:)<=x1(i1+1)),2);
            end
            for i2=1:36    
                 A(i2,2)=size(find(qvtx(2,:)>x2(i2)&qvtx(2,:)<=x2(i2+1)),2);
            end
            for i3=1:36
                 A(i3,3)=size(find(qvtx(3,:)>x3(i3)&qvtx(3,:)<=x3(i3+1)),2);
            end
            A = single(A);
            coordinate(j,:)=A(:)/size(fiber{j},2);
            j=j+1;
            qvtx=[];
            A=[];
        end
        coordinate = single(coordinate);
        p3 = ['data/local/output_',num2str(pici),'.txt'];                             
        if  exist(['data/local/'])==0 
            mkdir(['data/local/']);
        end
        dlmwrite(p3, coordinate, 'delimiter',' ');      
        coordinate=[];
    end
    info = 'processing data 50%';
end