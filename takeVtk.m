function info = takeVtk(fname)
    % tractseg�����������ֵ�?0.2�ó���Ԥ���ǩ����ȫ����ά����ȡ��ǩ��?72�ģ���73�ࣩ�����vtk
    whole=fname;
    [vtx_whole,fiberNum_whole,fiber_whole] = fiberReading(whole);  % ��ȡȫ��vtk

    f = 'predict_label/yuce_label.txt';
    fid = fopen(f);
    data=textscan(fid,'%d');
    fclose(fid);
    id = reshape(data{1,1}',10,fiberNum_whole)';

    predict = cell(1,104);
    for fiber_th=1:104
        for hang=1:size(id, 1)
            if ismember(fiber_th-1,id(hang,:))~=0
                predict{fiber_th}=[predict{fiber_th},hang];
            end
        end
    end

    index_ls={'AF_left','AF_right','ATR_left','ATR_right','CA','CC_1','CC_2'...
    ,'CC_3','CC_4','CC_5','CC_6','CC_7','CG_left','CG_right','CST_left',...
        'CST_right','FPT_left','FPT_right','FX_left','FX_right','ICP_left',...
        'ICP_right','IFO_left','IFO_right','ILF_left','ILF_right','MCP',...
        'MLF_left','MLF_right','OR_left','OR_right','POPT_left','POPT_right',...
        'SCP_left','SCP_right','SLF_III_left','SLF_III_right','SLF_II_left',...
        'SLF_II_right','SLF_I_left','SLF_I_right','STR_left','STR_right',...
        'ST_FO_left','ST_FO_right','ST_OCC_left','ST_OCC_right','ST_PAR_left',...
        'ST_PAR_right','ST_POSTC_left','ST_POSTC_right','ST_PREC_left',...
        'ST_PREC_right','ST_PREF_left','ST_PREF_right','ST_PREM_left',...
        'ST_PREM_right','T_OCC_left','T_OCC_right','T_PAR_left','T_PAR_right',...
        'T_POSTC_left','T_POSTC_right','T_PREC_left','T_PREC_right',...
        'T_PREF_left','T_PREF_right','T_PREM_left','T_PREM_right','UF_left',...
        'UF_right','CC','CPC','CR-F_left','CR-F_right',...
        'CR-P_left','CR-P_right','EC_left','EC_right',...
        'EmC_left','EmC_right','Intra-CBLM-IP_left',...
        'Intra-CBLM-IP_right','Intra-CBLM-PaT_left',...
        'Intra-CBLM-PaT_right','PLTC_left','PLTC_right',...
        'Sup-FP_left','Sup-FP_right','Sup-F_left',...
        'Sup-F_right','Sup-OT_left','Sup-OT_right',...
        'Sup-O_left','Sup-O_right','Sup-PO_left','Sup-PO_right',...
        'Sup-PT_left','Sup-PT_right','Sup-P_left','Sup-P_right',...
        'Sup_T_left','Sup_T_right', 'others'};

    for th=1:104
        if length(predict{th}) == 0
            fiber_73={};
            vtx_73=[];
            fiberNum_73=0;
        else
            fiber_73=fiber_whole(predict{th}); %��ȡ�ظ�fiber
            fiber_73_mat = cell2mat(fiber_73);
            vtx_73 = vtx_whole(:,fiber_73_mat); %��ȡ�ظ�vtx
            fiberNum_73=length(fiber_73);

            %���ظ������ߵ���Ϣ�͵���Ϣƥ��
            cha = fiber_73{1}(1);
            for w = 1:size(fiber_73{1},2)
                fiber_73{1}(w) = fiber_73{1}(w) - cha + 1;
            end
            for zz=2:length(fiber_73)
                cha = fiber_73{zz}(1) - fiber_73{zz-1}(size(fiber_73{zz-1},2));
                for w = 1:size(fiber_73{zz},2)        
                    fiber_73{zz}(w) = fiber_73{zz}(w) - cha + 1;
                end
            end 
        end
        %�洢�ظ����ֵ�vtk
        savedName=['data/result/',char(index_ls(th)),'.vtk'];
        if  exist(['data/result//'])==0 
            mkdir(['data/result//']);
        end
        fiberWritting(savedName,vtx_73,fiberNum_73,fiber_73);
    end
    info = 'over';
end
