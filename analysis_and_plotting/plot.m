
clear all;
clc; 


weak_scaling(4);
strong_scaling('512');


function strong_scaling(size)
A = []; Astd = []; C = []; Cstd=[]; E = []; Estd = [];
B = []; Bstd = []; D = []; Dstd=[]; F = []; Fstd = []; 
P = []; 
%size = '512';
precision = 'complex128';
for i = [1, 2, 4, 8, 16]
    codelet_type = '_'; %'_Python';
    func = 'r2r_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,size,num2str(i));
    A = [A, meanm];
    Astd = [Astd, stdm];
    
    codelet_type = '_aopt'; %'_Python';
    func = 'r2r_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,size,num2str(i));
    B = [B, meanm];
    Bstd = [Bstd, stdm];
    
    codelet_type = '_'; %'_Python';
    func = 'r2r_N2_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,size,num2str(i));
    C = [C, meanm];
    Cstd = [Cstd, stdm];
    
    codelet_type = '_aopt'; %'_Python';
    func = 'r2r_N2_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,size,num2str(i));
    D = [D, meanm];
    Dstd = [Dstd, stdm];
    
    codelet_type = '_CPP'; %'_Python';
    func = 'c' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,size,num2str(i));
    E = [E, meanm];
    Estd = [Estd, stdm];
    
    codelet_type = '_Python';
    func = 'c' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,size,num2str(i));
    F = [F, meanm];
    Fstd = [Fstd, stdm];
    
    
    P = [P, i];
end 
figure()
% p=errorbar(N,A,Astd,N,B,Bstd,N,C,Cstd,N,D,Dstd,N,E,Estd)
pl = [A',B',C',D',E',F'];
plstd = [Astd',Bstd',Cstd',Dstd',Estd',Fstd'];


p=errorbar([P',P',P',P',P', P'],pl,plstd);
title(['Strong scaling N=',size,' ',precision])

A = {[0, 0.4470, 0.7410],... 	          	
[0.8500, 0.3250, 0.0980],...	          	
[0.9290, 0.6940, 0.1250],...	          	
[0.4940, 0.1840, 0.5560],...	          
[0.4660, 0.6740, 0.1880],...	          
[0.3010, 0.7450, 0.9330],...	          	
[0.6350, 0.0780, 0.1840]};

xlabel('# OpenMP threads') 
ylabel('Time [s]')

for i = 1:5
p(i).LineWidth=3;
p(i).MarkerEdgeColor=A{i};
end
% errorbar(N,F,Fstd)
legend('r2r','r2r\_aopt','r2r\_N2','r2r\_N2\_aopt','native complex (cpp codelet)') 

end 


function weak_scaling(ompt)
A = []; Astd = []; C = []; Cstd=[]; E = []; Estd = [];
B = []; Bstd = []; D = []; Dstd=[]; F = []; Fstd = []; 
N = []; 
precision = 'complex128';
for i = [32, 64, 128, 256, 512]
    
    codelet_type = '_'; %'_Python';
    func = 'r2r_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,num2str(i),ompt);
    A = [A, meanm];
    Astd = [Astd, stdm];
   
    codelet_type = '_aopt'; %'_Python';
    func = 'r2r_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,num2str(i),ompt);
    B = [B, meanm];
    Bstd = [Bstd, stdm];
    
    codelet_type = '_'; %'_Python';
    func = 'r2r_N2_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,num2str(i),ompt);
    C = [C, meanm];
    Cstd = [Cstd, stdm];

    codelet_type = '_aopt'; %'_Python';
    func = 'r2r_N2_' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,num2str(i),ompt);
    D = [D, meanm];
    Dstd = [Dstd, stdm];
    
    codelet_type = '_CPP'; %'_Python';
    func = 'c' ; %'c'
    [meanm, medianm, stdm] = get_data(func, precision,codelet_type,num2str(i),ompt);
    E = [E, meanm];
    Estd = [Estd, stdm];
    
%     size = '128';
%     codelet_type = '_Python';
%     func = 'c' ; %'c'
%     [meanm, medianm, stdm] = get_data(func, precision,codelet_type,num2str(i),'');
%     F = [F, meanm];
%     Fstd = [Fstd, stdm];
%     
    
    N = [N, i];
end
figure()

pl = [A',B',C',D',E'];
plstd = [Astd',Bstd',Cstd',Dstd',Estd'];


p=errorbar([N',N',N',N',N'],pl,plstd);
title(['Weak scaling threads=',num2str(ompt),' ',precision])

A = {[0, 0.4470, 0.7410],... 	          	
[0.8500, 0.3250, 0.0980],...	          	
[0.9290, 0.6940, 0.1250],...	          	
[0.4940, 0.1840, 0.5560],...	          
[0.4660, 0.6740, 0.1880],...	          
[0.3010, 0.7450, 0.9330],...	          	
[0.6350, 0.0780, 0.1840]};

xlabel('Input size') 
ylabel('Time [s]')

for i = 1:5
p(i).LineWidth=3;
p(i).MarkerEdgeColor=A{i};

end
% errorbar(N,F,Fstd)
legend('r2r','r2r\_aopt','r2r\_N2','r2r\_N2\_aopt','native complex (cpp codelet)') %,'c Py')
end
function [meanm, medianm, stdm] = get_data(func,precision,codelet_type,size,ompt)
% size = '128';
% precision = 'complex128';
% codelet_type = '_aopt'; %'_Python';
% func = 'r2r_' ; %'c'

current_function=['batched_DFT',func,precision,codelet_type,'_',size];

disp(current_function)
directory = ['RUN_',num2str(ompt),'/.dacecache/',current_function,'/profiling/'];
Files = dir([directory,'*']);
Files(3).name;

m = (dlmread( [directory,'/',Files(3).name] , ',', 2, 3 ));

meanm = mean(m); medianm = median(m); stdm=std(m); 
end 