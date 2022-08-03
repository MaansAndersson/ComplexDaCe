clear all

% fileID = fopen('log1')
% formatSpec = '%f'
% A = fscanf(fileID,formatSpec)


M = []
n = []
STD=[]
for i = 1:10

N=2^i;
filename = ['log',num2str(N)];
m = (dlmread( filename , ',' ));
M = [M; mean(m)];
STD = [STD; std(m)]
n = [n; N];

end 

A = {[0, 0.4470, 0.7410],... 	          	
[0.8500, 0.3250, 0.0980],...	          	
[0.9290, 0.6940, 0.1250],...	          	
[0.4940, 0.1840, 0.5560],...	          
[0.4660, 0.6740, 0.1880],...	          
[0.3010, 0.7450, 0.9330],...	          	
[0.6350, 0.0780, 0.1840]};

% for i = 1:6
% plot(n,M(:,i),'*-')
% end 
%hold on 
p = errorbar(n.*ones(size(M)),M,STD);
for i = 1:6
p(i).LineWidth=3;
p(i).MarkerEdgeColor=A{i};
end
legend('complex64', ...
        'complex128',...
        'float32',...
        'float64',...
        '2 \times float32', ...
        '2 \times float64','location','northwest')

xlabel('# of Elements')
ylabel('Time to write [Ns]')
