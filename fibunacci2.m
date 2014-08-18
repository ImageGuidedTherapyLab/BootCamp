%% Example of the automatic Matlab publish function:
clear all 
close all
%%
% How to find the first 20 Fibunacci numbers:
N=20;
fib=zeros(N,1); % reserve memory

%%
% initialize the first two numbers
fib(1)=0;
fib(2)=1;

%%
% Loop through the rest:
for i=3:N
    fib(i)=fib(i-1)+fib(i-2);
end
%%
% print out the result
fprintf('%i\n',fib)

%%
% You can also include plots like this
semilogy(fib); title('The Fibunacci numbers as log-plot')

