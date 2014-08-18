% Example of debugger
 
clear all 
close all
% find the first 20 Fibunacci numbers
N=20;
fib=zeros(N,1); % reserve memory
fib(1)=0;
fib(2)=1;
for i=3:N
    fib(i)=fib(i-1)+fib(i-2);
    %lucky_numbers=round(1000*rand(abs(fib(i)-144),1));
    %fprintf('first lucky number:%i\n',lucky_numbers(1))
end
% print the last number computed
isfib=or(mod(sqrt(5*fib(end)^2-4),1)==0,mod(sqrt(5*fib(end)^2+4),1)==0);
fprintf('%i is a Fibunacci number: %i\n',fib(end),isfib)