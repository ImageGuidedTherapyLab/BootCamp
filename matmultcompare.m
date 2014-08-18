
clear all 
close all

Num = 1000;
A = rand(Num);
B = rand(Num);
C = zeros(Num);

disp('running loop')
tic
for iii = 1:Num
  for jjj = 1:Num
    for kkk = 1:Num
      C(iii,jjj) = A(iii,kkk) * B(kkk,jjj);
    end
  end
end
toc

disp('running blas 1')
tic
for iii = 1:Num
  for jjj = 1:Num
      C(iii,jjj) = A(iii,:) * B(:,jjj);
  end
end
toc

disp('running blas 2')
tic
for jjj = 1:Num
      C(:,jjj) = A * B(:,jjj);
end
toc

disp('running blas3')
tic
A*B;
toc
