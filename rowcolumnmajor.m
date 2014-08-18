clear all 
close all

A = [1 2 3 ; 4 5 6 ];
A(:)
B = [1 4; 2 5 ; 3 6 ];
B(:)

Num = 10000;
A = rand(Num);
B = rand(Num);
C = zeros(Num);

disp('column access')
tic
for iii = 1:Num
  for jjj = 1:Num
      C(iii,jjj) = A(iii,jjj) * B(iii,jjj);
  end
end
toc

disp('row access')
tic
  for jjj = 1:Num
for iii = 1:Num
      C(iii,jjj) = A(iii,jjj) * B(iii,jjj);
  end
end
toc
