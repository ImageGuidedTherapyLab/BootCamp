c=.1;
z=0;
for iter=1:20
    z=z^2+c;
    fprintf('%3.2f, ',z)
end
fprintf('\n')
