tic
N=512;
re=linspace(-2,1,(N+1)); re=re(1:end-1);
im=linspace(-1.5,1.5,(N+1)); im=im(1:end-1);
[R,I]=meshgrid(re,im);
C=complex(R,I);
Z=zeros(N);
for iter=1:1000
    Z=Z.*Z+C;
end
toc
imagesc(abs(Z)<2)