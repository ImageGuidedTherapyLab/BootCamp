tic
N=128;
re=linspace(-2,1,(N+1)); re=re(1:end-1);
im=linspace(-1.5,1.5,(N+1)); im=im(1:end-1);
Z=zeros(N);
for ii=1:N
    for jj=1:N
        c=complex(re(ii),im(jj));
        for iter=1:100
            Z(jj,ii)=Z(jj,ii)*Z(jj,ii)+c;
        end
    end
end
toc
imagesc(abs(Z)<2)
        