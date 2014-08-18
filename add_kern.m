N=1024;

A=rand(N,N);
B=rand(N,N);

AD=gpuarray(A(:));
BD=gpuarray(B(:));

kern = parallel.gpu.CUDAKernel('add_kern.ptx', ...
    'add_kern.cu');

kern.ThreadBlockSize=[N 1];
kern.GridSize=[N 1];

CD=gpuarray(zeros(N*N,1));
CD=feval(kern,CD,AD,BD);
C1=reshape(gather(CD),[N N]);
C2=A+B;

norm(C2-C1,'fro')

