N=1024;
r1=[-2 1 -1.5 1.5];
r1=[ -0.148913039599577  -0.148877945902814 ...
    -1.025922094089368  -1.025894125339368];

kern = parallel.gpu.CUDAKernel('Mandelbrot_kern.ptx', ...
    'Mandelbrot_kern.cu');
re=linspace(r1(1),r1(2),(N+1)); re=re(1:end-1);
im=linspace(r1(3),r1(4),(N+1)); im=im(1:end-1);
[R,I]=meshgrid(re,im);
Count=gpuarray(single(zeros([N*N 1])));
R=gpuarray((R(:)));
I=gpuarray((I(:)));

kern.ThreadBlockSize=[N 1];
kern.GridSize=[N 1];

tic
Count=feval(kern,Count,R,I);
toc
imagesc(re,im,log10(reshape(gather(Count),[N N])));colormap(gray)
