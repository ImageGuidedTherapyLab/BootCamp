__global__ 
void Mandelbrot(float* out,const double* re,const double* im){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    double zr=0;
    double zi=0;
    double cr=re[idx];
    double ci=im[idx];
    double count=0;
    for(int i=1;i<1000;i++){
        double zrt=zr*zr-zi*zi;
        double zit=2*zr*zi;
        zr=zrt+cr;
        zi=zit+ci;
        if(zr*zr+zi*zi<4)count++;
    }
    out[idx]=count;
}