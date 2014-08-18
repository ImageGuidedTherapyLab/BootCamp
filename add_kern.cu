__global__ 
void add(double* out,const double* a,const double* b){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    out[idx]=a[idx]+b[idx];
}