__global__ void copy_kernel(double *save, double *y) {
	const int threadID = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	save[threadID] = y[threadID];
	save[threadID + 1] = y[threadID + 1];
}

void cudaAcc_copy(int n, int threads, double *save, double *y) {
	copy_kernel<<<n / (2*threads), threads>>>(save, y);
}
