// mexcuda '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64' cmex_HGDALS_GPU.cu -lcudart -lcublas -lcusolver -lcusparse

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include "mex.h"
#include "mat.h"
#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cusolverDn.h>

#define prhs_y prhs[0]
#define prhs_x prhs[1]
#define prhs_B prhs[2]
#define prhs_A prhs[3]
#define prhs_lmbd prhs[4]
#define prhs_l2x prhs[5]
#define prhs_l2a prhs[6]
#define prhs_Niter prhs[7]
#define prhs_tol prhs[8]
#define prhs_loss prhs[9]

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define SIGN(x) (((x) > 0) ? 1 : -1)

#define BLOCK_SIZE 128

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at file %s and line %d:\n",__FILE__,__LINE__);\
	mexErrMsgTxt(cudaGetErrorString((x)));}} while(0)
		
#define CUBLAS_CALL(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
	printf("Error at file %s and line %d:\n",__FILE__,__LINE__);\
	mexErrMsgTxt("CUBLAS ERROR");}} while(0)
		
#define CUSOLVER_CALL(x) do { if((x)!=CUSOLVER_STATUS_SUCCESS) { \
	printf("Error at file %s and line %d:\n",__FILE__,__LINE__);\
	mexErrMsgTxt("CUSOLVER ERROR");}} while(0)
		
#define CUSPARSE_CALL(x) do { if((x)!=CUSPARSE_STATUS_SUCCESS) { \
	printf("CUSPARSE API failed at file %s and line %d:\n",__FILE__,__LINE__);\
	mexErrMsgTxt("CUSOLVER ERROR");}} while(0)

// ->->->->-> Evaluate cost funtion and calculate auxiliar vars "d_xerr" and "d_yerr" <-<-<-<-<- \\

__global__ void multiply_loss(double *d_yerr, double *d_y, double *d_B_xe, double *d_loss, int M, int T, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int MT = M * T;
	
	if (i < MT * E) {
		int t = (i % MT) / M;
		d_yerr[i] = (d_y[i] - d_B_xe[i]) * d_loss[t];
	}
}

__global__ void overwrite_border(double *d_xerr, double *d_x, int T, int N, int P, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int NP = N * P;
	
	if (i < NP * (E - 1)) {
		int e = i / NP;
		int ind = i % NP;
		int p = ind / N;
		ind = e * N * T + (T - P + p) * N + (ind % N);
		d_xerr[ind] = d_x[ind];
	}
}

// Main function which call all the kernel functions defined above
inline double eval_cost(cublasHandle_t handle, double *h_A, double *h_xerr, double *h_yerr, double *h_loss, \
	double *d_xerr, double *d_yerr, double *d_y, double *d_B, double *d_x, double *d_A, double *d_loss, double *d_B_x, \
	double lmbd, double l2x, double l2a, int T, int M, int N, int E, int P) {
	
	int NN = N * N;
	double F = 0.0;
	double tmp, acc;
	const double one = 1.0, zero = 0.0, minus_one = -1.0;
	
	// -> xerr(:) = x(:);
	CUDA_CALL(cudaMemcpy(d_xerr, d_x, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
	
	// -> Update the errors while taking into account the time reversal order
	// xerr(:,1:T-P) = xerr(:,1:T-P) - A(:,:,k)*x(:,(1:T-P)+k);
	// xerr(:,1:T*E-P) = xerr(:,1:T*E-P) - A(:,:,k)*x(:,(1:T*E-P)+k); // modified for epoched data
	for (int k = 0; k < P; k++) {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, T * E - P, N, &minus_one, d_A + NN * k, N, d_x + N * (k + 1), N, &one, d_xerr, N);
		CUDA_CALL(cudaDeviceSynchronize());
	}
	
	// the above "xerr" calculation could have been done withing 2 for-loops, but to minimize the number of kernel launches
	// all the epochs are pooled together, which contaminates the border, and that is rectified below.
	int GRID_SIZE_NPE = (uint32_T)ceil((N * P * (E - 1)) / (BLOCK_SIZE + 0.0f));
	overwrite_border << <GRID_SIZE_NPE, BLOCK_SIZE >> > (d_xerr, d_x, T, N, P, E);

	// -> B * x
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, T * E, N, &one, d_B, M, d_x, N, &zero, d_B_x, M);
	CUDA_CALL(cudaDeviceSynchronize());
	
	// -> yerr = (y - B * x) .* loss
	int GRID_SIZE_MTE = (uint32_T)ceil((M * T * E) / (BLOCK_SIZE + 0.0f));
	multiply_loss << <GRID_SIZE_MTE, BLOCK_SIZE >> > (d_yerr, d_y, d_B_x, d_loss, M, T, E);
	
	// Transfer from device to host the vars needed for cost function calculation
	CUDA_CALL(cudaMemcpy(h_xerr, d_xerr, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_yerr, d_yerr, M * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	
	// Consider the yerr's part
	for (int e = 0; e < E; e++) {
		int eMT = e * M * T;
		for (int t = 0; t < T; t++) {
			if (h_loss[t] != 0) {
				int base = eMT + t * M;
				for (int i = 0; i < M; i++) {
					tmp = h_yerr[base + i];
					F += tmp * tmp;
				}
			}
		}
	}
	
	//printf("%10.6f\n", F);
	
	// Consider the xerr's part
	acc = 0.0;
	for (int e = 0; e < E; e++) {
		int eNT = e * N * T;
		for (int t = 0; t < T; t++) {
			int base = eNT + t * N;
			for (int i = 0; i < N; i++) {
				tmp = h_xerr[base + i];
				acc += tmp * tmp;
			}
		}
	}
	F += lmbd * acc;
	
	//printf("%10.6f\n", acc);
	
	// Consider the l2x's penalty
	if (l2x != 0) {
		CUDA_CALL(cudaMemcpy(h_xerr, d_x, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
		acc = 0.0;
		for (int e = 0; e < E; e++) {
			int eNT = e * N * T;
			for (int t = 0; t < T; t++) {
				int base = eNT + t * N;
				for (int i = 0; i < N; i++) {
					tmp = h_xerr[base + i];
					acc += tmp * tmp;
				}
			}
		}
		F += l2x * acc;
		
		//printf("%10.6f\n", acc);
	}
	
	// Consider the l2a's penalty
	if (l2a != 0) {
		CUDA_CALL(cudaMemcpy(h_A, d_A, NN * P * sizeof(double), cudaMemcpyDeviceToHost));
		acc = 0.0;
		for (int k = 0; k < P; k++) {
			int k_NN = k * NN;
			for (int i = 0; i < NN; i++) {
				tmp = h_A[k_NN + i];
				acc += tmp * tmp;
			}
		}
		F += l2a * acc;
		
		//printf("%10.6f\n", acc);
	}
	
	//F = @(x,A,xerr) (sum(sum(((y - B*x).*loss).^2)) + lmbd*sum(sum(xerr(:,P+1:T).^2)) + lmbd*sum(sum(x(:,1:P).^2)) + l2a*sum(A(:).^2))/T;
	
	F = F / (T * E);
	
	//printf("Line %d:",__LINE__); mexErrMsgTxt("STOP HERE");
	
	return F;
}

// ->->->->-> Calculate x's gradient descent (d_dx: dim is NxT) direction <-<-<-<-<- \\

__global__ void update_dx(double *d_dx, const double *d_xerr, double lmbd, int T, int N, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < T * N * E) {
		d_dx[i] += lmbd * d_xerr[i];
	}
}

// dx(:,(1:T-P)+p,:) = dx(:,(1:T-P)+p,:) + dX0
__global__ void update_dx_II(double *d_dx, const double *d_dX0, int T, int N, int P, int E, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int NT = N * (T - P);
	
	if (i < NT * E) {
		int e = i / NT;
		int ind = i % NT;
		int t = ind / N;
		ind = e * N * T + (t + p) * N + (ind % N);
		d_dx[ind] += d_dX0[i];
	}
}

__global__ void update_dx_III(double *d_dx, const double *d_x, double l2x, int T, int N, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < T * N * E) {
		d_dx[i] += l2x * d_x[i];
	}
}

__global__ void dx_div_TE(double *d_dx, int T, int N, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int TE = T * E;
	
	if (i < N * TE) {
		d_dx[i] /= TE;
	}
}

// X0 = reshape(X(:,1:T-P,:),[N (T-P)*E]); or
// X0 = reshape(xerr(:,1:T-P,:),[N (T-P)*E]);
__global__ void fill_X0(double *d_X0, const double *d_x, int N, int T, int P, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int NT = N * (T - P);
	
	if (i < NT * E) {
		int e = i / NT;
		int ind = i % NT;
		int t = ind / N;
		ind = e * N * T + t * N + (ind % N);
		d_X0[i] = d_x[ind];
	}
}

// Main function which call all the kernel functions defined above
inline void calculate_x_GD(cublasHandle_t handle, double *d_dx, const double *d_B, const double *d_x, double *d_X0, double *d_dX0, \
	const double *d_A, const double *d_xerr, const double *d_yerr, int T, int M, int N, int P, int E, double lmbd, double l2x) {
	
	const double minus_one = -1.0, zero = 0.0, minus_lmbd = -lmbd;
	const int NN = N * N;
	
	// -> Calculate dx
	
	// => dx = -B' * ((y - B * x) .* loss)
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, T * E, M, &minus_one, d_B, M, d_yerr, M, &zero, d_dx, N);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	/*
	double *tmp = (double*)malloc(N * T * E * sizeof(double));
	CUDA_CALL(cudaMemcpy(tmp, d_dx, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	double sum = 0;
	for (int i = 0; i < N * T * E; i++)
		sum += tmp[i];
	printf("sum = %.8f\n", sum);
	*/
	
	// => dx = dx + lmbd * xerr
	int GRID_SIZE_TNE = (uint32_T)ceil((T * N * E) / (BLOCK_SIZE + 0.0f));
	update_dx << <GRID_SIZE_TNE, BLOCK_SIZE >> > (d_dx, d_xerr, lmbd, T, N, E);
	
	/*
	CUDA_CALL(cudaMemcpy(tmp, d_dx, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	sum = 0;
	for (int i = 0; i < N * T * E; i++)
		sum += tmp[i];
	printf("sum = %.8f\n", sum);
	*/
	
	// => dx(:,(1:T-P)+k) = dx(:,(1:T-P)+k) - lmbd * (A(:,:,k)' * xerr(:,1:T-P)); // in time-reversal order
	// Now, extended for multiple epochs:
	// X0 = reshape(xerr(:,1:T-P,:),[N (T-P)*E]);
	// dX0 = -lmbd * (A(:,:,k)' * X0);
	// dx(:,(1:T-P)+k,:) = dx(:,(1:T-P)+k,:) + dX0
	int GRID_SIZE_NTE = (uint32_T)ceil((N * (T - P) * E) / (BLOCK_SIZE + 0.0f));
	fill_X0 << <GRID_SIZE_NTE, BLOCK_SIZE >> > (d_X0, d_xerr, N, T, P, E);
	
	/*
	CUDA_CALL(cudaMemcpy(tmp, d_xerr, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	sum = 0;
	for (int i = 0; i < N * T * E; i++)
		sum += tmp[i];
	printf("xerr sum = %.8f\n", sum);
	
	CUDA_CALL(cudaMemcpy(tmp, d_X0, N * (T - P) * E * sizeof(double), cudaMemcpyDeviceToHost));
	sum = 0;
	for (int i = 0; i < N * (T - P) * E; i++)
		sum += tmp[i];
	printf("X0 sum = %.8f\n", sum);
	*/
	
	for (int p = 0; p < P; p++) {		
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, (T - P) * E, N, &minus_lmbd, d_A + NN * p, N, d_X0, N, &zero, d_dX0, N);
		CUDA_CALL(cudaDeviceSynchronize());
		
		update_dx_II << <GRID_SIZE_NTE, BLOCK_SIZE >> > (d_dx, d_dX0, T, N, P, E, p + 1);
	}
	
	/*
	CUDA_CALL(cudaMemcpy(tmp, d_dx, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	sum = 0;
	for (int i = 0; i < N * T * E; i++)
		sum += tmp[i];
	printf("sum = %.8f\n", sum);
	*/
	
	// => dx = dx + l2x * xe
	update_dx_III << <GRID_SIZE_TNE, BLOCK_SIZE >> > (d_dx, d_x, l2x, T, N, E);
	
	/*
	CUDA_CALL(cudaMemcpy(tmp, d_dx, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	sum = 0;
	for (int i = 0; i < N * T * E; i++)
		sum += tmp[i];
	printf("sum = %.8f\n", sum);
	mexErrMsgTxt("STOP HERE");
	*/
	
	// => dx = dx / (T * E)
	dx_div_TE << <GRID_SIZE_TNE, BLOCK_SIZE >> > (d_dx, T, N, E);
}

// ->->->->-> Calculate the autoregressive coefficients (d_A: dim is NxNxp) using the classical OLS method <-<-<-<-<- \\

// Z = [X(:,2:T-P+1); X(:,3:T-P+2); ...; X(:,P+1:T)] and X0 = X(:,1:T-P)
// Now, extended for multiple epochs
// Z = [reshape(X(:,2:T-P+1,:),[N (T-P)*E]); reshape(X(:,3:T-P+2,:),[N (T-P)*E]); ...; reshape(X(:,P+1:T,:),[N (T-P)*E])]
// and X0 = reshape(X(:,1:T-P,:),[N (T-P)*E]);
__global__ void fill_Z_block(double *d_Z, const double *d_x, int N, int P, int T, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int NP = N * P;
	int NPT = NP * (T - P);
	
	if (i < NPT * E) {
		int e = i / NPT;
		int ind = i % NPT;
		int t = ind / NP;
		ind = ind % NP;
		int p = ind / N;
		ind = e * N * T + (t + p + 1) * N + (ind % N);
		d_Z[i] = d_x[ind];
	}
}

// The <Type>syrk multiplication in CUBLAS only filled the lower triangular part of the symmetric matrix,
// so here we fill up the upper triangular part.
__global__ void fill_upper_triangular(double *d_AtA, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < N * (N - 1) / 2) {
		int x = 2 * N - 1;
		int icol = (int)floor((x - sqrt(x * x - 8.0 * i)) / 2);
		int func_icol = (icol * (x - icol)) / 2;
		int irow = icol + i - func_icol + 1;
		int il = icol * N + irow; // index for lower triangular element
		int iu = irow * N + icol; // index for upper triangular element
		d_AtA[iu] = d_AtA[il];
	}
}

// add lmbd to the diagonal to an NxN matrix
__global__ void add_diagonal_lmbd(double *d_ZZt, double lmbd, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < N) {
		d_ZZt[i * N + i] += lmbd;
	}
}

// A dimension is NR x NC
__global__ void transpose_matrix(double *d_At, const double *d_A, int NR, int NC) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < NR * NC) {
		int irow = i % NR;
		int icol = i / NR;
		d_At[irow * NC + icol] = d_A[i];
	}
}

// Main function which call all the kernel functions defined above: solve B = YZt/ZZt, where B = [A(1), A(2), ..., A(P)]
inline void calculate_A_OLS(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_Dncusolver, \
	double *d_A, const double *d_x, double *d_X0, double *d_Z, double *d_ZZt, double *d_ZXt, int * d_Ipiv, int *d_info,
	int pivot, int T, int M, int N, int P, int E, double l2a) {
		
	const double one = 1.0;
	const double zero = 0.0;
	int NP = N * P;
	
	// Z = [X(:,2:T-P+1); X(:,3:T-P+2); ...; X(:,P+1:T)]
	// Now, extended for multiple epochs
	// Z = [reshape(X(:,2:T-P+1,:),[N (T-P)*E]); reshape(X(:,3:T-P+2,:),[N (T-P)*E]); ...; reshape(X(:,P+1:T,:),[N (T-P)*E])]
	int GRID_SIZE_NPTE = (uint32_T)ceil((NP * (T - P) * E) / (BLOCK_SIZE + 0.0f));
	fill_Z_block << <GRID_SIZE_NPTE, BLOCK_SIZE >> > (d_Z, d_x, N, P, T, E);
	
	// -> Calculate ZZt
	CUBLAS_CALL(cublasDsyrk(handle_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, NP, (T - P) * E, &one, d_Z, NP, &zero, d_ZZt, NP));
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// fill up upper triangular part of d_ZZt
	int GRID_SIZE_NPtri = (uint32_T)ceil((NP * (NP - 1) / 2) / (BLOCK_SIZE + 0.0));
	fill_upper_triangular << <GRID_SIZE_NPtri, BLOCK_SIZE >> > (d_ZZt, NP);
	
	// add l2a to the diagonal of ZZt
	if (l2a != 0) {
		int GRID_SIZE_NP = (uint32_T)ceil(NP / (BLOCK_SIZE + 0.0));
		add_diagonal_lmbd << <GRID_SIZE_NP, BLOCK_SIZE >> > (d_ZZt, l2a, NP);
	}
	
	// -> Calculate ZXt = Z * X0' where X0 = X(:,1:T-P)
	// Now, extended for multiple epochs X0 = reshape(X(:,1:T-P,:),[N (T-P)*E]);
	int GRID_SIZE_NTE = (uint32_T)ceil((N * (T - P) * E) / (BLOCK_SIZE + 0.0f));
	fill_X0 << <GRID_SIZE_NTE, BLOCK_SIZE >> > (d_X0, d_x, N, T, P, E);
	cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, NP, N, (T - P) * E, &one, d_Z, NP, d_X0, N, &zero, d_ZXt, NP);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// -> Solve B such as ZZt * Bt = ZXt
	int lwork = 0; 			// size of workspace for getrf
    double *d_work = NULL; 	// device workspace for getrf
	
	// => step 1: query working space for getrf
    CUSOLVER_CALL(cusolverDnDgetrf_bufferSize(handle_Dncusolver, NP, NP, d_ZZt, NP, &lwork));
	
	CUDA_CALL(cudaDeviceSynchronize());
	
    CUDA_CALL(cudaMalloc((void**)&d_work, sizeof(double)*lwork));
	
	// => step 2: LU factorization
    if (pivot) {
        CUSOLVER_CALL(cusolverDnDgetrf(handle_Dncusolver, NP, NP, d_ZZt, NP, d_work, d_Ipiv, d_info));
    }
    else {
        CUSOLVER_CALL(cusolverDnDgetrf(handle_Dncusolver, NP, NP, d_ZZt, NP, d_work, NULL, d_info));
    }
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// => Step 3: calculate Bt (it is written as output in the same input right-hand side matrix ZYt)
	if (pivot) {
        CUSOLVER_CALL(cusolverDnDgetrs(handle_Dncusolver, CUBLAS_OP_N, NP, N, d_ZZt, NP, d_Ipiv, d_ZXt, NP, d_info));
    }
    else {
        CUSOLVER_CALL(cusolverDnDgetrs(handle_Dncusolver, CUBLAS_OP_N, NP, N, d_ZZt, NP, NULL, d_ZXt, NP, d_info));
    }
	
	// The cuSolver library functions prefer to keep asynchronous execution as much as possible.
	// Developers can always use the cudaDeviceSynchronize() function to ensure that the
	// execution of a particular cuSolver library routine has completed.
	//
	// It seems that between two CUSOLVER function is OK not to synch, like between the steps 2 and 3
	// above, but to play safe we synch here before calling my own (transpose) kernel
    CUDA_CALL(cudaDeviceSynchronize());
	
	// transpose d_ZXt to get B = [A(1), A(2), ..., A(P)]
	int GRID_SIZE_pNN = (uint32_T)ceil((NP * N) / (BLOCK_SIZE + 0.0));
	transpose_matrix << <GRID_SIZE_pNN, BLOCK_SIZE >> > (d_A, d_ZXt, NP, N);
	
	// -> Free resources
	cudaFree(d_work);	
}

// ->->->->-> Solve the system A*X = B, where A and B are matrices, by LU-based left division (X = A\B) <-<-<-<-<- \\

// Bt = transf(B), where B = [A(1), ..., A(P)] and Bt = [A(1)', ..., A(P)']
// A(k) dimension is N x N, for k = 1, ..., P
__global__ void transpose_A_block(double *d_At_block, const double *d_A, const int N, const int P) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int NN = N * N;
	
	if (i < P * NN) {
		int ind = i % NN;
		int iblock = i / NN;
		int irow = ind % N;
		int icol = ind / N;
		d_At_block[iblock * NN + irow * N + icol] = d_A[i];
	}
}

// d_AtA dimension is Nr x Nr (Nr is a multiple of N). d_AtA_block dimension is N x N.
// k and l indicates that d_AtA_block is the (l,k)-th block in the matrix
__global__ void copy_block(double *d_AtA_block, double *d_AtA, const int Nr, const int N, const int k, const int l) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < N * N) {
		int irow = i % N;
		int icol = i / N;
		d_AtA_block[i] = d_AtA[(k * N + icol) * Nr + l * N + irow];
	}
}

// All the elements of the matrix are initialized to zero
__global__ void reset_to_zero(double *d_bandMat, int numel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < numel) {
		d_bandMat[i] = 0;
	}
}

// The matrix d_bandMat is considered in vectorized format with dimension numel x 1
__global__ void multiply_banded_values_by_lambda(double *d_bandMat, double lmbd, int numel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < numel) {
		d_bandMat[i] *= lmbd;
	}
}

// Fill the identiy in the middle block of the banded matrix, i.e. in rectangular form representation.
// Nr is the number of rows of the banded matrix, where every block dimension is N x N.
// Therefore, Nr is a multiple of N.
// icolblock is the column index of the banded matrix to be filled up.
// We assume that the matrix is initially fill up with zeros, so we just copy ones in the "diagonal".
__global__ void insert_identity(double *d_bandMat, const int Nr, const int N, const int icolblock) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < Nr) {
		int icol = i % N;
		d_bandMat[(icolblock * N + icol) * Nr + i] = 1.0;
	}
}

// Add an scalar value to the main diagonal
__global__ void add_scalar_identity(double *d_bandMat, int Nr, int N, int icolblock, double val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < Nr) {
		int icol = i % N;
		d_bandMat[(icolblock * N + icol) * Nr + i] += val;
	}
}

// d_bandMat is as described above but, instead of adding ones as above, here we add the d_block's elements
// to the specified block.
__global__ void add_block(double *d_bandMat, const double *d_block, int Nr, int N, int icolblock, int irowblock0, int nnzb) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int numel = N * N;
	
	if (i < nnzb * numel) {
		int height = i / numel;
		int ind = i % numel;
		int irow = ind % N;
		int icol = ind / N;
		d_bandMat[(icolblock * N + icol) * Nr + (irowblock0 + height) * N + irow] += d_block[ind];
	}
}

// d_bandMat is as described above but, instead of adding ones as above, here we substract the d_block's elements.
__global__ void insert_minus_block(double *d_bandMat, const double *d_block, int Nr, int N, int icolblock, int irowblock0, int nnzb) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int numel = N * N;
	
	if (i < nnzb * numel) {
		int height = i / numel;
		int ind = i % numel;
		int irow = ind % N;
		int icol = ind / N;
		d_bandMat[(icolblock * N + icol) * Nr + (irowblock0 + height) * N + irow] = -d_block[ind];
	}
}

// Add BtB to the central block (diagonal) according to the indices of nonzero loss entries (indloss of dimension nnz x 1
__global__ void add_block_BtB_diag(double *d_bandMat, const double *d_BtB, const int *d_indloss, int Nr, int N, int icolblock, int nnz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int numel = N * N;
	
	if (i < nnz * numel) {
		int t = d_indloss[i / numel];
		int ind = i % numel;
		int irow = ind % N;
		int icol = ind / N;
		d_bandMat[(icolblock * N + icol) * Nr + t * N + irow] += d_BtB[ind];
	}
}

//  Transforming from banded matrix representation (BMR) to coordinate (COO) sparse format
__global__ void bmr2sparse(double *d_csrVal, const double *d_bandMat, const int nnz, const int *d_bmrInd) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < nnz) {
		d_csrVal[i] = d_bandMat[d_bmrInd[i]];
	}
}

// Main function which call all the kernel functions defined above
inline void calculate_x_OLS(cublasHandle_t handle_cublas, mxArray *rhs[], mxArray *lhs, double *d_x, \
	double *d_bandMat, double *d_csrVal, const int *d_bmrInd, const double *d_BtYvec, const double *d_BtB, \
	const int *d_indloss, const double *d_A, double *d_AtA, double *d_At_block, double *d_AtA_block, \
	int nnz, int nnz_indloss, int T, int P, int N, int E, double lmbd, double l2x) {
		
	const double one = 1.0;
	const double zero = 0.0;
	const int Ncolbm = 2 * P + 1;
	const int TN = T * N;
	const int NN = N * N;
	const int NP = N * P;
	
	// -> Calculate the cross-products before building the big sparse kronecker matrix
	// AtA = A'*A
	cublasStatus_t stat_cublas = cublasDsyrk(handle_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, NP, N, &one, d_A, N, &zero, d_AtA, NP);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	//CUDA_CALL(cudaDeviceSynchronize());
	if (stat_cublas != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS matrix multiplication failed\n");
    }
	
	// cublasDsyrk only filled the lower triangular part, so we fill up the other part
	int GRID_SIZE_NPtri = (uint32_T)ceil((NP * (NP - 1) / 2) / (BLOCK_SIZE + 0.0));
	fill_upper_triangular << <GRID_SIZE_NPtri, BLOCK_SIZE >> > (d_AtA, NP);
	
	// -> Transpose the autoregressive matrices, separately per block
	int GRID_SIZE_A_block = (uint32_T)ceil((N * NP) / (BLOCK_SIZE + 0.0));
	// Bt = transf(B), where B = [A(1), ..., A(P)] and Bt = [A(1)'; ...; A(P)']
	transpose_A_block << <GRID_SIZE_A_block, BLOCK_SIZE >> > (d_At_block, d_A, N, P);
	
	// -> Create the big sparse kronecker-sums matrix using the equivalent banded matrix representation
	
	// => First, initialize the banded matrix with the identity (add to the central block)
	//CUDA_CALL(cudaMemset(d_bandMat, 0, T * Ncolbm * NN * sizeof(double)));
	int GRID_SIZE_allbanded = (uint32_T)ceil((T * Ncolbm * NN) / (BLOCK_SIZE + 0.0));
	reset_to_zero << <GRID_SIZE_allbanded, BLOCK_SIZE >> > (d_bandMat, T * Ncolbm * NN);
	
	int GRID_SIZE_TN = (uint32_T)ceil(TN / (BLOCK_SIZE + 0.0));
	insert_identity << <GRID_SIZE_TN, BLOCK_SIZE >> > (d_bandMat, TN, N, P);
	
	// => Second, substract the linear terms to the upper and lower diagonals
	int nnzb = T - P;
	int GRID_SIZE_colblock = (uint32_T)ceil((nnzb * NN) / (BLOCK_SIZE + 0.0));
	for (int p = 1; p <= P; p++) {
		// insert -A into the banded matrix structure (upper triangular part)
		insert_minus_block << <GRID_SIZE_colblock, BLOCK_SIZE >> > (d_bandMat, d_A + NN * (p - 1), TN, N, P + p, 0, nnzb);
		// insert -A' into the banded matrix structure (lower triangular part)	
		insert_minus_block << <GRID_SIZE_colblock, BLOCK_SIZE >> > (d_bandMat, d_At_block + NN * (p - 1), TN, N, P - p, p, nnzb);
	}
	
	// => Third, add the quadratic terms
	int GRID_SIZE_NN = (uint32_T)ceil(NN / (BLOCK_SIZE + 0.0));
	for (int k = 1; k <= P; k++) {
		for (int l = 1; l <= P; l++) {
			// => fill up the cross-product block
			copy_block << <GRID_SIZE_NN, BLOCK_SIZE >> > (d_AtA_block, d_AtA, NP, N, k - 1, l - 1);	
			// => add it to the banded matrix
			add_block << <GRID_SIZE_colblock, BLOCK_SIZE >> > (d_bandMat, d_AtA_block, TN, N, P + k - l, MAX(0,l-k) + MIN(k,l), nnzb);
		}
	}
	
	// => Fourth, multiply the values in the banded matrix by lmbd
	multiply_banded_values_by_lambda << <GRID_SIZE_allbanded, BLOCK_SIZE >> > (d_bandMat, lmbd, T * Ncolbm * NN);
	
	// => Fifth, add kron(diag(loss.^2),BtB) to the diagonal in banded matrix format
	// Because loss(t) is either 0 or 1 it is enough to know the index of nonzeros values in the loss vector
	int GRID_SIZE_indcolblock = (uint32_T)ceil((nnz_indloss * NN) / (BLOCK_SIZE + 0.0));
	add_block_BtB_diag << <GRID_SIZE_indcolblock, BLOCK_SIZE >> > (d_bandMat, d_BtB, d_indloss, TN, N, P, nnz_indloss);
	
	// => Sixth, add l2x to the main diagonal, which also guarantee that the matrix is definite positive
	add_scalar_identity << <GRID_SIZE_TN, BLOCK_SIZE >> > (d_bandMat, TN, N, P, l2x);
	
	// => Seventh, reading Values from banded matrix to sparse (CSR) format
	int GRID_SIZE_nnz = (uint32_T)ceil(nnz / (BLOCK_SIZE + 0.0));
	bmr2sparse << <GRID_SIZE_nnz, BLOCK_SIZE >> > (d_csrVal, d_bandMat, nnz, d_bmrInd);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// -> Finally, copy to Matlab sparse array to do the division in their engine UNTIL THERE IS BETTER/FASTER cusolver/cusparse division function
	CUDA_CALL(cudaMemcpy(mxGetPr(rhs[0]), d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost));
	mexCallMATLAB(1, &lhs, 2, rhs, "mldivide");
	
	// copy the solution to device
	CUDA_CALL(cudaMemcpy(d_x, mxGetPr(lhs), N * T * E * sizeof(double), cudaMemcpyHostToDevice));
}

// ->->->->-> Main entry point and its directly supportive GPU kernel function start here <-<-<-<-<- \\

// flip the elements of the matrix in the left/right direction. d_A dimension is N x T
__global__ void fliplr(double *d_A, int N, int T) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < N * (T / 2)) {
		int irow = i % N;
		int icol = i / N;
		int indmirror = (T - icol - 1) * N + irow;
		double tmp = d_A[i];
		d_A[i] = d_A[indmirror];
		d_A[indmirror] = tmp;
	}
}

__global__ void fliplr3d(double *d_A, int N, int T, int E) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int Thalf = T / 2;
	int NThalf = N * Thalf;
	
	if (i < NThalf * E) {
		int e = i / NThalf;
		int ind = i % NThalf;
		int t = ind / N;
		i = e * N * T + t * N + (ind % N);
		ind = e * N * T + (T - t - 1) * N + (ind % N);
		double tmp = d_A[i];
		d_A[i] = d_A[ind];
		d_A[ind] = tmp;
	}
}

// create_BtYvec
// only fill up those entries for loss[t] != 0, where nnz is the number of nonzeros in the loss array.
__global__ void create_BtYvec(double *d_BtYvec, const double *d_BtY, const int *d_indloss, int T, int N, int E, int nnz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int Nnnz = N * nnz;
	
	if (i < Nnnz * E) {
		int ind = i % Nnnz;
		int t = d_indloss[ind / N];
		ind = (i / Nnnz) * (T * N) + t * N + (ind % N);
		d_BtYvec[ind] = d_BtY[ind];
	}
}

// next solution in GD's direction
// x = x_old - alpha*dx;
__global__ void move_gd_step(double *d_x, double *d_x_old, double *d_dx, int T, int N, int E, double alpha) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < T * N * E) {
		d_x[i] = d_x_old[i] - alpha * d_dx[i];
	}
}

// function [xe, Ae, iter] = cmex_HGDALS_GPU(y, x, B, A, lmbd, l2x, l2a, Niter, tol, loss)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// -> Pointers for input variables in device
    double *d_y = NULL, *d_x = NULL, *d_x_old = NULL, *d_B = NULL, *d_A = NULL, *d_A_old = NULL, *d_loss = NULL; // input variables in device
	
	// -> Pointers for input variables in host
	double *h_loss = NULL;
	
	// -> Pointers for evaluation of cost function (host)
	double *h_A = NULL, *h_xerr = NULL, *h_yerr = NULL;
	
	// -> Pointers for evaluation of cost function (device) and x's gradient descent operation
	double *d_xerr = NULL, *d_dx = NULL, *d_yerr = NULL, *d_B_x = NULL, *d_xerr_old = NULL, *d_yerr_old = NULL; // auxiliar variables in device 
	
	// -> Pointers for the big sparse (Kronecker's sum) matrix: x's OLS operation
	// device
	double *d_bandMat = NULL, *d_csrVal = NULL, *d_AtA = NULL, *d_AtA_block = NULL, *d_At_block = NULL, *d_BtYvec = NULL, *d_BtB = NULL;
	int *d_bmrInd = NULL, *d_indloss = NULL;
	
	// -> Pointers for the B = YZt/ZZt operation, or equivalently, Bt = ZZt\ZYt: A's OLS operation
	double *d_X0 = NULL, *d_dX0 = NULL, *d_Z = NULL, *d_ZZt = NULL, *d_ZXt = NULL;
	int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
	
	// -> Handles for CUDA libraries
	// CUSPARSE
	cusparseHandle_t handle_cusparse = NULL;
	// CUBLAS
	cublasHandle_t handle_cublas = NULL;
	// CUSOLVER dense and stream
    cusolverDnHandle_t handle_Dncusolver = NULL;
    cudaStream_t stream = NULL;	
	
	// -> GD's tested step sizes
	// const int Nstep = 7;
	//double alpha_vec_GD[Nstep] = { 1, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6 };
	const int Nstep = 5;
	double alpha_vec_GD[Nstep] = { 1, 1.0e-1, 1.0e-2, 1.0e-5, 1.0e-10};
	
	// -> Dimension variables
	// mwSize M, N, P, T, T_p, NN, NT, MT, NP, Ncolbm;
	int M, N, T, E, P, Ncolbm;
	
	// -> Number of nonzero elements in big sparse (Kronecker's sum) matrix
	int nnz;
	
	// -> Check input
    if (nrhs != 10) mexErrMsgTxt("Nine inputs required: [xe, Ae, iter] = cmex_HGDALS_GPU(y, x, B, A, lmbd, l2x, l2a, Niter, tol, loss)");
	
    // dimensions	
	const mwSize *y_dims = mxGetDimensions(prhs_y);
	M = y_dims[0];
	T = y_dims[1];
	E = y_dims[2];
	const mwSize *x_dims = mxGetDimensions(prhs_x);
	N = x_dims[0];
    const mwSize *A_dims = mxGetDimensions(prhs_A);
	P = A_dims[2];
	
	// pointers to input data
	if (mxGetNumberOfDimensions(prhs_y) == 2) {
		if (mxGetClassID(prhs_y) != mxDOUBLE_CLASS)
			mexErrMsgTxt("The 1st input (y) is either a 2D or 3D matrix. Dimension either MxT or MxTxNE (double).");
		E = 1;
	}
	else if ((mxGetNumberOfDimensions(prhs_y) != 3) || (mxGetClassID(prhs_y) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 1st input (y) is either a 2D or 3D matrix. Dimension either MxT or MxTxNE (double).");
	
	if (mxGetNumberOfDimensions(prhs_x) == 2) {
		if (mxGetClassID(prhs_x) != mxDOUBLE_CLASS)
			mexErrMsgTxt("The 2nd input (xe) is either a 2D or 3D matrix. Dimension either NxT or NxTxNE (double).");
	}
	else if ((mxGetNumberOfDimensions(prhs_x) != 3) || (x_dims[2] != E) || (mxGetClassID(prhs_x) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 2nd input (xe) is either a 2D or 3D matrix. Dimension either NxT or NxTxNE (double).");
	
	if ((mxGetNumberOfDimensions(prhs_B) != 2) || (mxGetClassID(prhs_B) != mxDOUBLE_CLASS) ||
		(mxGetM(prhs_B) != M) || (mxGetN(prhs_B) != N))
        mexErrMsgTxt("The 3rd input (B) is a 2D matrix. Dimension MxN (double).");
	
	if (mxGetNumberOfDimensions(prhs_A) == 2) {
		P = 1;
		if ((mxGetClassID(prhs_A) != mxDOUBLE_CLASS) || (mxGetM(prhs_A) != N) || (mxGetN(prhs_A) != N))
			mexErrMsgTxt("The 4th input (Ae) is either a 2D or a 3D matrix. Dimension either NxN or NxNxp (double).");
	}
	else if (mxGetNumberOfDimensions(prhs_A) == 3) {
		if ((mxGetClassID(prhs_A) != mxDOUBLE_CLASS) || (A_dims[0] != N) || (A_dims[1] != N))
			mexErrMsgTxt("The 4th input (Ae) is either a 2D or a 3D matrix. Dimension either NxN or NxNxp (double).");
	}
	else {
		mexErrMsgTxt("The 4th input (Ae) is either a 2D or a 3D matrix. Dimension either NxN or NxNxp (double).");
	}
	
	if (!mxIsScalar(prhs_lmbd) || (mxGetClassID(prhs_lmbd) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 5th input (lmbd) is a scalar (double).");
	
	if (!mxIsScalar(prhs_l2x) || (mxGetClassID(prhs_l2x) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 6th input (l2x) is a scalar (double).");
	
	if (!mxIsScalar(prhs_l2a) || (mxGetClassID(prhs_l2a) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 7th input (l2a) is a scalar (double).");
	
	if (!mxIsScalar(prhs_Niter) || (mxGetClassID(prhs_Niter) != mxINT32_CLASS))
        mexErrMsgTxt("The 8th input (Niter) is a scalar (int32).");
	
	if (!mxIsScalar(prhs_tol) || (mxGetClassID(prhs_tol) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 9th input (tol) is a scalar (double).");
	
	if ((mxGetNumberOfDimensions(prhs_loss) != 2) || (mxGetClassID(prhs_loss) != mxDOUBLE_CLASS) ||
		(mxGetM(prhs_loss) != 1) || (mxGetN(prhs_loss) != T))
        mexErrMsgTxt("The 10th input (loss) is a row vector. Dimension 1xT (double).");
	
	// scalar input arguments
	const double lmbd = mxGetScalar(prhs_lmbd);
	const double l2x = mxGetScalar(prhs_l2x);
	const double l2a = mxGetScalar(prhs_l2a);
	const int32_T Niter = (int32_T)mxGetScalar(prhs_Niter);
	const double tol = mxGetScalar(prhs_tol);
	
	// other input-related initializations
	Ncolbm = 2 * P + 1;
	
	// -> Check output
	if ((nlhs != 2) && (nlhs != 3))
		mexErrMsgTxt("Only two or three outputs are allowed.");
	
	// Print input-related information
	printf("T = %d, M = %d, N = %d, E = %d, P = %d, Ncolbm = %d, lmbd = %.6f, l2x = %.6f, l2a = %.6f\n", T, M, N, E, P, Ncolbm, lmbd, l2x, l2a);
    
    // -> Get the pointers in GPU
	CUDA_CALL(cudaMalloc((void**)&d_y, M * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_x, N * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_B, M * N * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_A, N * N * P * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_loss, T * sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_y, mxGetPr(prhs_y), M * T * E * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_x, mxGetPr(prhs_x), N * T * E * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_B, mxGetPr(prhs_B), M * N * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_A, mxGetPr(prhs_A), N * N * P * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_loss, mxGetPr(prhs_loss), T * sizeof(double), cudaMemcpyHostToDevice));
	
	// -> Get some of the input pointers in CPU
    h_loss = (double*)mxGetPr(prhs_loss);

	// -> Allocate memory for some of the pointers in CPU
    h_A = (double*)malloc(N * N * P * sizeof(double));
    h_xerr = (double*)malloc(N * T * E * sizeof(double));
    h_yerr = (double*)malloc(M * T * E * sizeof(double));
    
	// -> Auxiliar variables allocated on GPU
	CUDA_CALL(cudaMalloc((void**)&d_x_old, N * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_A_old, N * N * P * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_xerr, N * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_xerr_old, N * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_dx, N * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_yerr, M * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_yerr_old, M * T * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_B_x, M * T * E * sizeof(double)));
	
	// -> Create handle for CUSPARSE
	CUSPARSE_CALL(cusparseCreate(&handle_cusparse));
	
	// -> Create a handle for CUBLAS
    cublasStatus_t stat = cublasCreate(&handle_cublas);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS initialization failed\n");
    }
	
	// -> Create a handle for CUSOLVER dense and assign a stream to the handle
	CUSOLVER_CALL(cusolverDnCreate(&handle_Dncusolver));
    CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CALL(cusolverDnSetStream(handle_Dncusolver, stream));
	
	// -> Change time and loss to reversal order
	int GRID_SIZE_NTE = (uint32_T)ceil((N * T * E) / (BLOCK_SIZE + 0.0f));
	
	int GRID_SIZE_NThalfE = (uint32_T)ceil((N * (T / 2) * E) / (BLOCK_SIZE + 0.0f));
	fliplr3d << <GRID_SIZE_NThalfE, BLOCK_SIZE >> > (d_x, N, T, E);
	
	int GRID_SIZE_MThalfE = (uint32_T)ceil((M * (T / 2) * E) / (BLOCK_SIZE + 0.0f));
	fliplr3d << <GRID_SIZE_MThalfE, BLOCK_SIZE >> > (d_y, M, T, E);
	
	int GRID_SIZE_Thalf = (uint32_T)ceil((T / 2) / (BLOCK_SIZE + 0.0f));
	fliplr << <GRID_SIZE_Thalf, BLOCK_SIZE >> > (d_loss, 1, T);
	
	// -> Create Matlab rhs and lhs pointers for division
	mxArray *rhs[2];
	rhs[1] = mxCreateDoubleMatrix(N * T, E, mxREAL);
	mxArray *lhs = mxCreateDoubleMatrix(N * T, E, mxREAL);
	
	// => Running indices through the sparse diagonal space to map sparse indices.
	// coo: CUSPARSE coordinate format.
	// bmr: banded matrix format
	nnz = (T * Ncolbm - P * (P + 1)) * (N * N);
	
	rhs[0] = mxCreateSparse(N * T, N * T, nnz, mxREAL);
	mwIndex *irowptr = mxGetIr(rhs[0]); // length is nnz
    mwIndex *icolptr = mxGetJc(rhs[0]); // length is N * T + 1
	CUDA_CALL(cudaMalloc((void**)&d_csrVal, nnz * sizeof(double))); // where the values of the sparse matrix are going to reside in device
	
	//printf("Line %d:",__LINE__); mexErrMsgTxt("STOP HERE");
	
	// this may be confusing but no if it is taken into account that Matlab's sparse indices run in a row-major order,
	// compressed (in terms of csr's CUSPARSE format) by columns, whereas CUSPARSE indices run in a column-major order,
	// compressed by rows.
	int *h_cooRowInd = (int*)malloc(nnz * sizeof(int));
	int *h_bmrInd = (int*)malloc(nnz * sizeof(int));
	int ind = 0;
	for (int irb = 0; irb < T; irb++) {
		for (int ir = 0; ir < N; ir++) {
			int irow = irb * N + ir;
			for (int icb = MAX(0,irb-P); icb < MIN(T,irb+P+1); icb++) {
				for (int ic = 0; ic < N; ic++) {
					if (ind == nnz) {
						printf("Line %d: ", __LINE__);
						mexErrMsgTxt("Unexpected index access violation.");
					}
					h_cooRowInd[ind] = irow;
					irowptr[ind] = icb * N + ic;
					int icol_bmr = (P + icb - irb) * N + ic;
					h_bmrInd[ind] = icol_bmr * N * T + irow;
					ind++;
				}
			}
		}
	}
	if (ind != nnz) mexErrMsgTxt("The total number of accounted indices must be equals to nnz.");
	
	// => converting RowInd (COO) to RowPtr (CSR) format in device, then copy the csrRowPtr to corresponding Matlab's sparse pointer.
	int *d_cooRowInd, *d_csrRowPtr;
	CUDA_CALL(cudaMalloc((void**)&d_cooRowInd, nnz * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_cooRowInd, h_cooRowInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc((void**)&d_csrRowPtr, (N * T + 1) * sizeof(int)));
	CUSPARSE_CALL(cusparseXcoo2csr(handle_cusparse, d_cooRowInd, nnz, N * T, d_csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
	int *h_csrRowPtr = (int*)malloc((N * T + 1) * sizeof(int));
	CUDA_CALL(cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (N * T + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i <= N * T; i++)
		icolptr[i] = h_csrRowPtr[i];
	
	// => writing to device the banded matrix indices
	CUDA_CALL(cudaMalloc((void**)&d_bmrInd, nnz * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_bmrInd, h_bmrInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
	
	// free memory that will not be used after this
	free(h_cooRowInd);
	free(h_bmrInd);
	free(h_csrRowPtr);
	cudaFree(d_cooRowInd);
	cudaFree(d_csrRowPtr);	
	
	// -> Separating memory OUTSIDE the main loop
	CUDA_CALL(cudaMalloc((void**)&d_AtA, N * P * N * P * sizeof(double)));	
	CUDA_CALL(cudaMalloc((void**)&d_AtA_block, N * N * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_At_block, N * N * P * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_bandMat, T * Ncolbm * N * N * sizeof(double)));
	
	// -> Separate memory for and create d_BtYvec, d_BtB, and d_indloss OUTSIDE of the main loop
	
	// => indloss
	int nnz_indloss = 0;
	for (int t = 0; t < T; t++) {
		if (h_loss[t] != 0) nnz_indloss++;
	}
	int *h_indloss = (int*)malloc(nnz_indloss * sizeof(int));
	nnz_indloss = 0;
	for (int t = 0; t < T; t++) {
		if (h_loss[t] != 0) {
			h_indloss[nnz_indloss] = t;
			nnz_indloss++;
		}
	}
	CUDA_CALL(cudaMalloc((void**)&d_indloss, nnz_indloss * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_indloss, h_indloss, nnz_indloss * sizeof(int), cudaMemcpyHostToDevice));
	
	// free memory that will not be used after this
	free(h_indloss);
	
	// => BtYvec
	
	// BtY = Bt * y
	double *d_BtY = NULL;
	const double one = 1.0, zero = 0.0;
	CUDA_CALL(cudaMalloc((void**)&d_BtY, N * T * E * sizeof(double)));
	cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, T * E, M, &one, d_B, M, d_y, M, &zero, d_BtY, N);
	
	CUDA_CALL(cudaDeviceSynchronize()); // synch always after calling any of CUDA library's functions
	
	// BtYvec = vec(BtY * diag(loss))
	CUDA_CALL(cudaMalloc((void**)&d_BtYvec, N * T * E * sizeof(double)));
	//CUDA_CALL(cudaMemset(d_BtYvec, 0, NT * sizeof(double)));
	reset_to_zero << <GRID_SIZE_NTE, BLOCK_SIZE >> > (d_BtYvec, N * T * E);
	int GRID_SIZE_NEnnz = (uint32_T)ceil((nnz_indloss * N * E) / (BLOCK_SIZE + 0.0f));
	create_BtYvec << <GRID_SIZE_NEnnz, BLOCK_SIZE >> > (d_BtYvec, d_BtY, d_indloss, T, N, E, nnz_indloss);
	
	// -> Set rhs[1] to copy of BtYvec
	CUDA_CALL(cudaMemcpy(mxGetPr(rhs[1]), d_BtYvec, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	
	// free memory that will not be used after this
	cudaFree(d_BtY);
	
	// => d_BtB
	CUDA_CALL(cudaMalloc((void**)&d_BtB, N * N * sizeof(double)));
	cublasStatus_t stat_cublas = cublasDsyrk(handle_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, N, M, &one, d_B, M, &zero, d_BtB, N);
	if (stat_cublas != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS matrix multiplication failed\n");
    }
	
	// cublasDsyrk only filled the lower triangular part, so we fill up the other part
	int GRID_SIZE_NNtri = (uint32_T)ceil((N * (N - 1) / 2) / (BLOCK_SIZE + 0.0));
	fill_upper_triangular << <GRID_SIZE_NNtri, BLOCK_SIZE >> > (d_BtB, N);
	
	// -> Separate memory for d_X0, d_dX0, d_Z, d_ZZt, d_ZXt, d_Ipiv, and d_info OUTSIDE of the main loop
	const int pivot = 1;
	CUDA_CALL(cudaMalloc((void**)&d_X0, N * (T - P) * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_dX0, N * (T - P) * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_Z, N * P * (T - P) * E * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_ZZt, N * P * N * P * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_ZXt, N * P * N * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_Ipiv, N * P * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_info, sizeof(int)));
	
	// -> Main algorithm
    int iter, italpha, NGD = 1;
	double TALS, TGD, FGain_ALS, FGain_GD, F, Fprev, alpha;
    
	//double h_tmp[14];
	Fprev = eval_cost(handle_cublas, h_A, h_xerr, h_yerr, h_loss, d_xerr, d_yerr, d_y, d_B, d_x, d_A, d_loss, d_B_x, lmbd, l2x, l2a, T, M, N, E, P);
	printf("Initial value: F = %.6f\n", Fprev);
	
	mxArray *mat_iter = mxCreateDoubleMatrix(1, 1, mxREAL);
	double *iter_ptr = mxGetPr(mat_iter);
	
	for (iter = 0; iter < Niter; iter++) {
		*iter_ptr = iter + 1;
		mexCallMATLAB(0, NULL, 1, &mat_iter, "disp"); // to flush
		
		// => ALS algorithm
		CUDA_CALL(cudaDeviceSynchronize());
		auto begin = std::chrono::high_resolution_clock::now();	
		
		// >> Update x
		calculate_x_OLS(handle_cublas, rhs, lhs, d_x, d_bandMat, d_csrVal, d_bmrInd, \
			d_BtYvec, d_BtB, d_indloss, d_A, d_AtA, d_At_block, d_AtA_block, nnz, nnz_indloss, T, P, N, E, lmbd, l2x);
			
		// >> Update A
		calculate_A_OLS(handle_cublas, handle_Dncusolver, d_A, d_x, d_X0, d_Z, d_ZZt, d_ZXt, d_Ipiv, d_info, pivot, T, M, N, P, E, l2a);
		
		// >> Evaluate cost function after ALS
		// Be AWARE that this function MODIFIES xerr and yerr, WHICH ARE USED in function "calculate_x_GD" below.
		F = eval_cost(handle_cublas, h_A, h_xerr, h_yerr, h_loss, d_xerr, d_yerr, d_y, d_B, d_x, d_A, d_loss, d_B_x, lmbd, l2x, l2a, T, M, N, E, P);
		printf("ALS: F = %.10f\n", F);
		
		// >> Calculate ALS algorithm's elapsed time
		CUDA_CALL(cudaDeviceSynchronize());
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
		TALS = elapsed.count() * 1e-9;
		printf("Elapsed time (ALS) is %.3f seconds.\n", TALS);
		
		// >> Evaluate gain on ALS algorithm
		FGain_ALS = Fprev - F; // ALS always improve the solution, therefore FGain_ALS > 0
		Fprev = F;
		
		// >> Check for unexpected condition
		if (FGain_ALS < 0) {
			// Restore x and A old values
			CUDA_CALL(cudaMemcpy(d_x, d_x_old, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
			CUDA_CALL(cudaMemcpy(d_A, d_A_old, N * N * P * sizeof(double), cudaMemcpyDeviceToDevice));
			mexWarnMsgIdAndTxt("ALS:convergence", "The ALS algorithm should have converged before, in the GD step, maybe the X's OLS operation failed.");
            break;
		}
		
		if (FGain_ALS < tol) break; // convergence criteria
	
		// => Calculate number of iterations for gradient descent
		if (iter > 0) {
			NGD = (int)ceil((FGain_GD * TALS) / (FGain_ALS * TGD));
			if (NGD < 1) NGD = 1;
			// GD is allowed to run for no more than 20% of ALS runtime.
			NGD = MIN(NGD, (int)ceil(0.2 * TALS / TGD));
			//if (NGD > 10000) NGD = 10000;
		}
	
        // => Alternatively, use gradient descent
		CUDA_CALL(cudaDeviceSynchronize());
		begin = std::chrono::high_resolution_clock::now();
		
		// Best solution so far
		double Fbest = F, Fslope;
		CUDA_CALL(cudaMemcpy(d_x_old, d_x, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_A_old, d_A, N * N * P * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_xerr_old, d_xerr, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_yerr_old, d_yerr, M * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
		
		int itgd;
		for (itgd = 0; itgd < NGD; itgd++) {
			// >> Calculate gradient descent direction for x
			calculate_x_GD(handle_cublas, d_dx, d_B, d_x, d_X0, d_dX0, d_A, d_xerr, d_yerr, T, M, N, P, E, lmbd, l2x);	
			
			// >> Choose the highest successful step
			
			// optimise according to different step sizes
			for (italpha = 0; italpha < Nstep; italpha++) {
				alpha = alpha_vec_GD[italpha];
				
				// >>> Update x
				// x = x_old - alpha*dx;
				move_gd_step << <GRID_SIZE_NTE, BLOCK_SIZE >> > (d_x, d_x_old, d_dx, T, N, E, alpha);
				
				// >>> Update A
				calculate_A_OLS(handle_cublas, handle_Dncusolver, d_A, d_x, d_X0, d_Z, d_ZZt, d_ZXt, d_Ipiv, d_info, pivot, T, M, N, P, E, l2a);
				
				// >>> Evaluate cost function after GD
				// Be AWARE that this function MODIFIES "xerr" and "yerr", WHICH ARE USED in function "calculate_x_GD" above.
				// However, the current implementation is consistent as, after the break below, for the next "calculate_x_GD"
				// the updated values for xerr and yerr will be used corretly.
				F = eval_cost(handle_cublas, h_A, h_xerr, h_yerr, h_loss, d_xerr, d_yerr, \
					d_y, d_B, d_x, d_A, d_loss, d_B_x, lmbd, l2x, l2a, T, M, N, E, P);
					
				// >>> Evaluate break condition
				if (F < Fbest) break;
			}
			Fslope = (Fbest - F) / alpha;
			
			// clone x if solution improved: x_old = x;
			if (F < Fbest) {
				Fbest = F;
				CUDA_CALL(cudaMemcpy(d_x_old, d_x, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_A_old, d_A, N * N * P * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_xerr_old, d_xerr, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_yerr_old, d_yerr, M * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
			}
			else {
				// Restore old (best) values
				CUDA_CALL(cudaMemcpy(d_x, d_x_old, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_A, d_A_old, N * N * P * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_xerr, d_xerr_old, N * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_yerr, d_yerr_old, M * T * E * sizeof(double), cudaMemcpyDeviceToDevice));
				break; // off-from GD's loop: no further improvement
			}
			
			// >> Evaluate stop condition
			//if (italpha == Nstep - 1) break;
		}
		
		// => Evaluate gain on GD algorithm
		if (F < Fprev) {
			FGain_GD = Fprev - F; // GD either improves the solution, therefore FGain_GD > 0, or reached convergence 
			Fprev = F;
		}
		else {
			FGain_GD = 0;
		}
		printf("GD (%d of %d loops): alpha = %.8f, F = %.10f, dF/alpha = %.10f\n", itgd, NGD, alpha, F, Fslope);
		
		// => Calculate GD algorithm's elapsed time
		CUDA_CALL(cudaDeviceSynchronize());
		end = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
		TGD = elapsed.count() * 1e-9;
		printf("Elapsed time (GD) is %.3f seconds.\n", TGD);
		
		// -> Calculate the average time for one iteration unit
		TGD = TGD / (MIN(NGD,itgd+1));
	}
	
	// -> Time reversal of x
	fliplr3d << <GRID_SIZE_NThalfE, BLOCK_SIZE >> > (d_x, N, T, E);
	
	// -> Wrap the result up as a MATLAB gpuArray for return
	mwSize dim[3] = { N, T, E };
	plhs[0] = mxCreateNumericArray(3, dim, mxDOUBLE_CLASS, mxREAL);
	CUDA_CALL(cudaMemcpy(mxGetPr(plhs[0]), d_x, N * T * E * sizeof(double), cudaMemcpyDeviceToHost));
	
	dim[0] = N;
	dim[1] = N;
	dim[2] = P;
	plhs[1] = mxCreateNumericArray(3, dim, mxDOUBLE_CLASS, mxREAL);
	CUDA_CALL(cudaMemcpy(mxGetData(plhs[1]), d_A, N * N * P * sizeof(double), cudaMemcpyDeviceToHost));
	
	if (nlhs > 2) {
		plhs[2] = mat_iter;	
	}
	else mxDestroyArray(mat_iter);
	
	// Free Matlab arrays
	mxDestroyArray(lhs);
	mxDestroyArray(rhs[0]);
    mxDestroyArray(rhs[1]);
    
    // Free CPU memory
    free(h_A);
    free(h_xerr);
    free(h_yerr);
	
	// Free GPU memory
	cudaFree(d_y);
	cudaFree(d_x);
	cudaFree(d_x_old);
	cudaFree(d_B);
	cudaFree(d_A);
	cudaFree(d_A_old);
	cudaFree(d_loss);	
	cudaFree(d_xerr);
	cudaFree(d_xerr_old);
	cudaFree(d_dx);
	cudaFree(d_yerr);
	cudaFree(d_yerr_old);
	cudaFree(d_B_x);
	cudaFree(d_bandMat);
	cudaFree(d_csrVal);
	cudaFree(d_AtA);
	cudaFree(d_AtA_block);
	cudaFree(d_At_block);
	//cudaFree(d_BtY);
	cudaFree(d_BtYvec);
	cudaFree(d_BtB);	
	//cudaFree(d_cooRowInd);
	//cudaFree(d_csrRowPtr);
	cudaFree(d_bmrInd);
	cudaFree(d_indloss);
	cudaFree(d_X0);
	cudaFree(d_dX0);
	cudaFree(d_Z);
	cudaFree(d_ZZt);
	cudaFree(d_ZXt);
	cudaFree(d_Ipiv);
	
	// Free CUDA libraries handles and streams
	// CUBLAS
	cublasDestroy(handle_cublas); // destroy CUBLAS context
	// CUSPARSE
	CUSPARSE_CALL(cusparseDestroy(handle_cusparse));
	// CUSOLVER dense and stream
	cusolverDnDestroy(handle_Dncusolver);
    cudaStreamDestroy(stream);
	
	//cudaDeviceReset();  // kill the GPU (complete reset) but then in Matlab must run reset(gpuDevice); to get it back.
						// However, everything is sorted out just by avoiding the use of Matlab's GPUArray functions.
}