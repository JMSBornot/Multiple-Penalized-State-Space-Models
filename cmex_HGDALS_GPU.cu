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

__global__ void multiply_loss(double *d_yerr, double *d_y, double *d_B_xe, double *d_loss, int M, int T) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < M * T) {
		int t = i / M;
		d_yerr[i] = (d_y[i] - d_B_xe[i]) * d_loss[t];
	}
}

// Main function which call all the kernel functions defined above
inline double eval_cost(cublasHandle_t handle, double *h_A, double *h_xerr, double *h_yerr, double *h_loss, \
	double *d_xerr, double *d_yerr, double *d_y, double *d_B, double *d_x, double *d_A, double *d_loss, double *d_B_x, \
	double lmbd, double l2x, double l2a, int M, int N, int p, int T) {
	
	int NN = N * N;
	double F = 0.0;
	double tmp, acc;
	const double one = 1.0, zero = 0.0, minus_one = -1.0;
	
	// -> xerr(:) = x(:);
	CUDA_CALL(cudaMemcpy(d_xerr, d_x, N * T * sizeof(double), cudaMemcpyDeviceToDevice));
	
	// -> Update the errors while taking into account the time reversal order
	// xerr(:,1:T-p) = xerr(:,1:T-p) - A(:,:,k)*x(:,(1:T-p)+k);
	for (int k = 0; k < p; k++) {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, T - p, N, &minus_one, d_A + NN * k, N, d_x + N * (k + 1), N, &one, d_xerr, N);
		CUDA_CALL(cudaDeviceSynchronize());
	}

	// -> B * x
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, T, N, &one, d_B, M, d_x, N, &zero, d_B_x, M);
	CUDA_CALL(cudaDeviceSynchronize());
	
	// -> yerr = (y - B * x) .* loss
	int GRID_SIZE_MT = (uint32_T)ceil((M * T) / (BLOCK_SIZE + 0.0f));
	multiply_loss << <GRID_SIZE_MT, BLOCK_SIZE >> > (d_yerr, d_y, d_B_x, d_loss, M, T);
	
	// Transfer from device to host the vars needed for cost function calculation
	CUDA_CALL(cudaMemcpy(h_xerr, d_xerr, N * T * sizeof(double), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_yerr, d_yerr, M * T * sizeof(double), cudaMemcpyDeviceToHost));
	
	for (int t = 0; t < T; t++) {
		int t_M = t * M;
		if (h_loss[t] != 0) {
			for (int i = 0; i < M; i++) {
				tmp = h_yerr[t_M + i];
				F += tmp * tmp;
			}
		}
	}
	
	acc = 0.0;
	for (int t = 0; t < T; t++) {
		int t_N = t * N;
		for (int i = 0; i < N; i++) {
			tmp = h_xerr[t_N + i];
			acc += tmp * tmp;
		}
	}
	F += lmbd * acc;
	
	if (l2x != 0) {
		CUDA_CALL(cudaMemcpy(h_xerr, d_x, N * T * sizeof(double), cudaMemcpyDeviceToHost));
		acc = 0.0;
		for (int t = 0; t < T; t++) {
			int t_N = t * N;
			for (int i = 0; i < N; i++) {
				tmp = h_xerr[t_N + i];
				acc += tmp * tmp;
			}
		}
		F += l2x * acc;
	}
	
	if (l2a != 0) {
		CUDA_CALL(cudaMemcpy(h_A, d_A, NN * p * sizeof(double), cudaMemcpyDeviceToHost));
		acc = 0.0;
		for (int k = 0; k < p; k++) {
			int k_NN = k * NN;
			for (int i = 0; i < NN; i++) {
				tmp = h_A[k_NN + i];
				acc += tmp * tmp;
			}
		}
		F += l2a * acc;
	}
	
	//F = @(x,A,xerr) (sum(sum(((y - B*x).*loss).^2)) + lmbd*sum(sum(xerr(:,p+1:T).^2)) + lmbd*sum(sum(x(:,1:p).^2)) + l2a*sum(A(:).^2))/T;
	
	F = F / T;
	return F;
}

// ->->->->-> Calculate x's gradient descent (d_dx: dim is NxT) direction <-<-<-<-<- \\

__global__ void update_dx(double *d_dx, const double *d_xerr, double lmbd, int T, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < T * N) {
		d_dx[i] += lmbd * d_xerr[i];
	}
}

__global__ void dx_div_T(double *d_dx, int T, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < N * T) {
		d_dx[i] /= T;
	}
}

// Main function which call all the kernel functions defined above
inline void calculate_x_GD(cublasHandle_t handle, double *d_dx, const double *d_B, const double *d_x, \
	const double *d_A, const double *d_xerr, const double *d_yerr, int T, int M, int N, int p, double lmbd) {
	
	const double one = 1.0, minus_one = -1.0, zero = 0.0, minus_lmbd = -lmbd;
	const int NN = N * N;
	const int T_p = T - p;
	
	// -> Calculate dx
	
	// => dx = -B' * ((y - B * x) .* loss)
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, T, M, &minus_one, d_B, M, d_yerr, M, &zero, d_dx, N);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// => dx = dx + lmbd * xerr
	int GRID_SIZE_TN = (uint32_T)ceil((T * N) / (BLOCK_SIZE + 0.0f));
	update_dx << <GRID_SIZE_TN, BLOCK_SIZE >> > (d_dx, d_xerr, lmbd, T, N);
	
	// => dx(:,(1:T-p)+k) = dx(:,(1:T-p)+k) - lmbd * (A(:,:,k)' * xerr(:,1:T-p)); // in time-reversal order
	for (int k = 0; k < p; k++) {
		CUDA_CALL(cudaDeviceSynchronize());
	
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, T_p, N, &minus_lmbd, d_A + NN * k, N, d_xerr, N, &one, d_dx + N * (k + 1), N);
	}
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// => dx = dx / T
	dx_div_T << <GRID_SIZE_TN, BLOCK_SIZE >> > (d_dx, T, N);
}

// ->->->->-> Calculate the autoregressive coefficients (d_A: dim is NxNxp) using the classical OLS method <-<-<-<-<- \\

// Z = [X(:,2:T-p+1); X(:,3:T-p+2); ...; X(:,p+1:T)] and Y = X(:,1:T-p)
__global__ void fill_Z_block(double *d_Z, const double *d_x, int T, int N, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int pN = p * N;
	
	if (i < pN * (T - p)) {
		int t = i / pN;
		int ind = i % pN;
		int k = ind / N;
		int irow = ind % N;
		d_Z[i] = d_x[(t + k + 1) * N + irow];
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

// Main function which call all the kernel functions defined above: solve B = YZt/ZZt, where B = [A(1), A(2), ..., A(p)]
inline void calculate_A_OLS(cublasHandle_t handle_cublas, cusolverDnHandle_t handle_Dncusolver, \
	double *d_A, const double *d_x, double *d_Z, double *d_ZZt, double *d_ZXt, int * d_Ipiv, int *d_info,
	int pivot, int T, int M, int N, int p, double l2a) {
		
	const double one = 1.0;
	const double zero = 0.0;
	int pN = p * N;
	int T_p = T - p;
	
	// Z = [X(:,2:T-p+1); X(:,3:T-p+2); ...; X(:,p+1:T)]
	int GRID_SIZE_pN_T_p = (uint32_T)ceil((pN * (T - p)) / (BLOCK_SIZE + 0.0f));
	fill_Z_block << <GRID_SIZE_pN_T_p, BLOCK_SIZE >> > (d_Z, d_x, T, N, p);
	
	// -> Calculate ZZt
	CUBLAS_CALL(cublasDsyrk(handle_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, pN, T_p, &one, d_Z, pN, &zero, d_ZZt, pN));
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// fill up upper triangular part of d_ZZt
	int GRID_SIZE_pN_lowtri = (uint32_T)ceil((pN * (pN - 1) / 2) / (BLOCK_SIZE + 0.0));
	fill_upper_triangular << <GRID_SIZE_pN_lowtri, BLOCK_SIZE >> > (d_ZZt, pN);
	
	/*
	double h_tmp[4];
	CUDA_CALL(cudaMemcpy(h_tmp, d_ZZt, pN * pN * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\nZZt:\n");
	for (int i = 0; i < pN; i++) {
		for (int t = 0; t < pN; t++) {
			printf("%9.4f", h_tmp[t * N + i]);
		}
		printf("\n");
	}
	*/
	
	// add l2a to the diagonal of ZZt
	if (l2a != 0) {
		int GRID_SIZE_pN = (uint32_T)ceil(pN / (BLOCK_SIZE + 0.0));
		add_diagonal_lmbd << <GRID_SIZE_pN, BLOCK_SIZE >> > (d_ZZt, l2a, pN);
	}
	
	// -> Calculate ZXt = Z * Xt where Xt = X(:,1:T-p)'
	cublasDgemm(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, pN, N, T_p, &one, d_Z, pN, d_x, N, &zero, d_ZXt, pN);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	/*
	CUDA_CALL(cudaMemcpy(h_tmp, d_ZXt, pN * N * sizeof(double), cudaMemcpyDeviceToHost));
	printf("\n\nZXt:\n");
	for (int i = 0; i < pN; i++) {
		for (int t = 0; t < N; t++) {
			printf("%9.4f", h_tmp[t * N + i]);
		}
		printf("\n");
	}
	*/
	
	// -> Solve B such as ZZt * Bt = ZXt
	int lwork = 0; 			// size of workspace for getrf
    double *d_work = NULL; 	// device workspace for getrf
	
	// => step 1: query working space for getrf
    CUSOLVER_CALL(cusolverDnDgetrf_bufferSize(handle_Dncusolver, pN, pN, d_ZZt, pN, &lwork));
	
	CUDA_CALL(cudaDeviceSynchronize());
	
    CUDA_CALL(cudaMalloc((void**)&d_work, sizeof(double)*lwork));
	
	// => step 2: LU factorization
    if (pivot) {
        CUSOLVER_CALL(cusolverDnDgetrf(handle_Dncusolver, pN, pN, d_ZZt, pN, d_work, d_Ipiv, d_info));
    }
    else {
        CUSOLVER_CALL(cusolverDnDgetrf(handle_Dncusolver, pN, pN, d_ZZt, pN, d_work, NULL, d_info));
    }
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// => Step 3: calculate Bt (it is written as output in the same input right-hand side matrix ZYt)
	if (pivot) {
        CUSOLVER_CALL(cusolverDnDgetrs(handle_Dncusolver, CUBLAS_OP_N, pN, N, d_ZZt, pN, d_Ipiv, d_ZXt, pN, d_info));
    }
    else {
        CUSOLVER_CALL(cusolverDnDgetrs(handle_Dncusolver, CUBLAS_OP_N, pN, N, d_ZZt, pN, NULL, d_ZXt, pN, d_info));
    }
	
	// The cuSolver library functions prefer to keep asynchronous execution as much as possible.
	// Developers can always use the cudaDeviceSynchronize() function to ensure that the
	// execution of a particular cuSolver library routine has completed.
	//
	// It seems that between two CUSOLVER function is OK not to synch, like between the steps 2 and 3
	// above, but to play safe we synch here before calling my own (transpose) kernel
    CUDA_CALL(cudaDeviceSynchronize());
	
	// transpose d_ZXt to get B = [A(1), A(2), ..., A(p)]
	int GRID_SIZE_pNN = (uint32_T)ceil((pN * N) / (BLOCK_SIZE + 0.0));
	transpose_matrix << <GRID_SIZE_pNN, BLOCK_SIZE >> > (d_A, d_ZXt, pN, N);
	
	// -> Free resources
	cudaFree(d_work);	
}

// ->->->->-> Solve the system A*X = B, where A and B are matrices, by LU-based left division (X = A\B) <-<-<-<-<- \\

// Bt = transf(B), where B = [A(1), ..., A(p)] and Bt = [A(1)', ..., A(p)']
// A(k) dimension is N x N, for k = 1, ..., p
__global__ void transpose_A_block(double *d_At_block, const double *d_A, const int N, const int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int NN = N * N;
	
	if (i < p * NN) {
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
	int nnz, int nnz_indloss, int T, int p, int N, double lmbd, double l2x) {
		
	const double one = 1.0;
	const double zero = 0.0;
	const int Ncolbm = 2 * p + 1;
	const int TN = T * N;
	const int pN = p * N;
	const int NN = N * N;
	
	// -> Calculate the cross-products before building the big sparse kronecker matrix
	// AtA = A'*A
	cublasStatus_t stat_cublas = cublasDsyrk(handle_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, pN, N, &one, d_A, N, &zero, d_AtA, pN);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	//CUDA_CALL(cudaDeviceSynchronize());
	if (stat_cublas != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS matrix multiplication failed\n");
    }
	
	// cublasDsyrk only filled the lower triangular part, so we fill up the other part
	int GRID_SIZE_pN = (uint32_T)ceil((pN * (pN - 1) / 2) / (BLOCK_SIZE + 0.0));
	fill_upper_triangular << <GRID_SIZE_pN, BLOCK_SIZE >> > (d_AtA, pN);
	
	// -> Transpose the autoregressive matrices, separately per block
	int GRID_SIZE_A_block = (uint32_T)ceil((N * pN) / (BLOCK_SIZE + 0.0));
	// Bt = transf(B), where B = [A(1), ..., A(p)] and Bt = [A(1)'; ...; A(p)']
	transpose_A_block << <GRID_SIZE_A_block, BLOCK_SIZE >> > (d_At_block, d_A, N, p);
	
	// -> Create the big sparse kronecker-sums matrix using the equivalent banded matrix representation
	
	// => First, initialize the banded matrix with the identity (add to the central block)
	//CUDA_CALL(cudaMemset(d_bandMat, 0, T * Ncolbm * NN * sizeof(double)));
	int GRID_SIZE_allbanded = (uint32_T)ceil((T * Ncolbm * NN) / (BLOCK_SIZE + 0.0));
	reset_to_zero << <GRID_SIZE_allbanded, BLOCK_SIZE >> > (d_bandMat, T * Ncolbm * NN);
	
	int GRID_SIZE_TN = (uint32_T)ceil(TN / (BLOCK_SIZE + 0.0));
	insert_identity << <GRID_SIZE_TN, BLOCK_SIZE >> > (d_bandMat, TN, N, p);
	
	// => Second, substract the linear terms to the upper and lower diagonals
	int nnzb = T - p;
	int GRID_SIZE_colblock = (uint32_T)ceil((nnzb * NN) / (BLOCK_SIZE + 0.0));
	for (int k = 1; k <= p; k++) {
		// insert -A into the banded matrix structure (upper triangular part)
		insert_minus_block << <GRID_SIZE_colblock, BLOCK_SIZE >> > (d_bandMat, d_A + NN * (k - 1), TN, N, p + k, 0, nnzb);
		// insert -A' into the banded matrix structure (lower triangular part)	
		insert_minus_block << <GRID_SIZE_colblock, BLOCK_SIZE >> > (d_bandMat, d_At_block + NN * (k - 1), TN, N, p - k, k, nnzb);
	}
	
	// => Third, add the quadratic terms
	int GRID_SIZE_NN = (uint32_T)ceil(NN / (BLOCK_SIZE + 0.0));
	for (int k = 1; k <= p; k++) {
		for (int l = 1; l <= p; l++) {
			// => fill up the cross-product block
			copy_block << <GRID_SIZE_NN, BLOCK_SIZE >> > (d_AtA_block, d_AtA, pN, N, k - 1, l - 1);	
			// => add it to the banded matrix
			add_block << <GRID_SIZE_colblock, BLOCK_SIZE >> > (d_bandMat, d_AtA_block, TN, N, p + k - l, MAX(0,l-k) + MIN(k,l), nnzb);
		}
	}
	
	// => Fourth, multiply the values in the banded matrix by lmbd
	multiply_banded_values_by_lambda << <GRID_SIZE_allbanded, BLOCK_SIZE >> > (d_bandMat, lmbd, T * Ncolbm * NN);
	
	// => Fifth, add kron(diag(loss.^2),BtB) to the diagonal in banded matrix format
	// Because loss(t) is either 0 or 1 it is enough to know the index of nonzeros values in the loss vector
	int GRID_SIZE_indcolblock = (uint32_T)ceil((nnz_indloss * NN) / (BLOCK_SIZE + 0.0));
	add_block_BtB_diag << <GRID_SIZE_indcolblock, BLOCK_SIZE >> > (d_bandMat, d_BtB, d_indloss, TN, N, p, nnz_indloss);
	
	// => Sixth, add l2x to the main diagonal, which also guarantee that the matrix is definite positive
	add_scalar_identity << <GRID_SIZE_TN, BLOCK_SIZE >> > (d_bandMat, TN, N, p, l2x);
	
	// => Seventh, reading Values from banded matrix to sparse (CSR) format
	int GRID_SIZE_nnz = (uint32_T)ceil(nnz / (BLOCK_SIZE + 0.0));
	bmr2sparse << <GRID_SIZE_nnz, BLOCK_SIZE >> > (d_csrVal, d_bandMat, nnz, d_bmrInd);
	
	CUDA_CALL(cudaDeviceSynchronize());
	
	// -> Finally, copy to Matlab sparse array to do the division in their engine UNTIL THERE IS BETTER cusolver/cusparse division function
	CUDA_CALL(cudaMemcpy(mxGetPr(rhs[0]), d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost));
	mexCallMATLAB(1, &lhs, 2, rhs, "mldivide");
	
	/*
	MATFile *pmat = matOpen("sparsematrix.mat", "w");
	if (pmat == NULL) {
		mexErrMsgTxt("Error creating file.");
	}
	
	matPutVariable(pmat, "A", rhs[0]);
	matPutVariable(pmat, "Yvec", rhs[1]);
	matPutVariable(pmat, "X", lhs);
	
	printf("%d\n", __LINE__); mexErrMsgTxt("Stop here");
	*/
	
	// copy the solution to device
	CUDA_CALL(cudaMemcpy(d_x, mxGetPr(lhs), TN * sizeof(double), cudaMemcpyHostToDevice));
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

// create_BtYvec
// only fill up those entries for loss[t] != 0, where nnz is the nnz in this array.
__global__ void create_BtYvec(double *d_BtYvec, const double *d_BtY, const int *d_indloss, int N, int nnz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < nnz * N) {
		int t = d_indloss[i / N];
		int ind = t * N + (i % N);
		d_BtYvec[ind] = d_BtY[ind];
	}
}

// next solution in GD's direction
// x = x_old - alpha*dx;
__global__ void move_gd_step(double *d_x, double *d_x_old, double *d_dx, int T, int N, double alpha) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < T * N) {
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
	double *d_Z = NULL, *d_ZZt = NULL, *d_ZXt = NULL;
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
	const int Nstep = 7;
	double alpha_vec_GD[Nstep] = { 1, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6 };
	
	// -> Dimension variables
	// mwSize M, N, p, T, T_p, NN, NT, MT, pN, Ncolbm;
	int p, T_p, pN, Ncolbm;
	
	// -> Number of nonzero elements in big sparse (Kronecker's sum) matrix
	int nnz;
	
	// -> Check input
    if (nrhs != 10) mexErrMsgTxt("Nine inputs required: [xe, Ae, iter] = cmex_HGDALS_GPU(y, x, B, A, lmbd, l2x, l2a, Niter, tol, loss)");
	
    // dimensions	
	const int M = mxGetM(prhs_y);
    const int T = mxGetN(prhs_y);
    const int N = mxGetM(prhs_x);
	const int NN = N * N;
	const int NT = N * T;
	const int MT = M * T;
    const mwSize *A_dims = mxGetDimensions(prhs_A);
	p = A_dims[2];
	
	// pointers to input data
	if ((mxGetNumberOfDimensions(prhs_y) != 2) || (mxGetClassID(prhs_y) != mxDOUBLE_CLASS))
        mexErrMsgTxt("The 1st input (y) is a 2D matrix. Dimension MxT (double).");
	if ((mxGetNumberOfDimensions(prhs_x) != 2) || 
		(mxGetClassID(prhs_x) != mxDOUBLE_CLASS) || (mxGetN(prhs_x) != T))
        mexErrMsgTxt("The 2nd input (xe) is a 2D matrix. Dimension NxT (double).");
	if ((mxGetNumberOfDimensions(prhs_B) != 2) || (mxGetClassID(prhs_B) != mxDOUBLE_CLASS) ||
		(mxGetM(prhs_B) != M) || (mxGetN(prhs_B) != N))
        mexErrMsgTxt("The 3rd input (B) is a 2D matrix. Dimension MxN (double).");
	if (mxGetNumberOfDimensions(prhs_A) == 2) {
		p = 1;
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
	T_p = T - p;
	pN = p * N;
	Ncolbm = 2 * p + 1;
	
	// -> Check output
	if ((nlhs != 2) && (nlhs != 3))
		mexErrMsgTxt("Only two or three outputs are allowed.");
	
	// Print input-related information
	printf("T = %d, M = %d, N = %d, p = %d, Ncolbm = %d, T-p = %d, l2x = %.6f, l2a = %.6f\n", T, M, N, p, Ncolbm, T_p, l2x, l2a);
    
    // -> Get the pointers in GPU
	CUDA_CALL(cudaMalloc((void**)&d_y, MT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_x, NT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_B, M * N * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_A, NN * p * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_loss, T * sizeof(double)));
	CUDA_CALL(cudaMemcpy(d_y, mxGetPr(prhs_y), MT * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_x, mxGetPr(prhs_x), NT * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_B, mxGetPr(prhs_B), M * N * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_A, mxGetPr(prhs_A), NN * p * sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_loss, mxGetPr(prhs_loss), T * sizeof(double), cudaMemcpyHostToDevice));
	
	// -> Get some of the input pointers in CPU
    h_loss = (double*)mxGetPr(prhs_loss);

	// -> Allocate memory for some of the pointers in CPU
    h_A = (double*)malloc(NN * p * sizeof(double));
    h_xerr = (double*)malloc(NT * sizeof(double));
    h_yerr = (double*)malloc(MT * sizeof(double));
    
	// -> Auxiliar variables allocated on GPU
	CUDA_CALL(cudaMalloc((void**)&d_x_old, NT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_A_old, NN * p * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_xerr, NT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_xerr_old, NT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_dx, NT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_yerr, MT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_yerr_old, MT * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_B_x, MT * sizeof(double)));
	
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
	int GRID_SIZE_NT = (uint32_T)ceil((N * T) / (BLOCK_SIZE + 0.0f));
	int GRID_SIZE_NThalf = (uint32_T)ceil((N * (T / 2)) / (BLOCK_SIZE + 0.0f));
	fliplr << <GRID_SIZE_NThalf, BLOCK_SIZE >> > (d_x, N, T);
	int GRID_SIZE_MThalf = (uint32_T)ceil((M * (T / 2)) / (BLOCK_SIZE + 0.0f));
	fliplr << <GRID_SIZE_MThalf, BLOCK_SIZE >> > (d_y, M, T);
	int GRID_SIZE_Thalf = (uint32_T)ceil((T / 2) / (BLOCK_SIZE + 0.0f));
	fliplr << <GRID_SIZE_Thalf, BLOCK_SIZE >> > (d_loss, 1, T);
	
	// -> Create Matlab rhs and lhs pointers for division
	mxArray *rhs[2];
	rhs[1] = mxCreateDoubleMatrix(NT, 1, mxREAL);
	mxArray *lhs = mxCreateDoubleMatrix(NT, 1, mxREAL);
	
	// => Running indices through the sparse diagonal space to map sparse indices.
	// coo: CUSPARSE coordinate format.
	// bmr: banded matrix format
	nnz = (T * Ncolbm - p * (p + 1)) * NN;
	
	rhs[0] = mxCreateSparse(NT, NT, nnz, mxREAL);
	mwIndex *irowptr = mxGetIr(rhs[0]); // length is nnz
    mwIndex *icolptr = mxGetJc(rhs[0]); // length is NT + 1
	CUDA_CALL(cudaMalloc((void**)&d_csrVal, nnz * sizeof(double))); // where the values of the sparse matrix are going to reside in device
	
	// this may be confusing but no if it is taken into account that Matlab's sparse indices run in a row-major order,
	// compressed (in terms of csr's CUSPARSE format) by columns, whereas CUSPARSE indices run in a column-major order,
	// compressed by rows.
	int *h_cooRowInd = (int*)malloc(nnz * sizeof(int));
	int *h_bmrInd = (int*)malloc(nnz * sizeof(int));
	int ind = 0;
	for (int irb = 0; irb < T; irb++) {
		for (int ir = 0; ir < N; ir++) {
			int irow = irb * N + ir;
			for (int icb = MAX(0,irb-p); icb < MIN(T,irb+p+1); icb++) {
				for (int ic = 0; ic < N; ic++) {
					if (ind == nnz) {
						printf("Line %d: ", __LINE__);
						mexErrMsgTxt("Unexpected index access violation.");
					}
					h_cooRowInd[ind] = irow;
					irowptr[ind] = icb * N + ic;
					int icol_bmr = (p + icb - irb) * N + ic;
					h_bmrInd[ind] = icol_bmr * NT + irow;
					ind++;
				}
			}
		}
	}
	if (ind != nnz) mexErrMsgTxt("The total number of accounted indices must be equals to nnz.");
	
	// => converting RowInd (COO) to RowPtr (CSR) format, then copy the csrRowPtr to corresponding Matlab's sparse pointer.
	int *d_cooRowInd, *d_csrRowPtr;
	CUDA_CALL(cudaMalloc((void**)&d_cooRowInd, nnz * sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_cooRowInd, h_cooRowInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc((void**)&d_csrRowPtr, (NT + 1) * sizeof(int)));
	CUSPARSE_CALL(cusparseXcoo2csr(handle_cusparse, d_cooRowInd, nnz, NT, d_csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
	int *h_csrRowPtr = (int*)malloc((NT + 1) * sizeof(int));
	CUDA_CALL(cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (NT + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i <= NT; i++)
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
	CUDA_CALL(cudaMalloc((void**)&d_AtA, pN * pN * sizeof(double)));	
	CUDA_CALL(cudaMalloc((void**)&d_AtA_block, NN * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_At_block, N * pN * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_bandMat, T * Ncolbm * NN * sizeof(double)));
	
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
	CUDA_CALL(cudaMalloc((void**)&d_BtY, NT * sizeof(double)));
	cublasDgemm(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, T, M, &one, d_B, M, d_y, M, &zero, d_BtY, N);
	
	// BtYvec = vec(BtY * diag(loss))
	CUDA_CALL(cudaMalloc((void**)&d_BtYvec, NT * sizeof(double)));
	//CUDA_CALL(cudaMemset(d_BtYvec, 0, NT * sizeof(double)));
	reset_to_zero << <GRID_SIZE_NT, BLOCK_SIZE >> > (d_BtYvec, NT);
	int GRID_SIZE_Nnnz = (uint32_T)ceil((nnz_indloss * N) / (BLOCK_SIZE + 0.0f));
	create_BtYvec << <GRID_SIZE_Nnnz, BLOCK_SIZE >> > (d_BtYvec, d_BtY, d_indloss, N, nnz_indloss);
	
	// -> Set rhs[1] to copy of BtYvec
	CUDA_CALL(cudaMemcpy(mxGetPr(rhs[1]), d_BtYvec, NT * sizeof(double), cudaMemcpyDeviceToHost));
	
	// free memory that will not be used after this
	cudaFree(d_BtY);
	
	// => d_BtB
	CUDA_CALL(cudaMalloc((void**)&d_BtB, NN * sizeof(double)));
	cublasStatus_t stat_cublas = cublasDsyrk(handle_cublas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, N, M, &one, d_B, M, &zero, d_BtB, N);
	if (stat_cublas != CUBLAS_STATUS_SUCCESS) {
        mexErrMsgTxt("CUBLAS matrix multiplication failed\n");
    }
	
	// cublasDsyrk only filled the lower triangular part, so we fill up the other part
	int GRID_SIZE_NNtri = (uint32_T)ceil((N * (N - 1) / 2) / (BLOCK_SIZE + 0.0));
	fill_upper_triangular << <GRID_SIZE_NNtri, BLOCK_SIZE >> > (d_BtB, N);
	
	// -> Separate memory for d_Z, d_ZZt, d_ZXt, d_Ipiv, and d_info OUTSIDE of the main loop
	const int pivot = 1;
	CUDA_CALL(cudaMalloc((void**)&d_Z, pN * T_p * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_ZZt, pN * pN * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_ZXt, pN * N * sizeof(double)));
	CUDA_CALL(cudaMalloc((void**)&d_Ipiv, pN * sizeof(int)));
	CUDA_CALL(cudaMalloc((void**)&d_info, sizeof(int)));
	
	// -> Main algorithm
    int iter, italpha, NGD = 1;
	double TALS, TGD, FGain_ALS, FGain_GD, F, Fprev, alpha;
    
	//double h_tmp[14];
	Fprev = eval_cost(handle_cublas, h_A, h_xerr, h_yerr, h_loss, d_xerr, d_yerr, d_y, d_B, d_x, d_A, d_loss, d_B_x, lmbd, l2x, l2a, M, N, p, T);
	printf("Initial value: F = %.4f\n", Fprev);
	
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
			d_BtYvec, d_BtB, d_indloss, d_A, d_AtA, d_At_block, d_AtA_block, nnz, nnz_indloss, T, p, N, lmbd, l2x);
			
		// >> Update A
		calculate_A_OLS(handle_cublas, handle_Dncusolver, d_A, d_x, d_Z, d_ZZt, d_ZXt, d_Ipiv, d_info, pivot, T, M, N, p, l2a);
		
		/*
		MATFile *pmat = matOpen("sparsematrix.mat", "w");
		if (pmat == NULL) {
			mexErrMsgTxt("Error creating file.");
		}
		mxArray *mat_x = mxCreateDoubleMatrix(NT, 1, mxREAL);
		mxArray *mat_A = mxCreateDoubleMatrix(N, pN, mxREAL);
		CUDA_CALL(cudaMemcpy(mxGetPr(mat_x), d_x, NT * sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(mxGetPr(mat_A), d_A, p * NN * sizeof(double), cudaMemcpyDeviceToHost));
		matPutVariable(pmat, "A", mat_A);
		matPutVariable(pmat, "X", mat_x);
		if (matClose(pmat) != 0) {
			mexErrMsgTxt("Error closing file.");
		}
		printf("%d\n", __LINE__); mexErrMsgTxt("Stop here");
		*/
		
		// >> Evaluate cost function after ALS
		// Be AWARE that this function MODIFIES xerr and yerr, WHICH ARE USED in function "calculate_x_GD" below.
		F = eval_cost(handle_cublas, h_A, h_xerr, h_yerr, h_loss, d_xerr, d_yerr, d_y, d_B, d_x, d_A, d_loss, d_B_x, lmbd, l2x, l2a, M, N, p, T);
		printf("ALS: F = %.4f\n", F);
		
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
			CUDA_CALL(cudaMemcpy(d_x, d_x_old, N * T * sizeof(double), cudaMemcpyDeviceToDevice));
			CUDA_CALL(cudaMemcpy(d_A, d_A_old, NN * p * sizeof(double), cudaMemcpyDeviceToDevice));
			mexWarnMsgIdAndTxt("ALS:convergence", "The ALS algorithm should have converged before, in the GD step, maybe the X's OLS operation failed.");
            break;
		}
	
		// => Calculate number of iterations for gradient descent
		if (iter > 0) {
			//printf("(%.6f * %.6f) / (%.6f * %.6f) = %.6f\n", FGain_GD, TALS, FGain_ALS, TGD, (FGain_GD * TALS) / (FGain_ALS * TGD));
			NGD = (int)ceil((FGain_GD * TALS) / (FGain_ALS * TGD));
			if (NGD < 1) NGD = 1;
			if (NGD > 10000) NGD = 10000;
		}
	
        // => Alternatively, use gradient descent
		CUDA_CALL(cudaDeviceSynchronize());
		begin = std::chrono::high_resolution_clock::now();
		
		// Best solution so far
		double Fbest = F;
		CUDA_CALL(cudaMemcpy(d_x_old, d_x, NT * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_A_old, d_A, NN * p * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_xerr_old, d_xerr, NT * sizeof(double), cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy(d_yerr_old, d_yerr, MT * sizeof(double), cudaMemcpyDeviceToDevice));
		
		int itgd;
		for (itgd = 0; itgd < NGD; itgd++) {
			// >> Calculate gradient descent direction for x
			calculate_x_GD(handle_cublas, d_dx, d_B, d_x, d_A, d_xerr, d_yerr, T, M, N, p, lmbd);
			
			// >> Choose the highest successful step
			
			// optimise according to different step sizes
			for (italpha = 0; italpha < Nstep; italpha++) {
				alpha = alpha_vec_GD[italpha];
				
				// >>> Update x
				// x = x_old - alpha*dx;
				move_gd_step << <GRID_SIZE_NT, BLOCK_SIZE >> > (d_x, d_x_old, d_dx, T, N, alpha);
				
				// >>> Update A
				calculate_A_OLS(handle_cublas, handle_Dncusolver, d_A, d_x, d_Z, d_ZZt, d_ZXt, d_Ipiv, d_info, pivot, T, M, N, p, l2a);
				
				// >>> Evaluate cost function after GD
				// Be AWARE that this function MODIFIES xerr and yerr, WHICH ARE USED in function "calculate_x_GD" above.
				// However, the current implementation is consistent as, after the break below, for the next "calculate_x_GD"
				// the updated values for xerr and yerr will be used corretly.
				F = eval_cost(handle_cublas, h_A, h_xerr, h_yerr, h_loss, d_xerr, d_yerr, \
					d_y, d_B, d_x, d_A, d_loss, d_B_x, lmbd, l2x, l2a, M, N, p, T);
					
				// >>> Evaluate break condition
				if (F < Fbest) break;
			}
			
			// clone x if solution improved: x_old = x;
			if (F < Fbest) {
				Fbest = F;
				CUDA_CALL(cudaMemcpy(d_x_old, d_x, NT * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_A_old, d_A, NN * p * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_xerr_old, d_xerr, NT * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_yerr_old, d_yerr, MT * sizeof(double), cudaMemcpyDeviceToDevice));
			}
			else {
				// Restore old values
				CUDA_CALL(cudaMemcpy(d_x, d_x_old, NT * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_A, d_A_old, NN * p * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_xerr, d_xerr_old, NT * sizeof(double), cudaMemcpyDeviceToDevice));
				CUDA_CALL(cudaMemcpy(d_yerr, d_yerr_old, MT * sizeof(double), cudaMemcpyDeviceToDevice));
			}
			
			// >> Evaluate stop condition
			if (italpha == Nstep - 1) break;
		}
		
		// => Evaluate gain on GD algorithm
		int fail_GD = -1;
		if (F >= Fprev) {
			fail_GD = 1;
			FGain_GD = 0;
			// Restore old values
			//CUDA_CALL(cudaMemcpy(d_x, d_x_old, N * T * sizeof(double), cudaMemcpyDeviceToDevice));
			//CUDA_CALL(cudaMemcpy(d_A, d_A_old, NN * p * sizeof(double), cudaMemcpyDeviceToDevice));
			//CUDA_CALL(cudaMemcpy(d_xerr, d_xerr_old, NT * sizeof(double), cudaMemcpyDeviceToDevice));
			//CUDA_CALL(cudaMemcpy(d_yerr, d_yerr_old, MT * sizeof(double), cudaMemcpyDeviceToDevice));
		}
		else {
			fail_GD = 0;
			FGain_GD = Fprev - F; // GD either improves the solution, therefore FGain_GD > 0, or reached convergence 
			Fprev = F;
		}
		printf("GD (%d of %d loops): alpha = %.8f, F = %.4f\n", itgd, NGD, alpha, F);
		
		// => Calculate GD algorithm's elapsed time
		CUDA_CALL(cudaDeviceSynchronize());
		end = std::chrono::high_resolution_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
		TGD = elapsed.count() * 1e-9;
		printf("Elapsed time (GD) is %.3f seconds.\n", TGD);
		
		// -> Calculate the average time for one iteration unit
		TGD = TGD / NGD;
		
		// -> Evaluate stop condition
		if ((fail_GD == 0) && (italpha == Nstep - 1)) break;
	}
	
	// -> Time reversal of x
	fliplr << <GRID_SIZE_NThalf, BLOCK_SIZE >> > (d_x, N, T);
	
	// -> Wrap the result up as a MATLAB gpuArray for return
	plhs[0] = mxCreateDoubleMatrix(N, T, mxREAL);
	CUDA_CALL(cudaMemcpy(mxGetPr(plhs[0]), d_x, NT * sizeof(double), cudaMemcpyDeviceToHost));
	mwSize dim[3] = { N, N, p };
	plhs[1] = mxCreateNumericArray(3, dim, mxDOUBLE_CLASS, mxREAL);
	CUDA_CALL(cudaMemcpy(mxGetData(plhs[1]), d_A, NN * p * sizeof(double), cudaMemcpyDeviceToHost));
	
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