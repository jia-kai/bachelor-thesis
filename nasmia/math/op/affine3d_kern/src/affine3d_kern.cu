/*
 * $File: affine3d_kern.cu
 * $Date: Sun May 03 02:43:53 2015 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

static const int NR_THREAD_PER_BLOCK = 128;

namespace {

__host__ __device__ int min(int a, int b) {
    return a < b ? a : b;
}

__host__ __device__ int max(int a, int b) {
    return a > b ? a : b;
}

template<int axis>
__global__ void batched_affine3d_impl(
        float *dest, const float *src, const float *inv_affine_mat,
        int tot_osize, int odim2, int odim1, int odim0,
        int idim2, int idim1, int idim0) {

    int idx_tot = blockIdx.x * blockDim.x + threadIdx.x,
        stride_axis = 1, dim_axis = odim0;

    if (axis == 1) {
        dim_axis = odim1;
        stride_axis = odim0;
    }

    if (axis == 2) {
        dim_axis = odim2;
        stride_axis = odim0 * odim1;
    }

    int offset = idx_tot / stride_axis * (stride_axis * dim_axis) +
                idx_tot % stride_axis;

    if (offset >= tot_osize)
        return;
    dest += offset;
    int batch;
    float x2_, x1_, x0_;
    x0_ = offset % odim0; offset /= odim0;
    x1_ = offset % odim1; offset /= odim1;
    x2_ = offset % odim2; offset /= odim2;
    batch = offset;

    float const * const m = inv_affine_mat + batch * 12;
    float
        dx0 = m[axis], dx1 = m[4 + axis], dx2 = m[8 + axis],
        x0 = m[0] * x0_ + m[1] * x1_ + m[2] * x2_ + m[3],
        x1 = m[4] * x0_ + m[5] * x1_ + m[6] * x2_ + m[7],
        x2 = m[8] * x0_ + m[9] * x1_ + m[10] * x2_ + m[11];

    int istride1 = idim0,
        istride2 = idim1 * idim0;
    src += batch * (istride2 * idim2);
    for (int i = 0; i < dim_axis; i ++) {
        float x0f = floor(x0), x1f = floor(x1), x2f = floor(x2);
        int x0i = min(max(int(x0f), 0), idim0 - 2),
            x1i = min(max(int(x1f), 0), idim1 - 2),
            x2i = min(max(int(x2f), 0), idim2 - 2),
            idx = x2i * istride2 + x1i * istride1 + x0i;
        x0f = x0 - x0f; x1f = x1 - x1f; x2f = x2 - x2f;
        float itv00 = src[idx] * (1 - x0f) + src[idx + 1] * x0f,
              itv01 = src[idx + istride1] * (1 - x0f) +
                      src[idx + istride1 + 1] * x0f,
              itv10 = src[idx + istride2] * (1 - x0f) +
                      src[idx + istride2 + 1] * x0f,
              itv11 = src[idx + istride2 + istride1] * (1 - x0f) +
                      src[idx + istride2 + istride1 + 1] * x0f,
              itv0 = itv00 * (1 - x1f) + itv01 * x1f,
              itv1 = itv10 * (1 - x1f) + itv11 * x1f,
              itv = itv0 * (1 - x2f) + itv1 * x2f;
        *dest = itv;
        dest += stride_axis;
        x0 += dx0; x1 += dx1; x2 += dx2;
    }
}



float* alloc_dev(size_t size)  {
    float *ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        size_t tot = 0, free = 0;
        int device = -1;
        cudaMemGetInfo(&free, &tot);
        cudaGetDevice(&device);
        const double MC = 1.0 / (1024 * 1024);
        fprintf(stderr, "cuda err %d: %s: failed to alloc "
                "%zu bytes (%.2fMiB) of memory, free=%.2fMiB tot=%.2fMiB "
                "device=%d\n",
                int(err), cudaGetErrorString(err), size,
                size * MC, free * MC, tot * MC, device);
        abort();
    }
    return ptr;
}

} // anonymous namespace

/*!
 * \brief apply 3D affine transform to a batch of 3D images, with bilinear
 *      interpolation; dest and src are of the same size
 * \param dest nr_batch * odim2 * odim1 * odim0 image array
 * \param src nr_batch * idim2 * idim1 * idim0 image array
 * \param inv_affine_mat matrix to transform coordinates on dest onto src;
 *      shape: nr_batch * 3 * 4 matrix to describe the transformation for each
 *      image
 */
extern "C" int batched_affine3d(
        float *dest, const float *src, const float *inv_affine_mat,
        int batch,
        int odim2, int odim1, int odim0,
        int idim2, int idim1, int idim0,
        int device) {

#define CUDA_CHKERR(call)  \
    do { \
        cudaError_t code = (call); \
        if (code != cudaSuccess) { \
            fprintf(stderr, "cuda err %d:\n  %s\n(call %s at %s:%s:%d)\n", \
                    int(code), cudaGetErrorString(code), # call, \
                    __FILE__, __func__, __LINE__); \
            return -1; \
        } \
    } while(0)

    static bool first_init = true;
    if (first_init) {
        if (cuInit(0) != CUDA_SUCCESS) {
            fprintf(stderr, "failed to cuInit\n");
            return -1;
        }
        CUdevice cudev;
        if (cuDeviceGet(&cudev, device) != CUDA_SUCCESS) {
            fprintf(stderr, "failed to get device\n");
            return -1;
        }
        CUcontext ctx;
        if (cuCtxCreate(&ctx, 0, cudev) != CUDA_SUCCESS) {
            fprintf(stderr, "failed to cuCtxCreate\n");
            return -1;
        }
        first_init = false;
    }
    CUDA_CHKERR(cudaSetDevice(device));

    void (*fptr)(float*, const float*, const float*, int, int, int, int, int,
            int, int);

    int axis_size;
    if (odim2 <= odim1 && odim2 <= odim0) {
        axis_size = odim2;
        fptr = &batched_affine3d_impl<2>;
    }
    else if (odim1 <= odim2 && odim1 <= odim0) {
        axis_size = odim1;
        fptr = &batched_affine3d_impl<1>;
    }
    else {
        axis_size = odim0;
        fptr = &batched_affine3d_impl<0>;
    }

    size_t nr_output_float = size_t(batch)*odim2*odim1*odim0,
           tot_out_size = nr_output_float*sizeof(float),
           tot_in_size = size_t(batch)*idim2*idim1*idim0*sizeof(float),
           tot_mat_size = size_t(batch)*12*sizeof(float);

    int computing_size = nr_output_float / axis_size,
        block_size = ::min(NR_THREAD_PER_BLOCK, computing_size),
        grid_size = (computing_size - 1) / block_size + 1;

    float *dev_dest = alloc_dev(tot_out_size),
          *dev_src = alloc_dev(tot_in_size),
          *dev_mat = alloc_dev(tot_mat_size);

    CUDA_CHKERR(cudaMemsetAsync(dev_dest, -1, tot_out_size)); // init as NaN
    CUDA_CHKERR(cudaMemcpyAsync(dev_src, src, tot_in_size,
                cudaMemcpyHostToDevice));
    CUDA_CHKERR(cudaMemcpyAsync(dev_mat, inv_affine_mat, tot_mat_size,
                cudaMemcpyHostToDevice));

    fptr<<<grid_size, block_size>>>(
            dev_dest, dev_src, dev_mat,
            nr_output_float,
            odim2, odim1, odim0,
            idim2, idim1, idim0);


    CUDA_CHKERR(cudaMemcpyAsync(dest, dev_dest, tot_out_size,
                cudaMemcpyDeviceToHost));
    CUDA_CHKERR(cudaDeviceSynchronize());
    CUDA_CHKERR(cudaFree(dev_dest));
    CUDA_CHKERR(cudaFree(dev_src));
    CUDA_CHKERR(cudaFree(dev_mat));
    return 0;
}

// vim: syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
