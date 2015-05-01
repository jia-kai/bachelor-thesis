/*
 * $File: affine3d_kern.cu
 * $Date: Thu Apr 30 22:55:17 2015 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */


__device__ int min(int a, int b) {
    return a < b ? a : b;
}

__device__ int max(int a, int b) {
    return a > b ? a : b;
}

template<int axis>
__device__ void batched_affine3d_impl(
        float *dest, const float *src, const float *rev_affine_mat,
        int tot_size, int odim2, int odim1, int odim0,
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

    if (offset >= tot_size)
        return;
    dest += offset;
    int batch;
    float x2_, x1_, x0_;
    x0_ = offset % odim0; offset /= odim0;
    x1_ = offset % odim1; offset /= odim1;
    x2_ = offset % odim2; offset /= odim2;
    batch = offset;

    float const * const m = rev_affine_mat + batch * 16;
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
        int x0i = min(max(int(x0f), 0), idim0 - 1),
            x1i = min(max(int(x1f), 0), idim1 - 1),
            x2i = min(max(int(x2f), 0), idim2 - 1),
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

/*!
 * \brief apply 3D affine transform to a batch of 3D images, with bilinear
 *      interpolation; dest and src are of the same size
 * \param dest nr_batch * odim2 * odim1 * odim0 image array
 * \param src nr_batch * idim2 * idim1 * idim0 image array
 * \param rev_affine_mat matrix to transform coordinates on dest onto src;
 *      shape: nr_batch * 3 * 4 matrix to describe the transformation for each
 *      image
 * \param tot_size total size of output array
 */
extern "C" __global__ void batched_affine3d(
            float *dest, const float *src, const float *rev_affine_mat,
            int tot_size, int odim2, int odim1, int odim0,
            int idim2, int idim1, int idim0) {
    void (*fptr)(float*, const float*, const float*, int, int, int, int, int,
            int, int);
    if (odim2 <= odim1 && odim2 <= odim0)
        fptr = &batched_affine3d_impl<2>;
    else if (odim1 <= odim2 && odim1 <= odim0)
        fptr = &batched_affine3d_impl<1>;
    else
        fptr = &batched_affine3d_impl<0>;

    return fptr(dest, src, rev_affine_mat,
            tot_size, odim2, odim1, odim0,
            idim2, idim1, idim0);
}

// vim: syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
