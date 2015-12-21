#ifndef UTILGPU_CUH
#define UTILGPU_CUH

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>

template <typename PixelType>
struct CudaImage
{
    inline CudaImage(int rowNum,int colNum,PixelType* dataPtr=NULL)
        :rows(rowNum),cols(colNum),data(dataPtr),fresh(true),cpCount(NULL)
    {
        if(!data)
        {
            cudaMalloc(&data,sizeof(PixelType)*cols*rows);
            cpCount=new int(1);
        }
    }

    inline CudaImage(const CudaImage& img)
        :rows(img.rows),cols(img.cols),cpCount(img.cpCount),
          fresh(img.fresh),data(img.data)
    {
        if(cpCount)
            (*cpCount)++;
    }

    inline ~CudaImage(){
        if(cpCount)
        {
            (*cpCount)--;
            if(!cpCount) {
                if(data) cudaFree(data);
                delete cpCount;
            }
        }
    }

    int rows,cols,*cpCount;
    bool fresh;
    PixelType *data;
};

__host__ __device__ __forceinline__ int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

template <class T>
class operate {
public:
    static bool warpPerspectiveCaller(int in_rows,int in_cols,T* in_data,
                               int out_rows,int out_cols,T* out_data,
                               float* inv,T defVar);

    static cudaError_t addWithCuda(T *c, const T *a, const T *b, unsigned int size);
};

bool warpPerspective_uchar1(int in_rows,int in_cols,uchar1* in_data,
                            int out_rows,int out_cols,uchar1* out_data,
                            float* inv,uchar1 defVar);
//bool warpPerspective_uchar2();
bool warpPerspective_uchar3(int in_rows,int in_cols,uchar3* in_data,
                            int out_rows,int out_cols,uchar3* out_data,
                            float* inv,uchar3 defVar);

bool warpPerspective_uchar4(int in_rows,int in_cols,uchar4* in_data,
                            int out_rows,int out_cols,uchar4* out_data,
                            float* inv,uchar4 defVar);

bool renderFrameCaller(CudaImage<uchar3>& rgbIn,CudaImage<uchar4>& ele,
                       float* inv,int centerX,int centerY);


bool renderFramesCaller(CudaImage<uchar3>& rgbIn,int out_rows,int out_cols,
                        uchar4** out_datas,bool* freshs,
                       float* invs,int centerX,int centerY,int eleNum);
#endif // UTILGPU_CUH
