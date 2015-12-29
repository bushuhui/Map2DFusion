#ifndef UTILGPU_CUH
#define UTILGPU_CUH

#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
#include <stdio.h>

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

/* We define these the same for all machines.
   Changes from this to the outside world should be done in `_exit'.  */
#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%d\" from file <%s>, line %i.\n",
                err, err, file, line);
        exit(EXIT_FAILURE);
    }
}
#endif

template <typename PixelType>
struct CudaImage
{
    inline CudaImage(int rowNum,int colNum,PixelType* dataPtr=NULL)
        :rows(rowNum),cols(colNum),data(dataPtr),fresh(true),cpCount(NULL)
    {
        if(!data)
        {
            checkCudaErrors(cudaMalloc(&data,sizeof(PixelType)*cols*rows));
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
                if(data) checkCudaErrors(cudaFree(data));
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
