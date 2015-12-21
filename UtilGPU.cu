#include "UtilGPU.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <base/time/Global_Timer.h>


template <class T>
void __global__ addKernel1(T *c, const T *a, const T *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

template <class T>
cudaError_t operate<T>::addWithCuda(T *c, const T *a, const T *b, unsigned int size)
{
    T *dev_a = 0;
    T *dev_b = 0;
    T *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel1<T><<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

extern "C" void runtest()
{
    const int arraySize = 5;
    const double a_d[arraySize] = { 1.1, 2.2, 3.3, 4.4, 5.5 };
    const double b_d[arraySize] = { 10.1, 20.1, 30.1, 40.1, 50.1 };
    double c_d[arraySize] = { 0 };

    // Add vectors in parallel.
    operate<double> op;
    cudaError_t cudaStatus = op.addWithCuda(c_d, a_d, b_d, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return;
    }

    printf("{1.1,2.2,3.3,4.4,5.5} + {10.1,20.1,30.1,40.1,50.1} = {%f,%f,%f,%f,%f}\n",
        c_d[0], c_d[1], c_d[2], c_d[3], c_d[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }
}

template <class T>
__global__ void warpPerspectiveKernel(int in_rows,int in_cols,T* in_data,
                                      int out_rows,int out_cols,T* out_data,
                                      float* inv,T defVar)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < out_rows &&  x< out_cols)
    {
        float srcX=inv[0]*x+inv[1]*y+inv[2];
        float srcY=inv[3]*x+inv[4]*y+inv[5];
        float srcW=inv[6]*x+inv[7]*y+inv[8];
        srcW=1./srcW;srcX*=srcW;srcY*=srcW;
        if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)
        {
            out_data[x+y*out_cols]=in_data[(int)srcX+((int)srcY)*in_cols];
        }
        else
        {
            out_data[x+y*out_cols]=defVar;
        }
    }
}

template <class T>
bool operate<T>::warpPerspectiveCaller(int in_rows,int in_cols,T* in_data,
                           int out_rows,int out_cols,T* out_data,
                           float* inv,T defVar)
{
    T* in_dataGPU;
    T* out_dataGPU;
    float* invGPU;
    cudaMalloc((void**) &in_dataGPU, in_cols*in_rows*sizeof(T));
    cudaMalloc((void**) &out_dataGPU,out_cols*out_rows*sizeof(T));
    cudaMalloc((void**) &invGPU,9*sizeof(float));
    cudaMemcpy(in_dataGPU,in_data,in_cols*in_rows*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(invGPU,inv,9*sizeof(float),cudaMemcpyHostToDevice);

    dim3 threads(32,32);
        dim3 grid(divUp(out_cols, threads.x), divUp(out_rows, threads.y));
//        dim3 grid(20,20);

    pi::timer.enter("warpPerspectiveKernel");
    warpPerspectiveKernel<T><<<grid,threads>>>(in_rows,in_cols,in_dataGPU,
                                               out_rows,out_cols,out_dataGPU,
                                               invGPU,defVar);
    pi::timer.leave("warpPerspectiveKernel");

    cudaMemcpy(out_data,out_dataGPU,out_cols*out_rows*sizeof(T),cudaMemcpyDeviceToHost);
    cudaFree(in_dataGPU);cudaFree(out_dataGPU);cudaFree(invGPU);
    return true;
}

bool warpPerspective_uchar1(int in_rows,int in_cols,uchar1* in_data,
                            int out_rows,int out_cols,uchar1* out_data,
                            float* inv,uchar1 defVar)
{
    return operate<uchar1>::warpPerspectiveCaller(in_rows,in_cols,in_data,
                                                  out_rows,out_cols,out_data,
                                                  inv,defVar);
}
//bool warpPerspective_uchar2();
bool warpPerspective_uchar3(int in_rows,int in_cols,uchar3* in_data,
                            int out_rows,int out_cols,uchar3* out_data,
                            float* inv,uchar3 defVar)
{
    return operate<uchar3>::warpPerspectiveCaller(in_rows,in_cols,in_data,
                                                  out_rows,out_cols,out_data,
                                                  inv,defVar);
}

bool warpPerspective_uchar4(int in_rows,int in_cols,uchar4* in_data,
                            int out_rows,int out_cols,uchar4* out_data,
                            float* inv,uchar4 defVar)
{
    return operate<uchar4>::warpPerspectiveCaller(in_rows,in_cols,in_data,
                                                  out_rows,out_cols,out_data,
                                                  inv,defVar);
}


__global__ void renderFrameKernel(int in_rows,int in_cols,uchar3* in_data,//image in
                                  int out_rows,int out_cols,uchar4* out_data,
                                  bool fresh,uchar4 defVar,//image out
                                  float* inv,int centerX,int centerY//relations
                                  )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y<out_rows&&x<out_cols)
    {
        int idxOut=x+y*out_cols;

        // find source location
        float srcX=inv[0]*x+inv[1]*y+inv[2];
        float srcY=inv[3]*x+inv[4]*y+inv[5];
        float srcW=inv[6]*x+inv[7]*y+inv[8];
        srcW=1./srcW;srcX*=srcW;srcY*=srcW;

        if(fresh) //warp
        {
            if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)
            {
                uchar4* ptrOut=out_data+idxOut;
                *((uchar3*)ptrOut)=in_data[(int)srcX+((int)srcY)*in_cols];
                // compute weight
                {
                    //image weight
                    float difX=srcX-in_rows*0.5;
                    float difY=srcY-in_cols*0.5;
                    srcW=1000*(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols);
                    if(srcW<1) srcW=1;
                    //center weight
                }
                ptrOut->w=srcW;
            }
            else
            {
                out_data[idxOut]=defVar;
            }

        }
        else if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)// blender
        {
            uchar4* ptrOut=out_data+idxOut;
            // compute weight
            {
                //image weight
                float difX=srcX-in_rows*0.5;
                float difY=srcY-in_cols*0.5;
                srcW=1000*(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols);
                if(srcW<1) srcW=1;
                srcW=1;
                //center weight
            }
            if(ptrOut->w<srcW)
            {
                ptrOut->w=srcW;
                uchar3* ptrIn =in_data +(int)srcX+((int)srcY)*in_cols;
                *((uchar3*)ptrOut)=*ptrIn;
            }
        }
    }
}

__global__ void renderFramesKernel(int in_rows,int in_cols,uchar3* in_data,//image in
                                   int out_rows,int out_cols,uchar4** out_datas,
                                   bool* freshs,uchar4 defVar,//image out
                                   float* invs,int centerX,int centerY,int eleNum//relations
                                  )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y<out_rows&&x<out_cols)
    {
        for(int i=0;i<eleNum;i++)
        {
            uchar4* out_data=out_datas[i];
            bool    fresh=freshs[i];
            float*  inv=invs+9*i;

            int idxOut=x+y*out_cols;

            // find source location
            float srcX=inv[0]*x+inv[1]*y+inv[2];
            float srcY=inv[3]*x+inv[4]*y+inv[5];
            float srcW=inv[6]*x+inv[7]*y+inv[8];
            srcW=1./srcW;srcX*=srcW;srcY*=srcW;

            if(fresh) //warp
            {
                if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)
                {
                    uchar4* ptrOut=out_data+idxOut;
                    *((uchar3*)ptrOut)=in_data[(int)srcX+((int)srcY)*in_cols];
                    // compute weight
                    {
                        //image weight
                        float difX=srcX-in_rows*0.5;
                        float difY=srcY-in_cols*0.5;
                        srcW=1000*(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols);
                        if(srcW<1) srcW=1;
                        //center weight
                    }
                    ptrOut->w=srcW;
                }
                else
                {
                    out_data[idxOut]=defVar;
                }

            }
            else if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)// blender
            {
                uchar4* ptrOut=out_data+idxOut;
                // compute weight
                {
                    //image weight
                    float difX=srcX-in_rows*0.5;
                    float difY=srcY-in_cols*0.5;
                    srcW=1000*(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols);
                    if(srcW<1) srcW=1;
                    srcW=1;
                    //center weight
                }
                if(ptrOut->w<srcW)
                {
                    ptrOut->w=srcW;
                    uchar3* ptrIn =in_data +(int)srcX+((int)srcY)*in_cols;
                    *((uchar3*)ptrOut)=*ptrIn;
                }
            }
        }
    }
}

bool renderFrameCaller(CudaImage<uchar3>& rgbIn,CudaImage<uchar4>& ele,
                       float* inv,int centerX,int centerY)
{
    float* invGPU;
    cudaMalloc((void**) &invGPU,9*sizeof(float));
    cudaMemcpy(invGPU,inv,9*sizeof(float),cudaMemcpyHostToDevice);
    dim3 threads(32,32);
    uchar4 defVar;
    defVar.x=defVar.y=defVar.z=defVar.w=0;
    dim3 grid(divUp(ele.cols, threads.x), divUp(ele.rows, threads.y));
    renderFrameKernel<<<grid,threads>>>(rgbIn.rows,rgbIn.cols,rgbIn.data,
                                        ele.rows,ele.cols,ele.data,
                                        ele.fresh,defVar,invGPU,centerX,centerY);
    cudaFree(invGPU);
    return true;
}


bool renderFramesCaller(CudaImage<uchar3>& rgbIn,int out_rows,int out_cols,
                        uchar4** out_datas,bool* freshs,
                       float* invs,int centerX,int centerY,int eleNum)
{
    float* invGPU;
    uchar4** outDataGPU;
    bool*  freshesGPU;

    cudaMalloc((void**) &invGPU,9*sizeof(float)*eleNum);
    cudaMalloc((void**) &outDataGPU,sizeof(uchar4*)*eleNum);
    cudaMalloc((void**) &freshesGPU,sizeof(bool)*eleNum);
    cudaMemcpy(invGPU,invs,9*sizeof(float)*eleNum,cudaMemcpyHostToDevice);
    cudaMemcpy(outDataGPU,out_datas,sizeof(uchar4*)*eleNum,cudaMemcpyHostToDevice);
    cudaMemcpy(freshesGPU,freshs,sizeof(bool)*eleNum,cudaMemcpyHostToDevice);
    dim3 threads(32,32);
    uchar4 defVar;
    defVar.x=defVar.y=defVar.z=defVar.w=0;
    dim3 grid(divUp(out_cols, threads.x), divUp(out_rows, threads.y));
    renderFramesKernel<<<grid,threads>>>(rgbIn.rows,rgbIn.cols,rgbIn.data,
                                        out_rows,out_cols,outDataGPU,freshesGPU,
                                        defVar,invGPU,centerX,centerY,eleNum);
    cudaFree(invGPU);
    cudaFree(outDataGPU);
    cudaFree(freshesGPU);
    return true;
}



