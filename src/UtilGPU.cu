#include "UtilGPU.cuh"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <base/time/Global_Timer.h>


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
                srcW=1000*(0.25-(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols));
                if(srcW<1) srcW=1;
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
                                   float* invs,int* centers,int eleNum//relations
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
            uchar4* ptrOut=out_data+idxOut;

            // find source location
            float srcX=inv[0]*x+inv[1]*y+inv[2];
            float srcY=inv[3]*x+inv[4]*y+inv[5];
            float srcW=inv[6]*x+inv[7]*y+inv[8];
            srcW=1./srcW;srcX*=srcW;srcY*=srcW;

            if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)
            {
                // compute weight
                {
                    //image weight
                    float difX=srcX-in_rows*0.5;
                    float difY=srcY-in_cols*0.5;
                    srcW=(0.25-(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols));//0~0.25
                    //center weight
                    if(1)
                    {
                        difX=centers[i*2]-x;
                        difY=centers[i*2+1]-y;
                        srcW=5e4*srcW/sqrt(difX*difX+difY*difY+1);
                    }
                    else
                        srcW=1000*srcW;
                    if(srcW<1) srcW=1;
                    else if(srcW>255) srcW=255;
                }
                if(fresh)
                {
                    *((uchar3*)ptrOut)=in_data[(int)srcX+((int)srcY)*in_cols];

                    ptrOut->w=srcW;
                }
                else// blender
                {
                    if(ptrOut->w<srcW)
                    {
                        ptrOut->w=srcW;
                        uchar3* ptrIn =in_data +(int)srcX+((int)srcY)*in_cols;
                        *((uchar3*)ptrOut)=*ptrIn;
                    }
                }
            }
            else if(fresh)
            {
                *ptrOut=defVar;
            }
        }
    }
}

__global__ void renderFramesKernel(int in_rows,int in_cols,uchar3* in_data,//image in
                                   int out_rows,int out_cols,float4** out_datas,
                                   bool* freshs,float4 defVar,//image out
                                   float* invs,int* centers,int eleNum//relations
                                  )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y<out_rows&&x<out_cols)
    {
        for(int i=0;i<eleNum;i++)
        {
            float4* out_data=out_datas[i];
            bool    fresh=freshs[i];
            float*  inv=invs+9*i;

            int idxOut=x+y*out_cols;
            float4* ptrOut=out_data+idxOut;

            // find source location
            float srcX=inv[0]*x+inv[1]*y+inv[2];
            float srcY=inv[3]*x+inv[4]*y+inv[5];
            float srcW=inv[6]*x+inv[7]*y+inv[8];
            srcW=1./srcW;srcX*=srcW;srcY*=srcW;

            if(srcX<in_cols&&srcX>=0&&srcY<in_rows&&srcY>=0)
            {
                // compute weight
                {
                    //image weight
                    float difX=srcX-in_rows*0.5;
                    float difY=srcY-in_cols*0.5;
                    srcW=(0.25-(difX*difX+difY*difY)/(in_rows*in_rows+in_cols*in_cols));//0~0.25
                    //center weight
                    if(0)
                    {
                        difX=centers[i*2]-x;
                        difY=centers[i*2+1]-y;
                        srcW=1e5*srcW/(difX*difX+difY*difY+1000);
                    }
                }
                if(fresh||ptrOut->w<srcW)
                {
                    uchar3* ptrIn =in_data +(int)srcX+((int)srcY)*in_cols;
                    ptrOut->x=ptrIn->x*0.00392f;
                    ptrOut->y=ptrIn->y*0.00392f;
                    ptrOut->z=ptrIn->z*0.00392f;
                    ptrOut->w=srcW;
                }
            }
            else if(fresh)
            {
                *ptrOut=defVar;
            }
        }
    }
}

bool renderFrameCaller(CudaImage<uchar3>& rgbIn,CudaImage<uchar4>& ele,
                       float* inv,int centerX,int centerY)
{
    float* invGPU;
    checkCudaErrors(cudaMalloc((void**) &invGPU,9*sizeof(float)));
    checkCudaErrors(cudaMemcpy(invGPU,inv,9*sizeof(float),cudaMemcpyHostToDevice));
    dim3 threads(32,32);
    uchar4 defVar;
    defVar.x=defVar.y=defVar.z=defVar.w=0;
    dim3 grid(divUp(ele.cols, threads.x), divUp(ele.rows, threads.y));
    renderFrameKernel<<<grid,threads>>>(rgbIn.rows,rgbIn.cols,rgbIn.data,
                                        ele.rows,ele.cols,ele.data,
                                        ele.fresh,defVar,invGPU,centerX,centerY);
    checkCudaErrors(cudaFree(invGPU));
    return true;
}


bool renderFramesCaller(CudaImage<uchar3>& rgbIn,int out_rows,int out_cols,
                        uchar4** out_datas,bool* freshs,
                       float* invs,int* centers,int eleNum)
{
    float* invGPU;
    uchar4** outDataGPU;
    bool*  freshesGPU;
    int*   centersGPU;

    checkCudaErrors(cudaMalloc((void**) &invGPU,9*sizeof(float)*eleNum));
    checkCudaErrors(cudaMalloc((void**) &outDataGPU,sizeof(uchar4*)*eleNum));
    checkCudaErrors(cudaMalloc((void**) &freshesGPU,sizeof(bool)*eleNum));
    checkCudaErrors(cudaMalloc((void**) &centersGPU,2*sizeof(int)*eleNum));
    checkCudaErrors(cudaMemcpy(invGPU,invs,9*sizeof(float)*eleNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(outDataGPU,out_datas,sizeof(uchar4*)*eleNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(freshesGPU,freshs,sizeof(bool)*eleNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(centersGPU,centers,2*sizeof(int)*eleNum,cudaMemcpyHostToDevice));
    dim3 threads(32,32);
    uchar4 defVar;
    defVar.x=defVar.y=defVar.z=defVar.w=0;
    dim3 grid(divUp(out_cols, threads.x), divUp(out_rows, threads.y));
    renderFramesKernel<<<grid,threads>>>(rgbIn.rows,rgbIn.cols,rgbIn.data,
                                        out_rows,out_cols,outDataGPU,freshesGPU,
                                        defVar,invGPU,centersGPU,eleNum);
    checkCudaErrors(cudaFree(invGPU));
    checkCudaErrors(cudaFree(outDataGPU));
    checkCudaErrors(cudaFree(freshesGPU));
    checkCudaErrors(cudaFree(centersGPU));
    return true;
}

bool renderFramesCaller(CudaImage<uchar3>& rgbIn,int out_rows,int out_cols,
                        float4** out_datas,bool* freshs,
                       float* invs,int* centers,int eleNum)
{
    float* invGPU;
    float4** outDataGPU;
    bool*  freshesGPU;
    int*   centersGPU;

    checkCudaErrors(cudaMalloc((void**) &invGPU,9*sizeof(float)*eleNum));
    checkCudaErrors(cudaMalloc((void**) &outDataGPU,sizeof(float4*)*eleNum));
    checkCudaErrors(cudaMalloc((void**) &freshesGPU,sizeof(bool)*eleNum));
    checkCudaErrors(cudaMalloc((void**) &centersGPU,2*sizeof(int)*eleNum));
    checkCudaErrors(cudaMemcpy(invGPU,invs,9*sizeof(float)*eleNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(outDataGPU,out_datas,sizeof(float4*)*eleNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(freshesGPU,freshs,sizeof(bool)*eleNum,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(centersGPU,centers,2*sizeof(int)*eleNum,cudaMemcpyHostToDevice));
    dim3 threads(32,32);
    float4 defVar;
    defVar.x=defVar.y=defVar.z=defVar.w=0;
    dim3 grid(divUp(out_cols, threads.x), divUp(out_rows, threads.y));
    renderFramesKernel<<<grid,threads>>>(rgbIn.rows,rgbIn.cols,rgbIn.data,
                                        out_rows,out_cols,outDataGPU,freshesGPU,
                                        defVar,invGPU,centersGPU,eleNum);
    checkCudaErrors(cudaFree(invGPU));
    checkCudaErrors(cudaFree(outDataGPU));
    checkCudaErrors(cudaFree(freshesGPU));
    checkCudaErrors(cudaFree(centersGPU));
    return true;
}


