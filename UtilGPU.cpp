#include "UtilGPU.h"
#include "UtilGPU.cuh"
#include <iostream>
using namespace std;

UtilGPU::UtilGPU()
{
}

bool UtilGPU::warpPerspective(cv::Mat src,cv::Mat& dst,cv::Mat transmtx,cv::Size dstSize)
{
    dst.create(dstSize.height,dstSize.width,src.type());
    cv::Mat inv=transmtx.inv();
//    std::cout<<"inv:"<<inv<<endl;
    inv.convertTo(inv,CV_32F);
    if(src.type()==CV_8UC3)
    {
        uchar3 defVar;
        defVar.x=defVar.y=defVar.z=0;
        return warpPerspective_uchar3(src.rows,src.cols,(uchar3*)src.data,
                                               dst.rows,dst.cols,(uchar3*)dst.data,
                                      (float*)inv.data,defVar);
    }
    else if(src.type()==CV_8UC4)
    {
        uchar4 defVar;
        defVar.x=defVar.y=defVar.z=defVar.w=0;
        return warpPerspective_uchar4(src.rows,src.cols,(uchar4*)src.data,
                                      dst.rows,dst.cols,(uchar4*)dst.data,(float*)inv.data,defVar);
    }
    else if(src.type()==CV_8UC1)
    {
        uchar1 defVar;defVar.x=0;
        return warpPerspective_uchar1(src.rows,src.cols,(uchar1*)src.data,
                                      dst.rows,dst.cols,(uchar1*)dst.data,(float*)inv.data,defVar);
    }
}
