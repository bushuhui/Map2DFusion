#ifndef UTILGPU_H
#define UTILGPU_H
#include <opencv2/core/core.hpp>

class UtilGPU
{
public:
    UtilGPU();
    static bool warpPerspective(cv::Mat src,cv::Mat& dst,cv::Mat transmtx,cv::Size dstSize);
};

#endif // UTILGPU_H
