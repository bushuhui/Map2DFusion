#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

#include "base/types/SIM3.h"

#define HasPBA

class Camera;
class FastPathTable;

struct BundleObserve
{
    uint frame_id;
    uint point_id;
    pi::Point2d p_img;
    double  invSigma2;
};

struct GPSObserve
{
    GPSObserve():measure(0,0,0),invSigma2(1){}
    uint frame_id;
    pi::Point3d gps_pose,measure;
    double  invSigma2;
};

class Optimizer
{
public:
    /// Return inliers number, and outliers are flaged by invSigma2==-1
    static int PoseOptimizationG2O(std::vector<pi::Point3d>& mappoints,std::vector<pi::Point2d>& observes,
                                   std::vector<float>& invSigma2,Camera* cam,pi::SE3d& pose);

    /// Return observe outliers
    static std::vector<uint> BundleAdjustmentG2O(std::vector<pi::SE3d>& KeyFrames,std::vector<pi::Point3d>& Points,
                                                 std::vector<BundleObserve>& Observes,bool* pbStopFlag=NULL,
                                                 int unFixKFNum=0,Camera* cam=NULL,
                                                 const std::vector<GPSObserve>& GpsObs=std::vector<GPSObserve>());

    /// This acqually returns nothing
    static std::vector<uint> BundleAdjustmentPBA(std::vector<pi::SE3d>& KeyFrames,std::vector<pi::Point3d>& Points,
                                                 std::vector<BundleObserve>& Observes,bool* pbStopFlag=NULL,
                                                 int unFixKFNum=0,Camera* cam=NULL);

    /** \brief This fitting GPS with a Sim3 matrix combines the time diff betweeen GPS and Video time
        \param  KeyFrames   gives cvPath
        \param  pathTable   gives gpsPath
        \param  sim3        the initial Sim3 and will be optimized
        \param  timeDiff    the initial timeDiff and will be optimized
        \param  GPSinvSigma below 0 makes all keyframes fixed
        \return The averaged squared error, failed if below 0
    **/
    static double FitGPSSIM3WithTime(std::vector<std::pair<double,pi::SE3d> >& cvPath,
                                   FastPathTable* gpsPath,pi::SIM3d& sim3,double& timeDiff,double GPSinvSigma=-1);

};

#endif // OPTIMIZER_H
