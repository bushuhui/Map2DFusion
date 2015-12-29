#ifndef DIRECTFRAME_H
#define DIRECTFRAME_H

#include <base/types/SE3.h>
#include <base/Svar/Svar.h>
#include <base/types/SPtr.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


struct DirectEdge
{
    DirectEdge(const int kfId,const int frId,const pi::SE3d& relPose,
               pi::Array_<double,36> info)
        :id1(kfId),id2(frId),relaPose(relPose),infomation(info)
    {}

    uint                    id1,id2;
    pi::SE3d                relaPose;//pose2=pose1*relaPose;
    pi::Array_<double,36>   infomation;
};

class DirectFrameImpl;
class LSDDirectImpl;
class DVODirectImpl;

class DirectFrame
{
public:
    enum DirectType{LSD=0,DVO=1};
    DirectFrame(DirectType type=LSD);

    DirectFrame(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,
                cv::Mat k,bool isKF=false,DirectType type=LSD);

    ~DirectFrame();

    DirectType type();

    bool initial(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,
                 cv::Mat k,bool isKF=false);

    bool makeKF(cv::Mat depth);

    void minimize();

    // Pose functions
    int      id();

    pi::SE3d getPose();

    void setPose(const pi::SE3d& pose);

    // Edge functions
    const std::vector<DirectEdge>& edges();

    int edgeNum();

    int addEdge(const DirectEdge& edge);

    // TrackThings
    int trackFrame(DirectFrame& directData,const std::string& TrackerName="Default");

protected:
    friend class DirectFrameImpl;
    friend class LSDDirectImpl;
    friend class DVODirectImpl;
    SPtr<DirectFrameImpl> getImpl(){return impl;}

private:
    SPtr<DirectFrameImpl>   impl;
};

#endif // DIRECTFRAME_H
