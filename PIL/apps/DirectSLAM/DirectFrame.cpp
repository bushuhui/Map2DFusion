#include "DirectFrame.h"
#include "DirectFrameImpl.h"

DirectFrame::DirectFrame(DirectType type)
{
    if(type==LSD)
        impl=SPtr<DirectFrameImpl>(new LSDDirectImpl());
    else if(type==DVO)
        impl=SPtr<DirectFrameImpl>(new LSDDirectImpl());
}

DirectFrame::DirectFrame(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,
            cv::Mat k,bool isKF,DirectType type)
{
    if(type==LSD)
        impl=SPtr<DirectFrameImpl>(new LSDDirectImpl(id,timestamp,im_rgb,depth,k,isKF));
}

DirectFrame::~DirectFrame()
{
    impl=SPtr<DirectFrameImpl>();
}

DirectFrame::DirectType DirectFrame::type()
{
    return impl->type();
}

bool DirectFrame::initial(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,
             cv::Mat k,bool isKF)
{
    return impl->initial(id,timestamp,im_rgb,depth,k,isKF);
}

bool DirectFrame::makeKF(cv::Mat depth)
{
    return impl->makeKF(depth);
}

void DirectFrame::minimize()
{
    impl->minimize();
}

int  DirectFrame::id()
{
    return impl->id;
}

pi::SE3d DirectFrame::getPose()
{
    return impl->pose;
}

void DirectFrame::setPose(const pi::SE3d &pose)
{
    impl->pose=pose;
}

const std::vector<DirectEdge>& DirectFrame::edges()
{
    return impl->edges;
}

int DirectFrame::edgeNum()
{
    return impl->edges.size();
}

int DirectFrame::addEdge(const DirectEdge& edge)
{
    impl->edges.push_back(edge);
    return 0;
}

int DirectFrame::trackFrame(DirectFrame& directData,const std::string& TrackerName)
{
    return impl->trackFrame(directData,TrackerName);
}
