#ifndef DIRECTFRAMEIMPL_H
#define DIRECTFRAMEIMPL_H

#include "lsdslam/DataStructures/Frame.h"
#include "lsdslam/Tracking/SE3Tracker.h"
#include "lsdslam/Tracking/TrackingReference.h"
#include "lsdslam/util/settings.h"

#include <eigen3/Eigen/Core>
#include <base/time/Global_Timer.h>

#include "DirectFrame.h"

class DirectFrameImpl
{
public:
    DirectFrameImpl(){}
    virtual ~DirectFrameImpl(){}

    virtual bool initial(int id_,double timestamp_,cv::Mat im_rgb,cv::Mat depth,
                 cv::Mat k,bool isKF=false)=0;

    virtual bool makeKF(cv::Mat depth){}

    virtual void minimize(){}

    inline virtual int trackFrame(DirectFrame& directData,const std::string& TrackerName="Default")=0;

    virtual DirectFrame::DirectType type()=0;

    int                          id;
    double                       timestamp;
    pi::SE3d                     pose;
    std::vector<DirectEdge>      edges;
};


class LSDDirectImpl:public DirectFrameImpl
{
public:
    LSDDirectImpl():ref(NULL),frame(NULL){}

    LSDDirectImpl(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,cv::Mat k,bool isKF=false)
        :ref(NULL),frame(NULL)
    {
        initial(id,timestamp,im_rgb,depth,k,isKF);
    }

    virtual ~LSDDirectImpl()
    {
        minimize();
    }

    DirectFrame::DirectType type(){return DirectFrame::LSD;}

    virtual bool initial(int id_,double _timestamp,cv::Mat im_rgb,cv::Mat depth,cv::Mat k,bool isKF=false)
    {
        id=id_;
        timestamp=_timestamp;

        Eigen::Matrix3f _k;
        _k<<k.at<float>(0,0),0,k.at<float>(0,2),
                0,k.at<float>(1,1),k.at<float>(1,2),
                0,0,1;
        cv::Mat im_gray;
        if(im_rgb.channels()==3)
            cv::cvtColor(im_rgb,im_gray,CV_BGR2GRAY);
        else
            im_gray=im_rgb.clone();

        frame=new lsd_slam::Frame(id,im_rgb.cols,im_rgb.rows,_k,timestamp,im_gray.data);
        if(isKF)
        {
            makeKF(depth);
        }
    }

    virtual bool makeKF(cv::Mat depth)
    {
        if(ref) return 0;
        float factor=svar.GetDouble("Depth.Factor",0.0002);
        cv::Mat floatDepth(depth.rows,depth.cols,CV_32F);
        u_int16_t dp;
        for(int i=0,iend=depth.rows*depth.cols;i<iend;i++)
        {
            dp=depth.at<u_int16_t>(i);
            if(dp>500)
                floatDepth.at<float>(i)=dp*factor;
            else
                floatDepth.at<float>(i)=-1;
        }
        frame->setDepthFromGroundTruth((float*)floatDepth.data);
        ref=new lsd_slam::TrackingReference();
        ref->importFrame(frame);
    }

    virtual void minimize()
    {
        if(ref)
        {
            ref->invalidate();
            delete ref;
            ref=NULL;
        }
        if(frame)
        {
            delete frame;
            frame=NULL;
        }
    }

    pi::SE3d fromSophus(const SE3& se3)
    {
        pi::SE3d result;
        Eigen::Vector3d trans=se3.translation();
        result.get_translation()=*(pi::Point3d*)&trans;

        Eigen::Matrix3d rot =se3.inverse().rotationMatrix();
        result.get_rotation().fromMatrixUnsafe(rot);

        return result;
    }

    SE3 ToSophus(pi::SE3<double> se3_zy)
    {
        Eigen::Matrix3d eigen_rot;
        se3_zy.get_rotation().inv().getMatrixUnsafe(eigen_rot);
        Eigen::Vector3d eigen_trans=*((Eigen::Vector3d*)&se3_zy.get_translation());

        return Sophus::SE3(eigen_rot,eigen_trans);
    }

    inline virtual int trackFrame(DirectFrame& directData,const std::string& TrackerName="Default")
    {
        //check
        if(!frame||!ref||directData.type()!=DirectFrame::LSD) return -1;

        //get tracker
        SvarWithType<SPtr<lsd_slam::SE3Tracker> > &trackers=SvarWithType<SPtr<lsd_slam::SE3Tracker> >::instance();
        if(!trackers.exist(TrackerName))
            trackers.insert(TrackerName,SPtr<lsd_slam::SE3Tracker>
                            (new lsd_slam::SE3Tracker(frame->width(),frame->height(),frame->K())));
        SPtr<lsd_slam::SE3Tracker> tracker=trackers[TrackerName];

        //do track
        SPtr<DirectFrameImpl> impl=directData.getImpl();
        LSDDirectImpl *lsd_impl=(LSDDirectImpl*)impl.get();

        SE3 relativePose=ToSophus(pose.inverse()*lsd_impl->pose);
        pi::timer.enter("LSD::TrackFrame");
        relativePose=tracker->trackFrame(ref,lsd_impl->frame,relativePose);
        pi::timer.leave("LSD::TrackFrame");

        //finish track
        if(tracker->diverged||!tracker->trackingWasGood) return -2;//losted

        pi::SE3d relPose=fromSophus(relativePose);
        lsd_impl->pose=pose*relPose;

        Eigen::Matrix<double,6,6> info_eigen;
        for(int i=0;i<36;i++) ((double*)&info_eigen)[i]=((float*)&tracker->Hession)[i];
        pi::Array_<double,36> &info=*(pi::Array_<double,36>*)&info_eigen;
        directData.addEdge(DirectEdge(id,directData.id(),relPose,info));
    }

    lsd_slam::TrackingReference* ref;
    lsd_slam::Frame*             frame;
};

#endif // DIRECTFRAMEIMPL_H
