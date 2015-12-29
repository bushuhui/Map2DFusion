#include "lsdslam/DataStructures/Frame.h"
#include "lsdslam/Tracking/SE3Tracker.h"
#include "lsdslam/Tracking/TrackingReference.h"
#include "lsdslam/util/settings.h"

#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <iomanip>

#include <base/time/Global_Timer.h>
#include <base/time/Time.h>
#include <base/Svar/Svar.h>
#include <gui/gl/Win3D.h>
#include <gui/gl/glHelper.h>
#include <hardware/Camera/Camera.h>
#define HAS_DVO
#ifdef HAS_DVO
#include <dvo/core/rgbd_image.h>
#include <dvo/core/surface_pyramid.h>
#include <dvo/dense_tracking.h>

//#include <dvo_slam/local_tracker.h>
#endif

using namespace std;
#include "DirectFrame.h"

//#define USE_DIRECT

#ifdef USE_DIRECT

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lsdslam/DataStructures/Frame.h"
#include "lsdslam/Tracking/SE3Tracker.h"
#include "lsdslam/Tracking/TrackingReference.h"
#include "lsdslam/util/settings.h"

struct DirectEdge
{
    DirectEdge(const int kfId,const int frId,const pi::SE3d& relPose,
               const Eigen::Matrix<double,6,6>& info=Eigen::Matrix<double,6,6>::Identity())
        :id1(kfId),id2(frId),relaPose(relPose),infomation(info)
    {}

    uint            id1,id2;
    pi::SE3d        relaPose;//pose2=pose1*relaPose;
    Eigen::Matrix<double,6,6> infomation;
};

class DirectData
{
public:
    DirectData():ref(NULL),frame(NULL){}

    DirectData(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,cv::Mat k,bool isKF=false)
        :ref(NULL),frame(NULL)
    {
        Eigen::Matrix3f _k;
        _k<<k.at<float>(0,0),0,k.at<float>(0,2),
                0,k.at<float>(1,1),k.at<float>(1,2),
                0,0,1;
        initial(id,timestamp,im_rgb,depth,_k,isKF);
    }

    DirectData(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,const Eigen::Matrix3f& _k,bool isKF=false)
        :ref(NULL),frame(NULL)
    {
        initial(id,timestamp,im_rgb,depth,_k,isKF);
    }

    bool initial(int id,double timestamp,cv::Mat im_rgb,cv::Mat depth,const Eigen::Matrix3f& _k,bool isKF=false)
    {
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

    bool makeKF(cv::Mat depth)
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

    virtual ~DirectData()
    {
        minimize();
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

    virtual int trackFrame(DirectData* directData,const string& TrackerName="Default")
    {
        if(!frame||!ref) return -1;
        SvarWithType<SPtr<lsd_slam::SE3Tracker> > &trackers=SvarWithType<SPtr<lsd_slam::SE3Tracker> >::instance();
        if(!trackers.exist(TrackerName))
            trackers.insert(TrackerName,SPtr<lsd_slam::SE3Tracker>
                            (new lsd_slam::SE3Tracker(frame->width(),frame->height(),frame->K())));

        SPtr<lsd_slam::SE3Tracker> tracker=trackers[TrackerName];

        SE3 relativePose=ToSophus(pose.inverse()*directData->pose);
        relativePose=tracker->trackFrame(ref,directData->frame,relativePose);

        pi::SE3d relPose=fromSophus(relativePose);
        directData->pose=pose*relPose;

        directData->addEdge(id(),directData->id(),relPose);
    }

    bool addEdge(const int kfId,const int frId,const pi::SE3d& relPose,
                 const Eigen::Matrix<double,6,6>& info=Eigen::Matrix<double,6,6>::Identity())
    {
        edges.push_back(DirectEdge(kfId,frId,relPose,info));
    }

    int id(){if(frame) return frame->id();
            else return -1;}

    lsd_slam::TrackingReference* ref;
    lsd_slam::Frame*             frame;

    pi::SE3d                     pose;
    std::vector<DirectEdge>      edges;
};

#endif //USE_DIRECT

struct AssociateData
{
    double timestamp,timestampRGB,timestampDepth;
    double x,y,z,rx,ry,rz,w;
    string depthfile,rgbfile;
    pi::SE3d    se3;

    inline friend std::istream& operator >>(std::istream& is,AssociateData& rhs)
    {
        is>>rhs.timestamp>>rhs.x>>rhs.y>>rhs.z
         >>rhs.rx>>rhs.ry>>rhs.rz>>rhs.w;
        is>>rhs.timestampDepth>>rhs.depthfile;
        is>>rhs.timestampRGB>>rhs.rgbfile;
        rhs.se3=pi::SE3d(rhs.x,rhs.y,rhs.z,rhs.rx,rhs.ry,rhs.rz,rhs.w);
        return is;
    }

    inline friend std::ostream& operator <<(std::ostream& os,const AssociateData& rhs)
    {
        os<<rhs.timestamp<<" "<<rhs.x<<" "<<rhs.y<<" "<<rhs.z<<" ";
        os<<rhs.rx<<" "<<rhs.ry<<" "<<rhs.rz<<" "<<rhs.w<<" ";
        os<<rhs.timestampDepth<<" "<<rhs.depthfile<<" ";
        os<<rhs.timestampRGB<<" "<<rhs.rgbfile<<"\n";
        return os;
    }
};

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

typedef std::vector<AssociateData> Trajectory;

int TestDirectTracker()
{
    string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
    string dataFile=datapath+"/associate.txt";

    ifstream ifs(dataFile.c_str());
    if(!ifs.is_open())
    {
        cout<<"Can't open file "<<dataFile<<endl;
        return -1;
    }

    AssociateData f1,f2;
    ifs>>f1>>f2;
    cout<<"F1:"<<f1<<"F2:"<<f2;

    string rgbFile=datapath+"/"+f1.rgbfile;
    string depthFile=datapath+"/"+f1.depthfile;

    cv::Mat rgb=cv::imread(rgbFile,CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth=cv::imread(depthFile,cv::IMREAD_UNCHANGED);

    if(rgb.empty()||depth.empty())
    {
        cout<<"Failed to load keyframe.\n";
    }

    rgbFile=datapath+"/"+f2.rgbfile;
    depthFile=datapath+"/"+f2.depthfile;

    cv::Mat rgb1=cv::imread(rgbFile,CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat depth1=cv::imread(depthFile,cv::IMREAD_UNCHANGED);

    if(rgb1.empty()||depth1.empty())
    {
        cout<<"Failed to load frame2.\n";
    }
    assert(rgb.type() == CV_8U);
    assert(rgb1.type() == CV_8U);



    Eigen::Matrix3f k;
    k<< 523.44164, 0        , 314.19088,
            0        ,523.25609 , 268.74984,
            0     ,  0       ,1;

    cv::Mat floatDepth(depth.rows,depth.cols,CV_32F);

    u_int16_t dp;
    for(int i=0,iend=depth.rows*depth.cols;i<iend;i++)
    {
        dp=depth.at<u_int16_t>(i);
        if(dp>500)
            floatDepth.at<float>(i)=dp/5000.0;
        else
            floatDepth.at<float>(i)=-1;
    }

    pi::timer.enter("Preapare");
    lsd_slam::SE3Tracker tracker(rgb.cols,rgb.rows,k);
    lsd_slam::TrackingReference ref;
    lsd_slam::Frame kf(0,rgb.cols,rgb.rows,k,f1.timestamp,rgb.data);
    kf.setDepthFromGroundTruth((float*)floatDepth.data);
    ref.importFrame(&kf);

    lsd_slam::Frame fr(1,rgb.cols,rgb.rows,k,f2.timestamp,rgb1.data);

    pi::timer.leave("Preapare");



    SE3 initialPose;//=se3FromSim3(ref.keyframe->pose->getCamToWorld().inverse());
    pi::timer.enter("Track");
    initialPose=tracker.trackFrame(&ref,&fr,initialPose);
    pi::timer.leave("Track");

    initialPose=ToSophus(f1.se3)*initialPose;

    ref.invalidate();

    cout<<"T:"<<initialPose.translation()<<"R:"<<endl;
    initialPose=initialPose.inverse();
    cout<<"T:"<<initialPose.translation()<<"R:"<<initialPose.rotationMatrix();
    return 0;
}

class CameraRect:public pi::gl::GL_Object
{
public:
    CameraRect(const pi::SE3f& ps,const pi::gl::Color3b& c=pi::gl::Color3b(255,255,255),
               double fx=1.5,double fy=1,double length=1)
        :_fx(fx),_fy(fy),_l(length),pose(ps),color(c)
    {}

    virtual void draw()
    {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrix(pose);

        glColor3ub(color.x,color.y,color.z);

        float x=_l*_fx;
        float y=_l*_fy;
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(x,y,_l);

        glVertex3f(0,0,0);
        glVertex3f(-x,y,_l);

        glVertex3f(0,0,0);
        glVertex3f(x,-y,_l);

        glVertex3f(0,0,0);
        glVertex3f(-x,-y,_l);

        glVertex3f(x,y,_l);
        glVertex3f(-x,y,_l);

        glVertex3f(x,y,_l);
        glVertex3f(x,-y,_l);

        glVertex3f(x,y,_l);
        glVertex3f(-x,y,_l);

        glVertex3f(-x,-y,_l);
        glVertex3f(-x,y,_l);

        glVertex3f(x,-y,_l);
        glVertex3f(-x,-y,_l);
        glEnd();

        glPopMatrix();
    }

    float _fx,_fy,_l;
    pi::SE3f pose;
    pi::gl::Color3b color;
};


int DrawTrajectory()
{
    QApplication app(svar.GetInt("argc",1),
                     SvarWithType<char**>::instance()["argv"]);
    string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
    string dataFile=datapath+"/associate.txt";

    ifstream ifs(dataFile.c_str());
    if(!ifs.is_open())
    {
        cout<<"Can't open file "<<dataFile<<endl;
        return -1;
    }

    string line;
    Trajectory trajectory;
    pi::gl::Win3D win3d;
    win3d.setSceneRadius(10);
    while(getline(ifs,line))
    {
        AssociateData dt;
        stringstream str(line);
        str>>dt;
        trajectory.push_back(dt);
        CameraRect* rect=new CameraRect(dt.se3,pi::gl::Color3b(255,255,255),1.5,1,0.05);
        win3d.insert(rect);
    }

    cout<<"Got trajectory length:"<<trajectory.size()<<endl;

    win3d.show();
    glDisable(GL_LIGHTING);
    //    win3d.insert();
    return app.exec();

}

int TestSophus()
{
    pi::SE3d se3_zy;
    se3_zy.get_translation()=pi::Point3d(1,2,3);
    se3_zy.get_rotation()=pi::SO3d::exp(pi::Point3d(1,2,3));

    cout<<"ZY:"<<se3_zy.ln()<<endl;

    SE3 se3_s=ToSophus(se3_zy);
    cout<<"Sophus:"<<se3_s.log()<<endl;

    se3_zy=fromSophus(se3_s);

    Eigen::Matrix<double,3,4> H=se3_s.matrix3x4();
    cout<<"SophusMatrix:"<<H<<endl;
    double H1[3][4];
    se3_zy.getMatrixUnsafe(H1);
    cout<<"ZYMatrix:";
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
            cout<<H1[i][j]<<" ";
        cout<<endl;
    }
    return 0;
}

int LSDDirect()
{
    QApplication app(svar.GetInt("argc",1),
                     SvarWithType<char**>::instance()["argv"]);
    string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
    string dataFile=datapath+"/associate.txt";

    ifstream ifs(dataFile.c_str());
    if(!ifs.is_open())
    {
        cout<<"Can't open file "<<dataFile<<endl;
        return -1;
    }
    double factor=0.0002;
    if(svar.GetInt("IsMicro",0))
        factor=0.001;
    double rectLength=svar.GetDouble("CameraRect.length",0.01);

    string line;
    Trajectory trajectory;
    pi::gl::Win3D win3d;
    win3d.setSceneRadius(10);
    while(getline(ifs,line))
    {
        AssociateData dt;
        stringstream str(line);
        str>>dt;
        trajectory.push_back(dt);
        CameraRect* rect=new CameraRect(dt.se3,pi::gl::Color3b(255,255,255),1.5,1,rectLength);
        win3d.insert(rect);
    }

    cout<<"Got trajectory length:"<<trajectory.size()<<endl;

    win3d.show();
    glDisable(GL_LIGHTING);
    glColor3ub(255,0,0);

    string camName=svar.GetString("CameraName","KinectROS");
    lsd_slam::plotTrackingIterationInfo=svar.GetInt("PlotTrackingIterationInfo",0);
    lsd_slam::plotTracking=svar.GetInt("PlotTracking",0);

    Camera* cam=GetCameraFromName(camName);
    if(!cam||!cam->isValid())
    {
        cout<<"Can't load camera "<<camName<<endl;
        return app.exec();
    }
    cout<<"Camera "<<camName
       <<":"<<cam->info()<<endl;

    Eigen::Matrix3f k;
    k<< cam->Fx(), 0        , cam->Cx(),
            0        ,cam->Fy() , cam->Cy(),
            0     ,  0       ,1;

    bool needKF=true;
    lsd_slam::SE3Tracker* tracker=NULL;//(rgb.cols,rgb.rows,k);
    lsd_slam::TrackingReference* ref=NULL;
    shared_ptr<lsd_slam::Frame> currentFrame,lastFrame;
    SE3 initialPose,KF_Pose;
    ofstream ofs(svar.GetString("TrajectoryFile2Save","Trajectory.txt"));

    // Prepare reference
    int step=svar.GetInt("Step",1);
    for(int i=0;i<trajectory.size();i+=step)
    {
        pi::timer.leave("MainLoop");
        pi::timer.enter("MainLoop");
        //read one frame
        AssociateData& asso=trajectory[i];
        string rgbFile=datapath+"/"+asso.rgbfile;
        string depthFile=datapath+"/"+asso.depthfile;

        cv::Mat rgb=cv::imread(rgbFile,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat depth=cv::imread(depthFile,cv::IMREAD_UNCHANGED);

        if(rgb.empty()||depth.empty())
        {
            cout<<"Failed to load keyframe.\n";
            pi::timer.leave("MainLoop");
            break;
        }

        if(svar.GetInt("ShowImage",1))
        {
            cv::imshow("RGB",rgb);
            cv::imshow("DEPTH",depth);
        }

        uchar key=cv::waitKey(svar.GetDouble("WaitTime",20));
        if(key==27) i=trajectory.size();

        assert(rgb.type() == CV_8U);

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

        currentFrame=shared_ptr<lsd_slam::Frame>(new lsd_slam::Frame(i,rgb.cols,rgb.rows,k,asso.timestamp,rgb.data));
        currentFrame->setDepthFromGroundTruth((float*)floatDepth.data);

        cout<<"Loaded "<<asso<<endl;
        //prepare ref
        if(!ref)
        {
            ref=new lsd_slam::TrackingReference();
            ref->importFrame(currentFrame.get());
            KF_Pose=ToSophus(asso.se3);
            initialPose=KF_Pose;
            cout<<"Initial pose:"<<KF_Pose.log()<<endl;
            lastFrame=currentFrame;
            pi::timer.leave("MainLoop");
            continue;
        }
        if(!tracker)
        {
            tracker=new lsd_slam::SE3Tracker(currentFrame->width(),currentFrame->height(),k);
        }

        pi::timer.enter("Track");
        SE3 toRef  =tracker->trackFrame(ref,currentFrame.get(),KF_Pose.inverse()*initialPose);
        pi::timer.leave("Track");

        Eigen::Matrix<double,6,1> ln=toRef.log();
        double lnSq=ln.dot(ln);
        if(tracker->diverged||!tracker->trackingWasGood)
        {
            cout<<"Lost at "<<ln<<endl;
            double tracking_lastResidual = tracker->lastResidual;
            double tracking_lastUsage = tracker->pointUsage;
            double tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
            double tracking_lastGoodPerTotal = tracker->lastGoodCount / (currentFrame->width(SE3TRACKING_MIN_LEVEL)*currentFrame->height(SE3TRACKING_MIN_LEVEL));
            printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
                    currentFrame->id(),
                    100*tracking_lastGoodPerTotal,
                    100*tracking_lastGoodPerBad,
                    tracker->diverged ? "DIVERGED" : "NOT DIVERGED");
            if(svar.GetInt("ContinueIfLost",1))
            {
                ref->invalidate();
                ref->importFrame(currentFrame.get());
                initialPose=KF_Pose=ToSophus(asso.se3);
                lastFrame=currentFrame;
                pi::timer.leave("MainLoop");
                continue;
            }
            else
            {
                pi::timer.leave("MainLoop");
                break;
            }
        }
        initialPose=KF_Pose*toRef;
        cout<<setiosflags(ios::fixed)<<setprecision(8)<<"TrackResult:"<<fromSophus(initialPose)<<endl;
        cout<<"ShouldBe:"<<asso.se3<<endl;
        ofs<<setiosflags(ios::fixed)<<setprecision(8)<<asso.timestamp<<" "<<fromSophus(initialPose)<<endl;

        pi::gl::Color3b color=pi::gl::Color3b(0,255,0);

        if(lnSq>0.01)
        {
            pi::timer.enter("importFrame");
            ref->invalidate();
            lastFrame=currentFrame;
            cout<<"Inserting KF "<<asso.timestamp<<endl;
            ref->importFrame(lastFrame.get());
            KF_Pose=initialPose;
            color=pi::gl::Color3b(255,0,0);
            pi::timer.leave("importFrame");
        }
        {
            CameraRect* rect=new CameraRect(fromSophus(initialPose),color,1.5,1,rectLength);
            win3d.insert(rect);
            win3d.update();
        }
        pi::timer.leave("MainLoop");
    }
    if(ref)
        ref->invalidate();
    return app.exec();
}

int DirectSLAM()
{
    QApplication app(svar.GetInt("argc",1),
                     SvarWithType<char**>::instance()["argv"]);
    string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
    string dataFile=datapath+"/associate.txt";

    ifstream ifs(dataFile.c_str());
    if(!ifs.is_open())
    {
        cout<<"Can't open file "<<dataFile<<endl;
        return -1;
    }
    double factor=0.0002;
    if(svar.GetInt("IsMicro",0))
        factor=0.001;
    double rectLength=svar.GetDouble("CameraRect.length",0.01);

    string line;
    Trajectory trajectory;
    pi::gl::Win3D win3d;
    win3d.setSceneRadius(10);
    while(getline(ifs,line))
    {
        AssociateData dt;
        stringstream str(line);
        str>>dt;
        trajectory.push_back(dt);
        CameraRect* rect=new CameraRect(dt.se3,pi::gl::Color3b(255,255,255),1.5,1,rectLength);
        win3d.insert(rect);
    }

    cout<<"Got trajectory length:"<<trajectory.size()<<endl;

    win3d.show();
    glDisable(GL_LIGHTING);
    glColor3ub(255,0,0);

    string camName=svar.GetString("CameraName","KinectROS");
    lsd_slam::plotTrackingIterationInfo=svar.GetInt("PlotTrackingIterationInfo",0);
    lsd_slam::plotTracking=svar.GetInt("PlotTracking",0);

    Camera* cam=GetCameraFromName(camName);
    if(!cam||!cam->isValid())
    {
        cout<<"Can't load camera "<<camName<<endl;
        return app.exec();
    }
    cout<<"Camera "<<camName
       <<":"<<cam->info()<<endl;

    cv::Mat k=cv::Mat::eye(3,3,CV_32F);
    k.at<float>(0,0)=cam->Fx();
    k.at<float>(1,1)=cam->Fy();
    k.at<float>(0,2)=cam->Cx();
    k.at<float>(1,2)=cam->Cy();

    shared_ptr<DirectFrame> currentFrame,keyframe;
    pi::SE3d initialPose;
    ofstream ofs(svar.GetString("TrajectoryFile2Save","Trajectory.txt"));

    // Prepare reference
    int step=svar.GetInt("Step",1);
    for(int i=0;i<trajectory.size();i+=step)
    {
        pi::timer.leave("MainLoop");
        pi::timer.enter("MainLoop");
        //read one frame
        AssociateData& asso=trajectory[i];
        string rgbFile=datapath+"/"+asso.rgbfile;
        string depthFile=datapath+"/"+asso.depthfile;

        cv::Mat rgb=cv::imread(rgbFile,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat depth=cv::imread(depthFile,cv::IMREAD_UNCHANGED);

        if(rgb.empty()||depth.empty())
        {
            cout<<"Failed to load keyframe.\n";
            pi::timer.leave("MainLoop");
            break;
        }

        if(svar.GetInt("ShowImage",1))
        {
            cv::imshow("RGB",rgb);
            cv::imshow("DEPTH",depth);
        }
        uchar key=cv::waitKey(svar.GetDouble("WaitTime",20));
        if(key==27) i=trajectory.size();


        assert(rgb.type() == CV_8U);

        currentFrame=shared_ptr<DirectFrame>
                (new DirectFrame(i,asso.timestamp,rgb,depth,k));

        cout<<"Loaded "<<asso<<endl;
        //prepare ref
        if(!keyframe.get())
        {
            keyframe=currentFrame;
            keyframe->makeKF(depth);
            keyframe->setPose(asso.se3);
            initialPose=asso.se3;
            cout<<"Initial pose:"<<asso.se3<<endl;
            pi::timer.leave("MainLoop");
            continue;
        }

        pi::timer.enter("Track");
        currentFrame->setPose(initialPose);
        int ret=keyframe->trackFrame(*currentFrame,"Default");
        pi::timer.leave("Track");

        Eigen::Matrix<double,6,1> ln=ToSophus(currentFrame->edges()[0].relaPose).log();
        double lnSq=ln.dot(ln);

        SvarWithType<SPtr<lsd_slam::SE3Tracker> > &trackers=SvarWithType<SPtr<lsd_slam::SE3Tracker> >::instance();
        SPtr<lsd_slam::SE3Tracker> tracker=trackers["Default"];
        if(tracker->diverged||!tracker->trackingWasGood)
        {
            cout<<"Lost at "<<ln<<endl;
            if(svar.GetInt("ContinueIfLost",1))
            {
                keyframe=currentFrame;
                keyframe->makeKF(depth);
                keyframe->setPose(asso.se3);
                pi::timer.leave("MainLoop");
                continue;
            }
            else
            {
                pi::timer.leave("MainLoop");
                break;
            }
        }
        initialPose=currentFrame->getPose();
        cout<<setiosflags(ios::fixed)<<setprecision(8)<<"TrackResult:"<<initialPose<<endl;
        cout<<"ShouldBe:"<<asso.se3<<endl;
        ofs<<setiosflags(ios::fixed)<<setprecision(8)<<asso.timestamp<<" "<<initialPose<<endl;
        cout<<"Hession:"<<tracker->Hession<<endl;

        pi::gl::Color3b color=pi::gl::Color3b(0,255,0);

        if(lnSq>svar.GetDouble("Keyframe.MinDis",0.01))
        {
            pi::timer.enter("importFrame");
            keyframe=currentFrame;
            keyframe->makeKF(depth);
            cout<<"Inserting KF "<<asso.timestamp<<endl;
            color=pi::gl::Color3b(255,0,0);
            pi::timer.leave("importFrame");
        }
        {
            CameraRect* rect=new CameraRect(initialPose,color,1.5,1,rectLength);
            win3d.insert(rect);
            win3d.update();
        }
        pi::timer.leave("MainLoop");
    }
    return app.exec();
}


int tranTrajectory()
{
    svar.GetString("TransTrajectory.GroundTrue","groundtrue.txt");
    svar.GetString("TransTrajectory.in","in.txt");
    svar.GetString("TransTrajectory.out","out.txt");
}

#ifdef HAS_DVO


dvo::core::RgbdImagePyramidPtr load(dvo::core::RgbdCameraPyramid& camera, std::string rgb_file, std::string depth_file)
{
  cv::Mat rgb, grey, grey_s16, depth, depth_inpainted, depth_mask, depth_mono, depth_float;

  bool rgb_available = false;
  rgb = cv::imread(rgb_file, 1);
  depth = cv::imread(depth_file, -1);

  if(rgb.total() == 0 || depth.total() == 0) return dvo::core::RgbdImagePyramidPtr();

  if(rgb.type() != CV_32FC1)
  {
    if(rgb.type() == CV_8UC3)
    {
      cv::cvtColor(rgb, grey, CV_BGR2GRAY);
      rgb_available = true;
    }
    else
    {
      grey = rgb;
    }

    grey.convertTo(grey_s16, CV_32F);
  }
  else
  {
    grey_s16 = rgb;
  }

  if(depth.type() != CV_32FC1)
  {
    dvo::core::SurfacePyramid::convertRawDepthImageSse(depth, depth_float, 1.0f / 5000.0f);
  }
  else
  {
    depth_float = depth;
  }


  //depth_float.setTo(dvo::core::InvalidDepth, depth_float > 1.2f);

  dvo::core::RgbdImagePyramidPtr result = camera.create(grey_s16, depth_float);

  if(rgb_available)
    rgb.convertTo(result->level(0).rgb, CV_32FC3);

  return result;
}

Eigen::Affine3d toAffine(const pi::SE3d& se3)
{
    Eigen::Affine3d result=Eigen::Transform<double, 3, Eigen::Affine>::Identity();
    const pi::SO3d& so3=se3.get_rotation();
    Eigen::Quaterniond rotation(so3.w,so3.x,so3.y,so3.z);
    result= rotation.cast<double>()*result;

    const pi::Point3d& trans=se3.get_translation();
    result.translation()(0)=trans.x;
    result.translation()(1)=trans.y;
    result.translation()(2)=trans.z;

    return result;

}

pi::SE3d fromAffine(const Eigen::Affine3d& aff)
{
    Sophus::SE3d sophus(aff.rotation(),aff.translation());
    return fromSophus(sophus);
}

int DvoDirect()
{
    QApplication app(svar.GetInt("argc",1),
                     SvarWithType<char**>::instance()["argv"]);
    string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
    string dataFile=datapath+"/associate.txt";

    ifstream ifs(dataFile.c_str());
    if(!ifs.is_open())
    {
        cout<<"Can't open file "<<dataFile<<endl;
        return -1;
    }

    double rectLength=svar.GetDouble("CameraRect.length",0.01);

    string line;
    Trajectory trajectory;
    pi::gl::Win3D win3d;
    win3d.setSceneRadius(10);
    while(getline(ifs,line))
    {
        AssociateData dt;
        stringstream str(line);
        str>>dt;
        trajectory.push_back(dt);
        CameraRect* rect=new CameraRect(dt.se3,pi::gl::Color3b(255,255,255),1.5,1,rectLength);
        win3d.insert(rect);
    }

    cout<<"Got trajectory length:"<<trajectory.size()<<endl;

    win3d.show();
    glDisable(GL_LIGHTING);
    glColor3ub(255,0,0);

    /// 1. Load camera information

    string camName=svar.GetString("CameraName","KinectROS");

    Camera* cam=GetCameraFromName(camName);
    if(!cam||!cam->isValid())
    {
        cout<<"Can't load camera "<<camName<<endl;
        return app.exec();
    }
    cout<<"Camera "<<camName
       <<":"<<cam->info()<<endl;

    dvo::DenseTracker::Config cfg = dvo::DenseTracker::getDefaultConfig();
    cfg.UseWeighting = true;//false;
    cfg.UseInitialEstimate = true;
    cfg.FirstLevel = 4;
    cfg.LastLevel = 1;
    cfg.MaxIterationsPerLevel = 100;

    dvo::core::IntrinsicMatrix intrinsics = dvo::core::IntrinsicMatrix::create(cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy());

    dvo::core::RgbdCameraPyramid camera(640, 480, intrinsics);
    camera.build(cfg.getNumLevels());

    /// 2. Prepare tracking things
    dvo::DenseTracker tracker(cfg);
    dvo::core::RgbdImagePyramid::Ptr reference, current;

    Eigen::Affine3d reference_pose, current_pose, relative_pose;
    Eigen::Matrix<double, 6, 6> first_info, current_info;
    Eigen::Matrix2d first_error_precision;
    dvo::DenseTracker::Result result;

    ofstream ofs(svar.GetString("TrajectoryFile2Save","Trajectory.txt"));

    for(int i=0;i<trajectory.size();i++)
    {
        AssociateData& asso=trajectory[i];
        current=load(camera,datapath+"/"+asso.rgbfile,datapath+"/"+asso.depthfile);

        cv::imshow("RGB",current->level(0).rgb/255.0);
        cv::waitKey(svar.GetInt("WaitTime",20));

        if(!reference.get())
        {
            reference=current;
            reference_pose=toAffine(asso.se3);
            result.setIdentity();
            continue;
        }

        tracker.match(*reference,*current,result);
        relative_pose=result.Transformation;
        current_pose = reference_pose*relative_pose;

        if(result.isNaN())
        {
            //lost?
            std::cerr<<"Losted at "<<asso.timestamp<<endl;
        }

        pi::SE3d se3=fromAffine(current_pose);
        std::cerr<<"ShouldB:"<<asso.se3<<"\nResult:"<<se3<<endl;
        std::cerr<<"Error:"<<(asso.se3.inverse()*se3).ln()<<endl;
        std::cerr<<"Info:"<<result.Information<<endl;
        std::cerr<<"##################################################\n";

        Sophus::SE3d toRef(relative_pose.rotation(),relative_pose.translation());
        Eigen::Matrix<double,6,1> ln=toRef.log();
        double lnSq=ln.dot(ln);
        pi::gl::Color3b color=pi::gl::Color3b(0,255,0);
        if(lnSq>0.005||1)
        {
            //kf?
            result.setIdentity();
//            result.Transformation=Eigen::Affine3d::Identity();
            reference=current;
            reference_pose=current_pose;
            color=pi::gl::Color3b(255,0,0);
        }


        {
            CameraRect* rect=new CameraRect(se3,color,1.5,1,rectLength);
            win3d.insert(rect);
            win3d.update();
        }
    }

    return app.exec();
}
#if 0
int DvoLocalTracker()
{
    QApplication app(svar.GetInt("argc",1),
                     SvarWithType<char**>::instance()["argv"]);
    string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
    string dataFile=datapath+"/associate.txt";

    ifstream ifs(dataFile.c_str());
    if(!ifs.is_open())
    {
        cout<<"Can't open file "<<dataFile<<endl;
        return -1;
    }

    double rectLength=svar.GetDouble("CameraRect.length",0.01);

    string line;
    Trajectory trajectory;
    pi::gl::Win3D win3d;
    win3d.setSceneRadius(10);
    while(getline(ifs,line))
    {
        AssociateData dt;
        stringstream str(line);
        str>>dt;
        trajectory.push_back(dt);
        CameraRect* rect=new CameraRect(dt.se3,pi::gl::Color3b(255,255,255),1.5,1,rectLength);
        win3d.insert(rect);
    }

    cout<<"Got trajectory length:"<<trajectory.size()<<endl;

    win3d.show();
    glDisable(GL_LIGHTING);
    glColor3ub(255,0,0);

    /// 1. Load camera information

    string camName=svar.GetString("CameraName","KinectROS");

    Camera* cam=GetCameraFromName(camName);
    if(!cam||!cam->isValid())
    {
        cout<<"Can't load camera "<<camName<<endl;
        return app.exec();
    }
    cout<<"Camera "<<camName
       <<":"<<cam->info()<<endl;

    dvo::DenseTracker::Config cfg = dvo::DenseTracker::getDefaultConfig();
    cfg.UseWeighting = true;//false;
    cfg.UseInitialEstimate = true;
    cfg.FirstLevel = 4;
    cfg.LastLevel = 1;
    cfg.MaxIterationsPerLevel = 100;

    dvo::core::IntrinsicMatrix intrinsics = dvo::core::IntrinsicMatrix::create(cam->Fx(), cam->Fy(), cam->Cx(), cam->Cy());

    dvo::core::RgbdCameraPyramid camera(640, 480, intrinsics);
    camera.build(cfg.getNumLevels());

    /// 2. Prepare tracking things
    dvo::DenseTracker tracker(cfg);
    dvo::core::RgbdImagePyramid::Ptr reference, current;

    Eigen::Affine3d reference_pose, current_pose, relative_pose;
    Eigen::Matrix<double, 6, 6> first_info, current_info;
    Eigen::Matrix2d first_error_precision;
    dvo::DenseTracker::Result result;

    ofstream ofs(svar.GetString("TrajectoryFile2Save","Trajectory.txt"));

    for(int i=0;i<trajectory.size();i++)
    {
        AssociateData& asso=trajectory[i];
        current=load(camera,datapath+"/"+asso.rgbfile,datapath+"/"+asso.depthfile);

        cv::imshow("RGB",current->level(0).rgb/255.0);
        cv::waitKey(20);

        if(!reference.get())
        {
            reference=current;
            reference_pose=toAffine(asso.se3);
            result.setIdentity();
            continue;
        }

        tracker.match(*reference,*current,result);
        relative_pose=result.Transformation;
        current_pose = reference_pose*relative_pose;

        if(result.isNaN())
        {
            //lost?
            std::cerr<<"Losted at "<<asso.timestamp<<endl;
        }

        Sophus::SE3d toRef(relative_pose.rotation(),relative_pose.translation());
        Eigen::Matrix<double,6,1> ln=toRef.log();
        double lnSq=ln.dot(ln);

        pi::gl::Color3b color=pi::gl::Color3b(0,255,0);
        if(lnSq>0.005||1)
        {
            //kf?
            result.setIdentity();
//            result.Transformation=Eigen::Affine3d::Identity();
            reference=current;
            reference_pose=current_pose;
            color=pi::gl::Color3b(255,0,0);
        }

        pi::SE3d se3=fromAffine(current_pose);
        std::cerr<<"ShouldB:"<<asso.se3<<"\nResult:"<<se3<<endl;

        {
            CameraRect* rect=new CameraRect(se3,color,1.5,1,rectLength);
            win3d.insert(rect);
            win3d.update();
        }
    }

    return app.exec();
}
#endif


#else
int DvoDirect()
{
    return 0;
}
#endif

int Micro2TUM()
{
    string dataPath     = svar.GetString("DataPath",".");
    string associateOut = dataPath+"/../associate.txt";
    string groundfile   = dataPath+"/../groundtrue.txt";
    string::size_type n;
    n=dataPath.find_last_of("/");
    string foldername=dataPath.substr(n+1);
    ofstream asso(associateOut.c_str());
    ofstream groundtrue(groundfile.c_str());

    double timestamp=pi::tm_getTimeStamp();
    char Stimestamp[100];
    double timestep = 1./30;

    int i=0;
    while(1)
    {
        char info_file[100];
        sprintf(info_file,"frame-%06d.pose.txt",i);
        char depth_file[100];
        sprintf(depth_file,"frame-%06d.depth.png",i);
        char rgb_file[100];
        sprintf(rgb_file,"frame-%06d.color.png",i);
        string info_path=dataPath+"/"+info_file;
        cerr<<"Reading "<<info_path<<endl;
        ifstream info(info_path.c_str());
        if(!info.is_open()) break;
        double m[16];
        for(int j=0;j<16;j++)
                info>>m[j];
        pi::SE3d se3;
        se3.fromMatrix(m);

        sprintf(Stimestamp,"%8.3f",timestamp);

        groundtrue<<Stimestamp<<" "<<se3.inverse()<<endl;
        asso<<Stimestamp<<" "<<se3<<" "
           <<Stimestamp<<" "<<foldername<<"/"<<depth_file<<" "
          <<Stimestamp<<" "<<foldername<<"/"<<rgb_file<<endl;

        timestamp+=timestep;
        i++;
    }

    return 0;
}

int main(int argc,char** argv)
{
    svar.ParseMain(argc,argv);
    string act=svar.GetString("Act","TestDirectTracker");
    if(act=="TestDirectTracker") return TestDirectTracker();
    if(act=="DrawTrajectory")    return DrawTrajectory();
    if(act=="TestSophus")        return TestSophus();
    if(act=="DirectSLAM")        return DirectSLAM();
    if(act=="LSDDirect")        return  LSDDirect();
    if(act=="DvoDirect")         return DvoDirect();
    if(act=="Micro2TUM")         return Micro2TUM();
}
