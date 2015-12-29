#include <gui/gl/GL_Object.h>
#include <gui/gl/Win3D.h>
#include <gui/gl/glHelper.h>
#include <QColor>

#include <base/Svar/Svar.h>
#include <base/system/thread/ThreadBase.h>
#include <base/time/Global_Timer.h>

#include <fastfusion/OnlineFusionObject.h>

#include <hardware/Camera/Camera.h>

using namespace std;

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
typedef std::vector<AssociateData> Trajectory;

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


CameraInfo fromSE3(pi::SE3d pose,float fx,float fy,float cx,float cy)
{
    CameraInfo result;
    cv::Mat intrinsic = cv::Mat::eye(3,3,cv::DataType<double>::type);
    //Kinect Intrinsic Parameters
    intrinsic.at<double>(0,0) = fx;
    intrinsic.at<double>(1,1) = fy;
    intrinsic.at<double>(0,2) = cx;
    intrinsic.at<double>(1,2) = cy;

    result.setIntrinsic(intrinsic);
    cv::Mat rotation2 = cv::Mat::eye(3,3,cv::DataType<double>::type);
//    for(int i=0;i<3;i++) for(int j=0;j<3;j++) rotation2.at<double>(i,j) = rotation(i,j);
    pose.get_rotation().getMatrix((double*)rotation2.data);
    result.setRotation(rotation2);

    pi::Point3d trans=pose.get_translation();
    cv::Mat translation2 = cv::Mat::zeros(3,1,cv::DataType<double>::type);
    translation2.at<double>(0,0) = trans.x;
    translation2.at<double>(1,0) = trans.y;
    translation2.at<double>(2,0) = trans.z;
    result.setTranslation(translation2);
    return result;
}

class FusionThread:public pi::Runnable
{
public:
    FusionThread(pi::gl::Win3D* window=NULL,OnlineFusionObject* obj=NULL)
        :win3d(window),fusionObj(obj),shouldstop(false),running(false)
    {}
    void run()
    {
        if(!win3d||!fusionObj) return;
        running=true;
        string datapath=svar.GetString("DataPath","/data/zhaoyong/Linux/Program/Apps/Tests/lsdslamModified/data");
        string dataFile=datapath+"/associate.txt";

        double factor=1./5000;
        if(svar.GetInt("IsMicro",0)) factor=0.001;

        ifstream ifs(dataFile.c_str());
        if(!ifs.is_open())
        {
            cout<<"Can't open file "<<dataFile<<endl;
            running=false;
            return ;
        }

        double rectLength=svar.GetDouble("CameraRect.length",0.01);

        string line;
        Trajectory trajectory;
        win3d->setSceneRadius(10);
        while(getline(ifs,line))
        {
            AssociateData dt;
            stringstream str(line);
            str>>dt;
            trajectory.push_back(dt);
        }

        cout<<"Got trajectory length:"<<trajectory.size()<<endl;

        string camName=svar.GetString("CameraName","KinectROS");
        int    step   =svar.GetInt("Step",1);
        int    updateStep=svar.GetInt("UpdateStep",5);
        int    updateCount=-updateStep;
        bool   shouldUpdateMesh=false;

        Camera* cam=GetCameraFromName(camName);

        pi::Rate rate(30);
        for(int i=0;i<trajectory.size()&&!shouldstop;i+=step)
        {
            AssociateData& asso=trajectory[i];
            string rgb_file=datapath+"/"+asso.rgbfile;
            string dep_file=datapath+"/"+asso.depthfile;

            cv::Mat rgb=cv::imread(rgb_file);
            cv::Mat depth=cv::imread(dep_file,-1);

            if(rgb.empty()||depth.empty())
            {
                cerr<<"Failed to load file "<<rgb_file<<endl;
                break;
            }

            pi::timer.enter("AddFrame");
            fusionObj->addFrame(depth,fromSE3(asso.se3,cam->Fx(),cam->Fy(),cam->Cx(),cam->Cy()),
                                rgb,factor,10.0);
            pi::timer.leave("AddFrame");
            CameraRect* rect=new CameraRect(asso.se3,pi::gl::Color3b(255,255,255),1.5,1,rectLength);
            win3d->insert(rect);
            win3d->update();
            if(updateCount>0)
            {
                fusionObj->clear();
                updateCount=-10000;
                cerr<<"Fusion cleared!\n";
            }
            else updateCount++;
            rate.sleep();
        }
        running=false;

    }
    OnlineFusionObject* fusionObj;
    pi::gl::Win3D* win3d;
    bool shouldstop,running;
};

int main(int argc,char** argv)
{
    svar.ParseMain(argc,argv);
    QApplication app(argc,argv);

    pi::gl::Win3D win3d;
    win3d.setBackgroundColor(QColor(255,255,255));
    win3d.setForegroundColor(QColor(255,255,255));

    OnlineFusionObject* fusionObj=new OnlineFusionObject(0,0,0,0.005,0.005);
    fusionObj->_colorEnabled=true;
    fusionObj->_displayMode=svar.GetInt("DisplayMode",2);
    fusionObj->setThreadMeshing(svar.GetInt("UseThread",1));

    win3d.insert(fusionObj);

    FusionThread runnable(&win3d,fusionObj);
    pi::Thread thread;
    thread.start(&runnable);

    win3d.show();
    glDisable(GL_LIGHTING);
    int ret= app.exec();
    runnable.shouldstop=true;
    while(runnable.running) usleep(1000);
    return 0;
}
