#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#include <base/Svar/Svar.h>
#include <base/Svar/VecParament.h>
#include <base/time/Global_Timer.h>
#include <gui/gl/Win3D.h>

#include "Map2D.h"

using namespace std;

class TestSystem:public pi::Thread,public pi::gl::EventHandle
{
public:
    TestSystem()
    {
        if(svar.GetInt("Win3D.Enable",1))
        {
            win3d=SPtr<pi::gl::Win3D>(new pi::gl::Win3D());
        }

    }
    ~TestSystem()
    {
        if(map.get())
            map->save(svar.GetString("Map.File2Save","result.png"));
        map=SPtr<Map2D>();
        win3d=SPtr<pi::gl::Win3D>();
    }

    virtual bool KeyPressHandle(void* arg)
    {
        QKeyEvent* e=(QKeyEvent*)arg;
        switch (e->key()) {
        case Qt::Key_I:
        {
            std::pair<cv::Mat,pi::SE3d> frame;
            if(obtainFrame(frame))
            {
                pi::timer.enter("Map2D::feed");
                map->feed(frame.first,frame.second);
                if(win3d.get()&&tictac.Tac()>0.033)
                {
                    tictac.Tic();
                    win3d->update();
                }
                pi::timer.leave("Map2D::feed");
            }
        }
            break;
        case Qt::Key_P:
        {
            int& pause=svar.GetInt("Pause");
            pause=!pause;
        }
            break;
        case Qt::Key_Escape:
        {
            stop();
            return false;
        }
            break;
        default:
            return false;
            break;
        }
        return false;
    }

    int testBufferObject()
    {

    }

    bool obtainFrame(std::pair<cv::Mat,pi::SE3d>& frame)
    {
        string line;
        if(!getline(*in,line)) return false;
        stringstream ifs(line);
        string imgfile;
        ifs>>imgfile;
        imgfile=datapath+"/"+imgfile+".png";
        pi::timer.enter("obtainFrame");
        frame.first=cv::imread(imgfile);
        pi::timer.leave("obtainFrame");
        if(frame.first.empty()) return false;
        ifs>>frame.second;
        return true;
    }

    int testMap2D()
    {
        cout<<"Act=TestMap2D\n";
        datapath=svar.GetString("Map2D.DataPath","");
        if(!datapath.size())
        {
            cerr<<"Map2D.DataPath is not seted!\n";
            return -1;
        }
        svar.ParseFile(datapath+"/config.cfg");
        if(!svar.exist("Plane"));
        {
//            cerr<<"Plane is not defined!\n";
//            return -2;
        }

        if(!in.get())
            in=SPtr<ifstream>(new ifstream((datapath+"/trajectory.txt").c_str()));

        if(!in->is_open())
        {
            cerr<<"Can't open file "<<(datapath+"/trajectory.txt")<<endl;
            return -3;
        }
        deque<std::pair<cv::Mat,pi::SE3d> > frames;
        for(int i=0,iend=svar.GetInt("PrepareFrameNum",10);i<iend;i++)
        {
            std::pair<cv::Mat,pi::SE3d> frame;
            if(!obtainFrame(frame)) break;
            frames.push_back(frame);
        }
        cout<<"Loaded "<<frames.size()<<" frames.\n";

        if(!frames.size()) return -4;

        map=Map2D::create(svar.GetInt("Map2D.Type",Map2D::TypeGPU),
                          svar.GetInt("Map2D.Thread",true));
        if(!map.get())
        {
            cerr<<"No map2d created!\n";
            return -5;
        }
        VecParament vecP=svar.get_var("Camera.Paraments",VecParament());
        if(vecP.size()!=6)
        {
            cerr<<"Invalid camera parameters!\n";
            return -5;
        }
        map->prepare(svar.get_var<pi::SE3d>("Plane",pi::SE3d()),
                     PinHoleParameters(vecP[0],vecP[1],vecP[2],vecP[3],vecP[4],vecP[5]),
                    frames);

        if(win3d.get())
        {
            win3d->SetEventHandle(this);
            win3d->insert(map);
            win3d->setSceneRadius(1000);
            win3d->Show();
            tictac.Tic();
        }
        else
        {
            int& needStop=svar.GetInt("ShouldStop");
            while(!needStop) sleep(20);
        }

        if(svar.GetInt("AutoFeedFrames",1))
        {
            pi::Rate rate(100);
            while(!shouldStop())
            {
                if(map->queueSize()<2)
                {
                    std::pair<cv::Mat,pi::SE3d> frame;
                    if(!obtainFrame(frame)) break;
                    map->feed(frame.first,frame.second);
                }
                if(win3d.get()&&tictac.Tac()>0.033)
                {
                    tictac.Tic();
                    win3d->update();
                }
                rate.sleep();
            }
        }
    }

    virtual void run()
    {
        string act=svar.GetString("Act","Default");
        if(act=="TestBufferObject") testBufferObject();
        else if(act=="TestMap2D"||act=="Default") testMap2D();
        else cout<<"No act "<<act<<"!\n";
    }

    string        datapath;
    pi::TicTac    tictac;
    SPtr<pi::gl::Win3D> win3d;
    SPtr<ifstream>      in;
    SPtr<Map2D>   map;
};

int main(int argc,char** argv)
{
    svar.ParseMain(argc,argv);

    if(svar.GetInt("Win3D.Enable",0))
    {
        QApplication app(argc,argv);
        TestSystem sys;
        sys.start();
        return app.exec();
    }
    else
    {
        TestSystem sys;
        sys.run();
    }
}
