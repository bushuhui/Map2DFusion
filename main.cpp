#include <iostream>
#include <fstream>
#include <base/Svar/Svar.h>
#include <base/Svar/VecParament.h>
#include <base/time/Global_Timer.h>
#include <gui/gl/Win3D.h>
#include <Map2D.h>
#include <opencv2/highgui/highgui.hpp>

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
        default:
            break;
        }
    }

    int testBufferObject()
    {

    }

    bool obtainFrame(std::pair<cv::Mat,pi::SE3d>& frame)
    {
        string line;
        if(!getline(in,line)) return false;
        stringstream ifs(line);
        string imgfile;
        ifs>>imgfile;
        imgfile=datapath+"/"+imgfile+".png";
        frame.first=cv::imread(imgfile);
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
        if(!svar.ParseFile(datapath+"/config.cfg"))
        {
            cerr<<"Can't open file "<<(datapath+"/trajectory.txt")<<endl;
            return -2;
        }

        in.open((datapath+"/trajectory.txt").c_str());
        if(!in.is_open())
        {
            cerr<<"Can't open file "<<(datapath+"/trajectory.txt")<<endl;
            return -3;
        }

        deque<std::pair<cv::Mat,pi::SE3d> > frames;
        for(int i=0;i<100;i++)
        {
            std::pair<cv::Mat,pi::SE3d> frame;
            if(!obtainFrame(frame)) break;
            frames.push_back(frame);
        }
        cout<<"Loaded "<<frames.size()<<" frames.\n";

        if(!frames.size()) return -4;

        map=Map2D::create(svar.GetInt("Map2D.Type",Map2D::TypeGPU));
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
        }
        else
        {
            int& needStop=svar.GetInt("ShouldStop");
            while(!needStop) sleep(20);
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
    SPtr<pi::gl::Win3D> win3d;
    ifstream      in;
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
