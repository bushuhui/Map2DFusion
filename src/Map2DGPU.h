#ifndef MAP2DGPU_H
#define MAP2DGPU_H
#include "Map2D.h"
#include "Map2DCPU.h"
#ifdef HAS_CUDA

#include <cuda_runtime.h>

class Map2DGPU:public Map2D,public pi::Thread
{
    struct Map2DGPUPrepare//change when prepare
    {
        uint queueSize(){pi::ReadMutex lock(mutexFrames);
                      return _frames.size();}

        bool prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                     const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames);

        pi::Point2d Project(const pi::Point3d& pt)
        {
            double zinv=1./pt.z;
            return pi::Point2d(_camera.fx*pt.x*zinv+_camera.cx,
                               _camera.fy*pt.y*zinv+_camera.cy);
        }

        pi::Point3d UnProject(const pi::Point2d& pt)
        {
            return pi::Point3d((pt.x-_camera.cx)*_fxinv,
                               (pt.y-_camera.cy)*_fyinv,1.);
        }

        std::deque<std::pair<cv::Mat,pi::SE3d> > getFrames()
        {
            pi::ReadMutex lock(mutexFrames);
            return _frames;
        }

        PinHoleParameters                        _camera;
        double                                   _fxinv,_fyinv;
        pi::SE3d                                 _plane;//all fixed
        std::deque<std::pair<cv::Mat,pi::SE3d> > _frames;//camera coordinate
        pi::MutexRW                              mutexFrames;
    };

    struct Map2DGPUEle
    {
        Map2DGPUEle()
            :img(NULL),cuda_pbo_resource(NULL),
             pbo(0),texName(0),Ischanged(false)
        {
        }
        ~Map2DGPUEle();

        bool updateTextureGPU();
        bool updateTextureCPU();

        float4* img;//BGRA
        cudaGraphicsResource *cuda_pbo_resource;

        uint    pbo;
        uint    texName;

        bool    Ischanged;
        pi::MutexRW mutexData;
    };

    struct Map2DGPUData//change when spread and prepare
    {
        Map2DGPUData():_w(0),_h(0){}
        Map2DGPUData(double eleSize_,double lengthPixel_,pi::Point3d max_,pi::Point3d min_,
                     int w_,int h_,const std::vector<SPtr<Map2DGPUEle> >& d_)
            :_eleSize(eleSize_),_eleSizeInv(1./eleSize_),
              _lengthPixel(lengthPixel_),_lengthPixelInv(1./lengthPixel_),
              _min(min_),_max(max_),_w(w_),_h(h_),_data(d_){}

        bool   prepare(SPtr<Map2DGPUPrepare> prepared);// only done Once!

        double eleSize()const{return _eleSize;}
        double lengthPixel()const{return _lengthPixel;}
        double eleSizeInv()const{return _eleSizeInv;}
        double lengthPixelInv()const{return _lengthPixelInv;}
        const pi::Point3d& min()const{return _min;}
        const pi::Point3d& max()const{return _max;}
        const int w()const{return _w;}
        const int h()const{return _h;}

        std::vector<SPtr<Map2DGPUEle> > data()
        {pi::ReadMutex lock(mutexData);return _data;}

        SPtr<Map2DGPUEle> ele(uint idx)
        {
            pi::WriteMutex lock(mutexData);
            if(idx>_data.size()) return SPtr<Map2DGPUEle>();
            else if(!_data[idx].get())
            {
                _data[idx]=SPtr<Map2DGPUEle>(new Map2DGPUEle());
            }
            return _data[idx];
        }

    private:
        //IMPORTANT: everything should never changed after prepared!
        double      _eleSize,_lengthPixel,_eleSizeInv,_lengthPixelInv;
        pi::Point3d _max,_min;
        int         _w,_h;
        std::vector<SPtr<Map2DGPUEle> >  _data;
        pi::MutexRW mutexData;
    };

public:

    Map2DGPU(bool thread=true);

    virtual ~Map2DGPU(){_valid=false;}

    virtual bool prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                    const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames);

    virtual bool feed(cv::Mat img,const pi::SE3d& pose);//world coordinate

    virtual void draw();

    virtual bool save(const std::string& filename);

    virtual uint queueSize(){
        if(prepared.get()) return prepared->queueSize();
        else               return 0;
    }

    virtual void run();

private:

    bool getFrame(std::pair<cv::Mat,pi::SE3d>& frame);
    bool renderFrame(const std::pair<cv::Mat,pi::SE3d>& frame);
    bool spreadMap(double xmin,double ymin,double xmax,double ymax);


    //source
    SPtr<Map2DGPUPrepare>             prepared;
    SPtr<Map2DGPUData>                data;
    pi::MutexRW                       mutex;

    bool                              _valid,_thread,_changed;
    cv::Mat                           weightImage;
    int&                              alpha;
};

#endif // HAS_GPU
#endif // MAP2DGPU_H
