#ifndef MAP2D_H
#define MAP2D_H
#include <deque>
#include <opencv2/features2d/features2d.hpp>

#include <base/types/SPtr.h>
#include <base/types/SE3.h>
#include <gui/gl/GL_Object.h>


struct PinHoleParameters
{
    PinHoleParameters(){}
    PinHoleParameters(int _w,int _h,double _fx,double _fy,double _cx,double _cy)
        :w(_w),h(_h),fx(_fx),fy(_fy),cx(_cx),cy(_cy){}
    double w,h,fx,fy,cx,cy;
};

class Map2D:public pi::gl::GL_Object
{
public:
    enum Map2DType{TypeCPU=0,TypeRender=1,NoType=2,TypeGPU=3};
    static SPtr<Map2D> create(int type=TypeCPU,bool thread=true);

    virtual ~Map2D(){}

    virtual bool prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                    const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames){return false;}

    virtual bool feed(cv::Mat img,const pi::SE3d& pose){return false;}

    virtual void draw(){}

    virtual bool save(const std::string& filename){return false;}

    virtual uint queueSize(){return 0;}
};

#endif // MAP2D_H
