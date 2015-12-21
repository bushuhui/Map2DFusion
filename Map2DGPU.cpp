#include "Map2DGPU.h"
#include <gui/gl/glHelper.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <base/Svar/Svar.h>
#include <base/time/Global_Timer.h>
#include <gui/gl/SignalHandle.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_gl_interop.h>
#include "UtilGPU.cuh"

using namespace std;


/**

  __________max
  |    |    |
  |____|____|
  |    |    |
  |____|____|
 min
 */

Map2DGPU::Map2DGPUEle::~Map2DGPUEle()
{
    if(texName) pi::gl::Signal_Handle::instance().delete_texture(texName);
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffersARB(1, &pbo);
    if(img) cudaFree(img);
}

bool Map2DGPU::Map2DGPUPrepare::prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                                        const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    if(frames.size()==0||camera.w<=0||camera.h<=0||camera.fx==0||camera.fy==0)
    {
        cerr<<"Map2DGPU::Map2DGPUPrepare::prepare:Not valid prepare!\n";
        return false;
    }
    _camera=camera;_fxinv=1./camera.fx;_fyinv=1./camera.fy;
    _plane =plane;
    _frames=frames;
    for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=_frames.begin();it!=_frames.end();it++)
    {
        pi::SE3d& pose=it->second;
        pose=plane.inverse()*pose;//plane coordinate
    }
    return true;
}

bool Map2DGPU::Map2DGPUData::prepare(SPtr<Map2DGPUPrepare> prepared)
{
    if(_w||_h) return false;//already prepared
    {
        _max=pi::Point3d(-1e10,-1e10,-1e10);
        _min=-_max;
        for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=prepared->_frames.begin();
            it!=prepared->_frames.end();it++)
        {
            pi::SE3d& pose=it->second;
            pi::Point3d& t=pose.get_translation();
            _max.x=t.x>_max.x?t.x:_max.x;
            _max.y=t.y>_max.y?t.y:_max.y;
            _max.z=t.z>_max.z?t.z:_max.z;
            _min.x=t.x<_min.x?t.x:_min.x;
            _min.y=t.y<_min.y?t.y:_min.y;
            _min.z=t.z<_min.z?t.z:_min.z;
        }
        if(_min.z*_max.z<=0) return false;
        cout<<"Box:Min:"<<_min<<",Max:"<<_max<<endl;
    }
    //estimate w,h and bonding box
    {
        double minh;
        if(_min.z>0) minh=_min.z;
        else minh=-_max.z;
        pi::Point3d line=prepared->UnProject(pi::Point2d(prepared->_camera.w,prepared->_camera.h))
                -prepared->UnProject(pi::Point2d(0,0));
        double radius=0.5*minh*sqrt((line.x*line.x+line.y*line.y));
        _lengthPixel=2*radius/sqrt(prepared->_camera.w*prepared->_camera.w
                                   +prepared->_camera.h*prepared->_camera.h);
        _lengthPixel/=svar.GetDouble("Map2D.Scale",1);
        _lengthPixelInv=1./_lengthPixel;
        _min=_min-pi::Point3d(radius,radius,0);
        _max=_max+pi::Point3d(radius,radius,0);
        pi::Point3d center=0.5*(_min+_max);
        _min=2*_min-center;_max=2*_max-center;
        _eleSize=ELE_PIXELS*_lengthPixel;
        _eleSizeInv=1./_eleSize;
        {
            _w=ceil((_max.x-_min.x)/_eleSize);
            _h=ceil((_max.y-_min.y)/_eleSize);
            _max.x=_min.x+_eleSize*_w;
            _max.y=_min.y+_eleSize*_h;
            _data.resize(_w*_h);
        }
    }
    return true;
}

Map2DGPU::Map2DGPU(bool thread)
    :alpha(svar.GetInt("Map2D.Alpha",0)),
     _valid(false),_thread(thread)
{

    // Otherwise pick the device with highest Gflops/s
//    int devID = gpuGetMaxGflopsDeviceId();
}

bool Map2DGPU::prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    //insert frames
    SPtr<Map2DGPUPrepare> p(new Map2DGPUPrepare);
    SPtr<Map2DGPUData>    d(new Map2DGPUData);

    if(p->prepare(plane,camera,frames))
        if(d->prepare(p))
        {
            pi::WriteMutex lock(mutex);
            prepared=p;
            data=d;
            weightImage.release();
            if(_thread&&!isRunning())
                start();
            _valid=true;
            return true;
        }
    return false;
}

bool Map2DGPU::feed(cv::Mat img,const pi::SE3d& pose)
{
    if(!_valid) return false;
    SPtr<Map2DGPUPrepare> p;
    SPtr<Map2DGPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }
    std::pair<cv::Mat,pi::SE3d> frame(img,p->_plane.inverse()*pose);
    if(_thread)
    {
        pi::WriteMutex lock(p->mutexFrames);
        p->_frames.push_back(frame);
        if(p->_frames.size()>20) p->_frames.pop_front();
        return true;
    }
    else
    {
        return renderFrame(frame);
    }
}

bool Map2DGPU::renderFrame(const std::pair<cv::Mat,pi::SE3d>& frame)
{
    if(1)return renderFrameGPU(frame);
    SPtr<Map2DGPUPrepare> p;
    SPtr<Map2DGPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }
    if(frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3)
    {
        cerr<<"Map2DGPU::renderFrame: frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";
        return false;
    }
    // pose->pts
    std::vector<pi::Point2d>          imgPts;
    {
        imgPts.reserve(4);
        imgPts.push_back(pi::Point2d(0,0));
        imgPts.push_back(pi::Point2d(p->_camera.w,0));
        imgPts.push_back(pi::Point2d(0,p->_camera.h));
        imgPts.push_back(pi::Point2d(p->_camera.w,p->_camera.h));
    }
    vector<pi::Point2d> pts;
    pts.reserve(imgPts.size());
    pi::Point3d downLook(0,0,-1);
    if(frame.second.get_translation().z<0) downLook=pi::Point3d(0,0,1);
    for(int i=0;i<imgPts.size();i++)
    {
        pi::Point3d axis=frame.second.get_rotation()*p->UnProject(imgPts[i]);
        if(axis.dot(downLook)<0.4)
        {
            return false;
        }
        axis=frame.second.get_translation()
                -axis*(frame.second.get_translation().z/axis.z);
        pts.push_back(pi::Point2d(axis.x,axis.y));
    }
    // dest location?
    double xmin=pts[0].x;
    double xmax=xmin;
    double ymin=pts[0].y;
    double ymax=ymin;
    for(int i=1;i<pts.size();i++)
    {
        if(pts[i].x<xmin) xmin=pts[i].x;
        if(pts[i].y<ymin) ymin=pts[i].y;
        if(pts[i].x>xmax) xmax=pts[i].x;
        if(pts[i].y>ymax) ymax=pts[i].y;
    }
    if(xmin<d->min().x||xmax>d->max().x||ymin<d->min().y||ymax>d->max().y)
    {
        if(p!=prepared)//what if prepare called?
        {
            return false;
        }
        if(!spreadMap(xmin,ymin,xmax,ymax))
        {
            return false;
        }
        else
        {
            pi::ReadMutex lock(mutex);
            if(p!=prepared)//what if prepare called?
            {
                return false;
            }
            d=data;//new data
        }
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    if(xminInt<0||yminInt<0||xmaxInt>d->w()||ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
//        cerr<<"Map2DGPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        xmin=d->min().x+d->eleSize()*xminInt;
        ymin=d->min().y+d->eleSize()*yminInt;
        xmax=d->min().x+d->eleSize()*xmaxInt;
        ymax=d->min().y+d->eleSize()*ymaxInt;
    }
    // prepare dst image
    cv::Mat src;
    if(weightImage.empty()||weightImage.cols!=frame.first.cols||weightImage.rows!=frame.first.rows)
    {
        pi::WriteMutex lock(mutex);
        int w=frame.first.cols;
        int h=frame.first.rows;
        weightImage.create(h,w,CV_8UC4);
        pi::byte *p=(weightImage.data);
        float x_center=w/2;
        float y_center=h/2;
        float dis_max=sqrt(x_center*x_center+y_center*y_center);
        int weightType=svar.GetInt("Map2D.WeightType",0);
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
            {
                float dis=(i-y_center)*(i-y_center)+(j-x_center)*(j-x_center);
                dis=1-sqrt(dis)/dis_max;
                p[1]=p[2]=p[0]=0;
                if(0==weightType)
                    p[3]=dis*254.;
                else p[3]=dis*dis*254;
                if(p[3]<2) p[3]=2;
                p+=4;
            }
        src=weightImage.clone();
    }
    else
    {
        pi::ReadMutex lock(mutex);
        src=weightImage.clone();
    }
    pi::Array_<pi::byte,4> *psrc=(pi::Array_<pi::byte,4>*)src.data;
    pi::Array_<pi::byte,3> *pimg=(pi::Array_<pi::byte,3>*)frame.first.data;
//    float weight=(frame.second.get_rotation()*pi::Point3d(0,0,1)).dot(downLook);
    for(int i=0,iend=weightImage.cols*weightImage.rows;i<iend;i++)
    {
        *((pi::Array_<pi::byte,3>*)psrc)=*pimg;
//        psrc->data[3]*=weight;
        psrc++;
        pimg++;
    }

    if(svar.GetInt("ShowSRC",0))
    {
        cv::imshow("src",src);
    }

    cv::Mat dst((ymaxInt-yminInt)*ELE_PIXELS,(xmaxInt-xminInt)*ELE_PIXELS,src.type());

    std::vector<cv::Point2f>          imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0;i<imgPts.size();i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));
    }
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for(int i=0;i<imgPtsCV.size();i++)
    {
        destPoints.push_back(cv::Point2f((pts[i].x-xmin)*d->lengthPixelInv(),
                             (pts[i].y-ymin)*d->lengthPixelInv()));
    }

    cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);
    pi::timer.enter("cv::warpPerspective");
    cv::warpPerspective(src, dst, transmtx, dst.size(),cv::INTER_LINEAR);
    pi::timer.leave("cv::warpPerspective");

    if(svar.GetInt("ShowDST",0))
    {
        cv::imshow("dst",dst);
    }
    // apply dst to eles
    pi::timer.enter("Apply");
    std::vector<SPtr<Map2DGPUEle> > dataCopy=d->data();
    for(int x=xminInt;x<xmaxInt;x++)
        for(int y=yminInt;y<ymaxInt;y++)
        {
            SPtr<Map2DGPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get())
            {
                ele=d->ele(y*d->w()+x);
            }

            cv::Mat tmp=cv::Mat::zeros(ELE_PIXELS,ELE_PIXELS,CV_8UC4);

            if(ele->img)
            {
                pi::ReadMutex lock(ele->mutexData);
                cudaMemcpy(tmp.data, ele->img, ELE_PIXELS*ELE_PIXELS*sizeof(uchar4), cudaMemcpyDeviceToHost);

            }
            else
            {
                pi::WriteMutex lock(ele->mutexData);
                cudaMalloc((void**)&ele->img,sizeof(uchar4)*ELE_PIXELS*ELE_PIXELS);
            }
            if(0)
            {
                cv::imshow("img",tmp);
                int& pause=svar.GetInt("Pause");
                pause=1;
                while(pause) sleep(10);
            }
            pi::Array_<pi::byte,4> *eleP=(pi::Array_<pi::byte,4>*)tmp.data;
            pi::Array_<pi::byte,4> *dstP=(pi::Array_<pi::byte,4>*)dst.data;
            dstP+=(x-xminInt)*ELE_PIXELS+(y-yminInt)*ELE_PIXELS*dst.cols;
            int skip=dst.cols-tmp.cols;
            for(int eleY=0;eleY<ELE_PIXELS;eleY++,dstP+=skip)
                for(int eleX=0;eleX<ELE_PIXELS;eleX++,dstP++,eleP++)
                {
                    if(eleP->data[3]<dstP->data[3])
                        *eleP=*dstP;
                }

            {
                pi::WriteMutex lock(ele->mutexData);
                cudaMemcpy(ele->img, tmp.data, ELE_PIXELS*ELE_PIXELS*sizeof(uchar4), cudaMemcpyHostToDevice);
                ele->Ischanged=true;
            }
        }
    pi::timer.leave("Apply");

    if(!svar.GetInt("Win3D.Enable"))//show result
    {
        cv::Mat result(ELE_PIXELS*d->h(),ELE_PIXELS*d->w(),CV_8UC4);
        cv::Mat tmp(ELE_PIXELS,ELE_PIXELS,CV_8UC4);
        for(int x=0;x<d->w();x++)
            for(int y=0;y<d->h();y++)
        {
            SPtr<Map2DGPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get()) continue;
            pi::ReadMutex lock(ele->mutexData);
            cudaMemcpy(tmp.data,ele->img,ELE_PIXELS*ELE_PIXELS*sizeof(uchar4),cudaMemcpyDeviceToHost);
            tmp.copyTo(result(cv::Rect(ELE_PIXELS*x,ELE_PIXELS*y,ELE_PIXELS,ELE_PIXELS)));
        }
        cv::resize(result,result,cv::Size(1000,result.rows*1000/result.cols));
        cv::imshow("img",result);
        cv::waitKey(0);
    }
    return true;
}

bool Map2DGPU::renderFrameGPU(const std::pair<cv::Mat,pi::SE3d>& frame)
{
    SPtr<Map2DGPUPrepare> p;
    SPtr<Map2DGPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }
    if(frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h
            ||frame.first.type()!=CV_8UC3)
    {
        cerr<<"Map2DGPU::renderFrame: frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";
        return false;
    }
    // pose->pts
    std::vector<pi::Point2d>          imgPts;
    {
        imgPts.reserve(4);
        imgPts.push_back(pi::Point2d(0,0));
        imgPts.push_back(pi::Point2d(p->_camera.w,0));
        imgPts.push_back(pi::Point2d(0,p->_camera.h));
        imgPts.push_back(pi::Point2d(p->_camera.w,p->_camera.h));
    }
    vector<pi::Point2d> pts;
    pts.reserve(imgPts.size());
    pi::Point3d downLook(0,0,-1);
    if(frame.second.get_translation().z<0) downLook=pi::Point3d(0,0,1);
    for(int i=0;i<imgPts.size();i++)
    {
        pi::Point3d axis=frame.second.get_rotation()*p->UnProject(imgPts[i]);
        if(axis.dot(downLook)<0.4)
        {
            return false;
        }
        axis=frame.second.get_translation()
                -axis*(frame.second.get_translation().z/axis.z);
        pts.push_back(pi::Point2d(axis.x,axis.y));
    }
    // dest location?
    double xmin=pts[0].x;
    double xmax=xmin;
    double ymin=pts[0].y;
    double ymax=ymin;
    for(int i=1;i<pts.size();i++)
    {
        if(pts[i].x<xmin) xmin=pts[i].x;
        if(pts[i].y<ymin) ymin=pts[i].y;
        if(pts[i].x>xmax) xmax=pts[i].x;
        if(pts[i].y>ymax) ymax=pts[i].y;
    }
    if(xmin<d->min().x||xmax>d->max().x||ymin<d->min().y||ymax>d->max().y)
    {
        if(p!=prepared)//what if prepare called?
        {
            return false;
        }
        if(!spreadMap(xmin,ymin,xmax,ymax))
        {
            return false;
        }
        else
        {
            pi::ReadMutex lock(mutex);
            if(p!=prepared)//what if prepare called?
            {
                return false;
            }
            d=data;//new data
        }
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    if(xminInt<0||yminInt<0||xmaxInt>d->w()||
            ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
//        cerr<<"Map2DGPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        xmin=d->min().x+d->eleSize()*xminInt;
        ymin=d->min().y+d->eleSize()*yminInt;
        xmax=d->min().x+d->eleSize()*xmaxInt;
        ymax=d->min().y+d->eleSize()*ymaxInt;
    }
    // prepare dst image
    std::vector<cv::Point2f>          imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0;i<imgPts.size();i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));
    }
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for(int i=0;i<imgPtsCV.size();i++)
    {
        destPoints.push_back(cv::Point2f((pts[i].x-xmin)*d->lengthPixelInv(),
                             (pts[i].y-ymin)*d->lengthPixelInv()));
    }

    cv::Mat inv = cv::getPerspectiveTransform( destPoints,imgPtsCV);
    inv.convertTo(inv,CV_32FC1);
    //warp and render with CUDA
    pi::timer.enter("Map2DGPU::UploadImage");
    CudaImage<uchar3> cudaFrame(frame.first.rows,frame.first.cols);
    cudaMemcpy(cudaFrame.data,frame.first.data,
               cudaFrame.cols*cudaFrame.rows*sizeof(uchar3),cudaMemcpyHostToDevice);
    pi::timer.leave("Map2DGPU::UploadImage");

    // apply dst to eles
    pi::timer.enter("Map2DGPU::Apply");
    std::vector<SPtr<Map2DGPUEle> > dataCopy=d->data();
    pi::Point3d translation=frame.second.get_translation();
    int cenX=(translation.x-d->min().x)*d->lengthPixelInv();
    int cenY=(translation.y-d->min().y)*d->lengthPixelInv();
    if(svar.GetInt("Map2DGPU.RenderElesTogether",1))
    {
        int w=xmaxInt-xminInt;
        int h=ymaxInt-yminInt;
        int wh=w*h;
        uchar4** out_datas=new uchar4*[wh];
        bool* freshs=new bool[wh];
        float* invs=new float[wh*9];
        for(int x=xminInt,i=0;x<xmaxInt;x++,i++)
            for(int y=yminInt,j=0;y<ymaxInt;y++,j++)
            {
//                int centerX=cenX-x*ELE_PIXELS;
//                int centerY=cenY-y*ELE_PIXELS;
                int idx=(i+j*w);
                cv::Mat trans=cv::Mat::eye(3,3,CV_32FC1);
                trans.at<float>(2)=i*ELE_PIXELS;
                trans.at<float>(5)=j*ELE_PIXELS;
                trans=inv*trans;
                trans.convertTo(trans,CV_32FC1);
                memcpy(invs+idx*9,trans.data,sizeof(float)*9);

                SPtr<Map2DGPUEle> ele=dataCopy[y*d->w()+x];
                bool fresh=false;
                if(!ele.get())
                {
                    ele=d->ele(y*d->w()+x);
                }
                {
                    ele->mutexData.lock();
                    if(!ele->img)
                    {
                        cudaMalloc((void**)&ele->img,sizeof(uchar4)*ELE_PIXELS*ELE_PIXELS);
                        fresh=true;
                    }
                    out_datas[idx]=ele->img;
                    freshs[idx]=fresh;

                }
            }

        pi::timer.enter("RenderKernal");
        renderFramesCaller(cudaFrame,ELE_PIXELS,ELE_PIXELS,
                                out_datas,freshs,
                                invs,cenX,cenY,wh);
        pi::timer.leave("RenderKernal");
        for(int x=xminInt,i=0;x<xmaxInt;x++,i++)
            for(int y=yminInt,j=0;y<ymaxInt;y++,j++)
            {
                SPtr<Map2DGPUEle> ele=dataCopy[y*d->w()+x];
                if(!ele.get())
                {
                    ele=d->ele(y*d->w()+x);
                }
                ele->mutexData.unlock();
                ele->Ischanged=true;
            }
        delete[] out_datas;
        delete[] freshs;
        delete[] invs;
    }
    else
    {
        for(int x=xminInt;x<xmaxInt;x++)
            for(int y=yminInt;y<ymaxInt;y++)
            {
                int centerX=cenX-x*ELE_PIXELS;
                int centerY=cenY-y*ELE_PIXELS;
                cv::Mat trans=cv::Mat::eye(3,3,CV_32FC1);
                trans.at<float>(2)=(x-xminInt)*ELE_PIXELS;
                trans.at<float>(5)=(y-yminInt)*ELE_PIXELS;
                trans=inv*trans;
                trans.convertTo(trans,CV_32FC1);
                SPtr<Map2DGPUEle> ele=dataCopy[y*d->w()+x];
                bool fresh=false;
                if(!ele.get())
                {
                    ele=d->ele(y*d->w()+x);
                }
                {
                    pi::WriteMutex lock(ele->mutexData);
                    if(!ele->img)
                    {
                        cudaMalloc((void**)&ele->img,sizeof(uchar4)*ELE_PIXELS*ELE_PIXELS);
                        fresh=true;
                    }
                    CudaImage<uchar4> cudaEle(ELE_PIXELS,ELE_PIXELS,ele->img);
                    cudaEle.fresh=fresh;
                    pi::timer.enter("RenderKernal");
                    renderFrameCaller(cudaFrame,cudaEle,
                                      (float*)trans.data,centerX,centerY);
                    pi::timer.leave("RenderKernal");
                    ele->Ischanged=true;
                }

            }
    }

    pi::timer.leave("Map2DGPU::Apply");

    if(!svar.GetInt("Win3D.Enable"))//show result
    {
        cv::Mat result(ELE_PIXELS*d->h(),ELE_PIXELS*d->w(),CV_8UC4);
        cv::Mat tmp(ELE_PIXELS,ELE_PIXELS,CV_8UC4);
        for(int x=0;x<d->w();x++)
            for(int y=0;y<d->h();y++)
        {
            SPtr<Map2DGPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get()) continue;
            pi::ReadMutex lock(ele->mutexData);
            cudaMemcpy(tmp.data,ele->img,ELE_PIXELS*ELE_PIXELS*sizeof(uchar4),cudaMemcpyDeviceToHost);
            tmp.copyTo(result(cv::Rect(ELE_PIXELS*x,ELE_PIXELS*y,ELE_PIXELS,ELE_PIXELS)));
        }
        cv::resize(result,result,cv::Size(1000,result.rows*1000/result.cols));
        cv::imshow("img",result);
        cv::waitKey(0);
    }
    return true;
}


bool Map2DGPU::spreadMap(double xmin,double ymin,double xmax,double ymax)
{
    pi::timer.enter("Map2DGPU::spreadMap");
    SPtr<Map2DGPUData> d;
    {
        pi::ReadMutex lock(mutex);
        d=data;
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    xminInt=min(xminInt,0); yminInt=min(yminInt,0);
    xmaxInt=max(xmaxInt,d->w()); ymaxInt=max(ymaxInt,d->h());
    int w=xmaxInt-xminInt;
    int h=ymaxInt-yminInt;
    pi::Point2d min,max;
    {
        min.x=d->min().x+d->eleSize()*xminInt;
        min.y=d->min().y+d->eleSize()*yminInt;
        max.x=min.x+w*d->eleSize();
        max.y=min.y+h*d->eleSize();
    }
    std::vector<SPtr<Map2DGPUEle> > dataOld=d->data();
    std::vector<SPtr<Map2DGPUEle> > dataCopy;
    dataCopy.resize(w*h);
    {
        for(int x=0,xend=d->w();x<xend;x++)
            for(int y=0,yend=d->h();y<yend;y++)
            {
                dataCopy[x-xminInt+(y-yminInt)*w]=dataOld[y*d->w()+x];
            }
    }
    //apply
    {
        pi::WriteMutex lock(mutex);
        data=SPtr<Map2DGPUData>(new Map2DGPUData(d->eleSize(),d->lengthPixel(),
                                                 pi::Point3d(max.x,max.y,d->max().z),
                                                 pi::Point3d(min.x,min.y,d->min().z),
                                                 w,h,dataCopy));
    }
    pi::timer.leave("Map2DGPU::spreadMap");
    return true;
}

bool Map2DGPU::getFrame(std::pair<cv::Mat,pi::SE3d>& frame)
{
    pi::ReadMutex lock(mutex);
    pi::ReadMutex lock1(prepared->mutexFrames);
    if(prepared->_frames.size())
    {
        frame=prepared->_frames.front();
        prepared->_frames.pop_front();
        return true;
    }
    else return false;
}

void Map2DGPU::run()
{
    std::pair<cv::Mat,pi::SE3d> frame;
    while(!shouldStop())
    {
        if(_valid)
        {
            if(getFrame(frame))
            {
                pi::timer.enter("Map2DGPU::renderFrame");
                renderFrame(frame);
                pi::timer.leave("Map2DGPU::renderFrame");
            }
        }
        sleep(10);
    }
}

void Map2DGPU::draw()
{
    if(!_valid) return;
    static bool inited=false;
    if(!inited)
    {
        cudaGLSetGLDevice(0);
        glewInit();
    }

    SPtr<Map2DGPUPrepare> p;
    SPtr<Map2DGPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrix(p->_plane);
    //draw deque frames
    pi::TicTac ticTac;
    ticTac.Tic();
    {
        std::deque<std::pair<cv::Mat,pi::SE3d> > frames=p->getFrames();
        glDisable(GL_LIGHTING);
        glBegin(GL_LINES);
        for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=frames.begin();it!=frames.end();it++)
        {
            pi::SE3d& pose=it->second;
            glColor3ub(255,0,0);
            glVertex(pose.get_translation());
            glVertex(pose*pi::Point3d(1,0,0));
            glColor3ub(0,255,0);
            glVertex(pose.get_translation());
            glVertex(pose*pi::Point3d(0,1,0));
            glColor3ub(0,0,255);
            glVertex(pose.get_translation());
            glVertex(pose*pi::Point3d(0,0,1));
        }
        glEnd();
    }
    //draw global area
    {
        pi::Point3d _min=d->min();
        pi::Point3d _max=d->max();
        glColor3ub(255,0,0);
        glBegin(GL_LINES);
        glVertex3d(_min.x,_min.y,0);
        glVertex3d(_min.x,_max.y,0);
        glVertex3d(_min.x,_min.y,0);
        glVertex3d(_max.x,_min.y,0);
        glVertex3d(_max.x,_min.y,0);
        glVertex3d(_max.x,_max.y,0);
        glVertex3d(_min.x,_max.y,0);
        glVertex3d(_max.x,_max.y,0);
        glEnd();
    }

    //draw textures
    if(0)
    {
        glEnable(GL_TEXTURE_2D);
        glEnable(GL_BLEND);
        //    glEnable(GL_LIGHTING);
        if(alpha)
        {
            glEnable(GL_ALPHA_TEST);
            glAlphaFunc(GL_GREATER, 0.1f);
            glBlendFunc(GL_SRC_ALPHA,GL_ONE);
        }
        GLint last_texture_ID;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);
        std::vector<SPtr<Map2DGPUEle> > dataCopy=d->data();
        int wCopy=d->w(),hCopy=d->h();
        glColor3ub(255,255,255);
        size_t num_bytes=ELE_PIXELS*ELE_PIXELS*sizeof(uchar4);
        for(int x=0;x<wCopy;x++)
            for(int y=0;y<hCopy;y++)
            {
                int idxData=y*wCopy+x;
                float x0=d->min().x+x*d->eleSize();
                float y0=d->min().y+y*d->eleSize();
                float x1=x0+d->eleSize();
                float y1=y0+d->eleSize();
                SPtr<Map2DGPUEle> ele=dataCopy[idxData];
                if(!ele.get())  continue;
                if(!ele->img) continue;
                if(ele->Ischanged&&ticTac.Tac()<0.02)
                {
                    pi::timer.enter("glTexImage2D");
                    pi::ReadMutex lock1(ele->mutexData);

                    if(ele->pbo==0)// bind pbo with cuda_pbo_resource
                    {
                        glGenBuffers(1, &ele->pbo);
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ele->pbo);
                        glBufferData(GL_PIXEL_UNPACK_BUFFER, num_bytes, NULL, GL_DYNAMIC_COPY);
                        cudaGraphicsGLRegisterBuffer(&ele->cuda_pbo_resource, ele->pbo, cudaGraphicsMapFlagsWriteDiscard);
                    }

                    if(ele->texName==0)// create texture
                    {
                        glGenTextures(1, &ele->texName);
                        glBindTexture(GL_TEXTURE_2D,ele->texName);

                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ELE_PIXELS, ELE_PIXELS,
                                     0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
                    }

                    //flush data from ele->img to ele->cuda_pbo_resource
                    if(0)
                    {
                        cudaGraphicsMapResources(1, &ele->cuda_pbo_resource, 0);
                        cudaGraphicsResourceGetMappedPointer((void **)&ele->img, &num_bytes, ele->cuda_pbo_resource);
                        cudaGraphicsUnmapResources(1, &ele->cuda_pbo_resource, 0);
                    }

                    //flush buffer to texture
                    //                {
                    //                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ele->pbo);
                    //                    glBindTexture(GL_TEXTURE_2D,ele->texName);

                    //                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ELE_PIXELS, ELE_PIXELS,/*window_width, window_height,*/
                    //                                    GL_BGRA, GL_UNSIGNED_BYTE, NULL);
                    //                }
                    ele->Ischanged=false;
                    pi::timer.leave("glTexImage2D");
                }

                // draw things
                //flush buffer to texture
                {
                    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ele->pbo);
                    glBindTexture(GL_TEXTURE_2D,ele->texName);

                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ELE_PIXELS, ELE_PIXELS,/*window_width, window_height,*/
                                    GL_BGRA, GL_UNSIGNED_BYTE, NULL);
                }
                glBegin(GL_QUADS);
                glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0,0);
                glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y1,0);
                glTexCoord2f(1.0f, 1.0f); glVertex3f(x1,y1,0);
                glTexCoord2f(1.0f, 0.0f); glVertex3f(x1,y0,0);
                glEnd();
            }
        glBindTexture(GL_TEXTURE_2D, last_texture_ID);
    }
    glPopMatrix();
}

bool Map2DGPU::save(const std::string& filename)
{
    // determin minmax
//    pi::Point2i minInt(1e6,1e6),maxInt(-1e6,-1e6);

//    std::vector<SPtr<Map2DGPUEle> > dataCopy;
//    int wCopy,hCopy;
//    {
//        pi::ReadMutex lock(mutexData);
//        wCopy=_w;hCopy=_h;
//        dataCopy=data;
//    }
//    for(int x=0;x<wCopy;x++)
//        for(int y=0;y<hCopy;y++)
//        {
//            SPtr<Map2DGPUEle> ele=dataCopy[wCopy*y+x];
//            if(!ele.get()) continue;
//            {
//                pi::ReadMutex lock(ele->mutexData);
//                if(ele->img.empty()) continue;
//            }
//            minInt.x=min(minInt.x,x); minInt.y=min(minInt.y,y);
//            maxInt.x=max(maxInt.x,x); maxInt.y=max(maxInt.y,y);
//        }
//    maxInt=maxInt+pi::Point2i(1,1);
//    pi::Point2i wh=maxInt-minInt;
//    cv::Mat result(wh.y*ELE_PIXELS,wh.x*ELE_PIXELS,CV_8UC4);
    return false;
}
