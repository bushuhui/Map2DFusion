#include "Map2DRender.h"
#include <gui/gl/glHelper.h>
#include <GL/gl.h>
#include <base/Svar/Svar.h>
#include <base/time/Global_Timer.h>
#include <gui/gl/SignalHandle.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include "UtilGPU.h"

using namespace std;


/**

  __________max
  |    |    |
  |____|____|
  |    |    |
  |____|____|
 min
 */

Map2DRender::Map2DRenderEle::~Map2DRenderEle()
{
//    if(texName) pi::gl::Signal_Handle::instance().delete_texture(texName);
}

bool Map2DRender::Map2DRenderPrepare::prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                                        const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    if(frames.size()==0||camera.w<=0||camera.h<=0||camera.fx==0||camera.fy==0)
    {
        cerr<<"Map2DRender::Map2DRenderPrepare::prepare:Not valid prepare!\n";
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

bool Map2DRender::Map2DRenderData::prepare(SPtr<Map2DRenderPrepare> prepared)
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

Map2DRender::Map2DRender(bool thread)
    :alpha(svar.GetInt("Map2D.Alpha",0)),
     _valid(false),_thread(thread)
{
}

bool Map2DRender::prepare(const pi::SE3d& plane,const PinHoleParameters& camera,
                const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    //insert frames
    SPtr<Map2DRenderPrepare> p(new Map2DRenderPrepare);
    SPtr<Map2DRenderData>    d(new Map2DRenderData);

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

bool Map2DRender::feed(cv::Mat img,const pi::SE3d& pose)
{
    if(!_valid) return false;
    SPtr<Map2DRenderPrepare> p;
    SPtr<Map2DRenderData>    d;
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

bool Map2DRender::getFrame(std::pair<cv::Mat,pi::SE3d>& frame)
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

bool Map2DRender::renderFrame(const std::pair<cv::Mat,pi::SE3d>& frame)
{
    return false;
}

bool Map2DRender::getFrames(std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    pi::ReadMutex lock(mutex);
    pi::ReadMutex lock1(prepared->mutexFrames);
    if(prepared->_frames.size())
    {
        frames=prepared->_frames;
        prepared->_frames.clear();
        return true;
    }
    else return false;
}

bool Map2DRender::renderFrames(std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    // 0. Prepare things
    SPtr<Map2DRenderPrepare> p;
    SPtr<Map2DRenderData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }

    std::vector<pi::Point2d>  imgPts;
    {
        imgPts.reserve(4);
        imgPts.push_back(pi::Point2d(0,0));
        imgPts.push_back(pi::Point2d(p->_camera.w,0));
        imgPts.push_back(pi::Point2d(0,p->_camera.h));
        imgPts.push_back(pi::Point2d(p->_camera.w,p->_camera.h));
    }

    std::vector<cv::Point2f>          imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0;i<imgPts.size();i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));
    }

    {
        pi::WriteMutex lock(mutex);
        if(weightImage.empty())
        {
            int w=p->_camera.w;
            int h=p->_camera.h;
            weightImage=cv::Mat(h,w,CV_8UC1,cv::Scalar(255));
            if(0)
            {
                pi::byte *p=(weightImage.data);
                float x_center=w*0.5;
                float y_center=h*0.5;
                float dis_maxInv=1./sqrt(x_center*x_center+y_center*y_center);
                for(int i=0;i<h;i++)
                    for(int j=0;j<w;j++,p++)
                    {
                        float dis=(i-y_center)*(i-y_center)+(j-x_center)*(j-x_center);
                        dis=1-sqrt(dis)*dis_maxInv;
                        *p=dis*dis*254;
                        if(*p<1) *p=1;
                    }
            }
        }
    }
    // 1. Unproject frames to the plane and warp the images while update the area
    std::vector<cv::Mat>        imgwarped(frames.size());
    std::vector<cv::Mat>        maskwarped(frames.size());
    std::vector<cv::Point2f>    cornersWorld(frames.size());
    std::vector<cv::Size>       sizes(frames.size());

    pi::Point2d min,max;
    int idx=0;
    std::vector<pi::Point2d>  planePts=imgPts;
    pi::Point3d downLook(0,0,-1);
    for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=frames.begin();
        it<frames.end();it++,idx++)
    {
        cv::Mat& img=it->first;
        pi::SE3d& pose=it->second;
        pi::Point2d curMin(1e6,1e6),curMax(-1e6,-1e6);
        bool bOK=true;
        if(pose.get_translation().z<0) downLook=pi::Point3d(0,0,1);
        else downLook=pi::Point3d(0,0,-1);
        for(int j=0;j<imgPts.size();j++)
        {
            pi::Point3d axis=pose.get_rotation()*p->UnProject(imgPts[j]);
            if(axis.dot(downLook)<0.4)
            {
                bOK=false;break;
            }
            axis=pose.get_translation()-axis*(pose.get_translation().z/axis.z);
            planePts[j]=pi::Point2d(axis.x,axis.y);
        }
        if(!bOK)
        {
            continue;
        }

        for(int i=0;i<planePts.size();i++)
        {
            if(planePts[i].x<curMin.x) curMin.x=planePts[i].x;
            if(planePts[i].y<curMin.y) curMin.y=planePts[i].y;
            if(planePts[i].x>curMax.x) curMax.x=planePts[i].x;
            if(planePts[i].y>curMax.y) curMax.y=planePts[i].y;
        }
        {
            if(curMin.x<min.x) min.x=curMin.x;
            if(curMin.y<min.y) min.y=curMin.y;
            if(curMax.x>max.x) max.x=curMax.x;
            if(curMax.y>max.y) max.y=curMax.y;
        }
        cornersWorld[idx]=cv::Point2f(curMin.x,curMin.y);
        sizes[idx]=cv::Size((curMax.x-curMin.x)*d->lengthPixelInv(),
                          (curMax.y-curMin.y)*d->lengthPixelInv());

        std::vector<cv::Point2f> destPoints;
        destPoints.reserve(imgPtsCV.size());
        for(int i=0;i<imgPtsCV.size();i++)
        {
            destPoints.push_back(cv::Point2f((planePts[i].x-curMin.x)*d->lengthPixelInv(),
                                             (planePts[i].y-curMin.y)*d->lengthPixelInv()));
        }
        cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints);
//        cv::warpPerspective(img, imgwarped[idx], transmtx, sizes[idx],cv::INTER_LINEAR);
//        cv::warpPerspective(weightImage, maskwarped[idx], transmtx, sizes[idx],cv::INTER_LINEAR);
        UtilGPU::warpPerspective(img,imgwarped[idx],transmtx,sizes[idx]);
        UtilGPU::warpPerspective(weightImage, maskwarped[idx], transmtx, sizes[idx]);
        if(0)
        {
            cv::imshow("imgwarped",imgwarped[idx]);
            cv::imshow("maskwarped",maskwarped[idx]);
            cv::waitKey(0);
        }
    }

    // 2. spread the map and find seams of warped images
    if(min.x<d->min().x||min.y<d->min().y||max.x>d->max().x||max.y>d->max().y)
    {
        if(!spreadMap(min.x,min.y,max.x,max.y))
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

    int xminInt=floor((min.x-d->min().x)*d->eleSizeInv());
    int yminInt=floor((min.y-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((max.x-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((max.y-d->min().y)*d->eleSizeInv());

    if(xminInt<0||yminInt<0||xmaxInt>d->w()||ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
//        cerr<<"Map2DCPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        min.x=d->min().x+d->eleSize()*xminInt;
        min.y=d->min().y+d->eleSize()*yminInt;
        max.x=d->min().x+d->eleSize()*xmaxInt;
        max.y=d->min().y+d->eleSize()*ymaxInt;
    }
    std::vector<cv::Point> cornersImages(frames.size());
    for(int i=0;i<frames.size();i++)
    {
        if(imgwarped[i].empty())
        {
            cornersImages[i]=cv::Point(0,0);
            sizes[i]=cv::Size(0,0);
            continue;
        }
        cornersImages[i]=cv::Point((cornersWorld[i].x-min.x)*d->lengthPixelInv(),
                                   (cornersWorld[i].y-min.y)*d->lengthPixelInv());
    }
    if(svar.GetInt("Map2DRender.EnableSeam",1))//find seam
    {
        std::vector<cv::Mat>  seamwarped(frames.size());
        for(int i=0;i<maskwarped.size();i++)
        {
            seamwarped[i]=maskwarped[i].clone();
        }
        using namespace cv;
        using namespace cv::detail;
        string seam_find_type = "dp_colorgrad";
        cv::Ptr<cv::detail::SeamFinder> seam_finder;
        if (seam_find_type == "no")
            seam_finder = new cv::detail::NoSeamFinder();
        else if (seam_find_type == "voronoi")
            seam_finder = new cv::detail::VoronoiSeamFinder();
        else if (seam_find_type == "gc_color")
        {
    #ifdef HAVE_OPENCV_GPU
            if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
                seam_finder = new cv::detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
            else
    #endif
                seam_finder = new cv::detail::GraphCutSeamFinder(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
        }
        else if (seam_find_type == "gc_colorgrad")
        {
    #ifdef HAVE_OPENCV_GPU
            if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
                seam_finder = new cv::detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
            else
    #endif
                seam_finder = new cv::detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        }
        else if (seam_find_type == "dp_color")
            seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
        else if (seam_find_type == "dp_colorgrad")
            seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
        if (seam_finder.empty())
        {
            cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
            return 1;
        }

        seam_finder->find(imgwarped, cornersImages, seamwarped);
        int eleSize=3;
        Mat element = getStructuringElement( 0,Size( 2*eleSize + 1, 2*eleSize+1 ), Point(eleSize, eleSize ) );
        for(int i=0;i<seamwarped.size();i++)
        {
            if(imgwarped[i].empty()) continue;
            dilate(seamwarped[i], seamwarped[i], element);
            maskwarped[i]=seamwarped[i]&maskwarped[i];
        }
    }

    // 3. blender images
    cv::Mat result, result_mask;
    {
        using namespace cv;
        using namespace cv::detail;
        Ptr<Blender> blender;
        int blend_type = Blender::FEATHER;
        bool try_gpu = false;
        double blend_strength=5;

        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_gpu);
            Size dst_sz = cv::Size((xmaxInt-xminInt)*ELE_PIXELS,(ymaxInt-yminInt)*ELE_PIXELS);
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_gpu);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
                fb->setSharpness(1.f/blend_width);
            }
            blender->prepare(Rect(0,0,dst_sz.width,dst_sz.height));
        }

        // Blend the current image
        for(int i=0;i<frames.size();i++)
        {
            if(!imgwarped[i].empty())
            {
                if(blend_type = Blender::FEATHER)
                    imgwarped[i].convertTo(imgwarped[i],CV_16SC3);
                blender->feed(imgwarped[i], maskwarped[i], cornersImages[i]);
                if(0)
                {
                    cv::imshow("imgwarped",imgwarped[i]);
                    cv::imshow("maskwarped",maskwarped[i]);
                    cv::waitKey(0);
                }
            }
        }
        blender->blend(result, result_mask);
    }
    if(1)
    {
        cv::imwrite("result.jpg",result);
        result.convertTo(result,CV_8U);
        cv::resize(result,result,cv::Size(1000,1000./result.cols*result.rows));
        cv::imshow("result",result);
        cv::imshow("result_mask",result_mask);
        cv::waitKey(0);
    }
    stop();
    return true;
}


bool Map2DRender::spreadMap(double xmin,double ymin,double xmax,double ymax)
{
    pi::timer.enter("Map2DRender::spreadMap");
    SPtr<Map2DRenderData> d;
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
    std::vector<SPtr<Map2DRenderEle> > dataOld=d->data();
    std::vector<SPtr<Map2DRenderEle> > dataCopy;
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
        data=SPtr<Map2DRenderData>(new Map2DRenderData(d->eleSize(),d->lengthPixel(),
                                                 pi::Point3d(max.x,max.y,d->max().z),
                                                 pi::Point3d(min.x,min.y,d->min().z),
                                                 w,h,dataCopy));
    }
    pi::timer.leave("Map2DRender::spreadMap");
    return true;
}

void Map2DRender::run()
{
    std::deque<std::pair<cv::Mat,pi::SE3d> > frames;
    while(!shouldStop())
    {
        if(_valid)
        {
            if(getFrames(frames))
            {
                pi::timer.enter("Map2DRender::renderFrame");
                renderFrames(frames);
                pi::timer.leave("Map2DRender::renderFrame");
            }
        }
        sleep(10);
    }
    svar.GetInt("ShouldStop")=1;
}

void Map2DRender::draw()
{
    if(!_valid) return;

    SPtr<Map2DRenderPrepare> p;
    SPtr<Map2DRenderData>    d;
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
    std::vector<SPtr<Map2DRenderEle> > dataCopy=d->data();
    int wCopy=d->w(),hCopy=d->h();
    glColor3ub(255,255,255);
    for(int x=0;x<wCopy;x++)
        for(int y=0;y<hCopy;y++)
        {
            int idxData=y*wCopy+x;
            float x0=d->min().x+x*d->eleSize();
            float y0=d->min().y+y*d->eleSize();
            float x1=x0+d->eleSize();
            float y1=y0+d->eleSize();
            SPtr<Map2DRenderEle> ele=dataCopy[idxData];
            if(!ele.get())  continue;
            if(ele->img.empty()) continue;
            if(ele->texName==0)
            {
                glGenTextures(1, &ele->texName);
            }
            if(ele->Ischanged&&ticTac.Tac()<0.02)
            {
                pi::timer.enter("glTexImage2D");
                pi::ReadMutex lock1(ele->mutexData);
                glBindTexture(GL_TEXTURE_2D,ele->texName);
//                if(ele->img.elemSize()==1)
                    glTexImage2D(GL_TEXTURE_2D, 0,
                                 GL_RGBA, ele->img.cols,ele->img.rows, 0,
                                 GL_BGRA, GL_UNSIGNED_BYTE,ele->img.data);
                    if(svar.GetInt("ShowTex",0))
                        cv::imshow("tex",ele->img);
                //    glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_DECAL);
                //    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
                //glTexEnvfv(GL_TEXUTRE_ENV,GL_TEXTURE_ENV_COLOR,&ColorRGBA);
                //                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,  GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
                /*
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);*/
                ele->Ischanged=false;
                pi::timer.leave("glTexImage2D");
            }
            glBindTexture(GL_TEXTURE_2D,ele->texName);
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0,0);
            glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y1,0);
            glTexCoord2f(1.0f, 1.0f); glVertex3f(x1,y1,0);
            glTexCoord2f(1.0f, 0.0f); glVertex3f(x1,y0,0);
            glEnd();
        }
    glBindTexture(GL_TEXTURE_2D, last_texture_ID);
    glPopMatrix();
}

bool Map2DRender::save(const std::string& filename)
{
    // determin minmax
//    pi::Point2i minInt(1e6,1e6),maxInt(-1e6,-1e6);

//    std::vector<SPtr<Map2DRenderEle> > dataCopy;
//    int wCopy,hCopy;
//    {
//        pi::ReadMutex lock(mutexData);
//        wCopy=_w;hCopy=_h;
//        dataCopy=data;
//    }
//    for(int x=0;x<wCopy;x++)
//        for(int y=0;y<hCopy;y++)
//        {
//            SPtr<Map2DRenderEle> ele=dataCopy[wCopy*y+x];
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
