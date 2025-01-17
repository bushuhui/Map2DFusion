#include "Map2DItem.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <base/Svar/Scommand.h>
#include <base/time/Global_Timer.h>
#include <string>
using namespace std;

namespace mapcontrol{

void Map2DItemHandle(void* ptr,std::string cmd,std::string para)
{
    stringstream sst(para);
    sst>>cmd;
    if(cmd=="Map2DUpdate")
    {
        pi::timer.enter("Map2DUpdate");
//        cout<<"Map2DUpdate Handled.\n";
        sst>>cmd;//image name
        cv::Mat img=SvarWithType<cv::Mat>::instance()[cmd];
        cv::Mat imgWeight=SvarWithType<cv::Mat>::instance()[cmd+"Weight"];
        if(img.empty()||(img.type()!=CV_8UC3))
        {
            cerr<<"Map2DItemHandle::Map2DUpdate: Not correct image!\n";
            return;
        }

        pi::Point3d tl,rb;
        sst>>tl>>rb;

        Map2DItem* item=(Map2DItem*)ptr;
        if(imgWeight.empty()||imgWeight.type()!=CV_32FC1)
        {
            cv::cvtColor(img,img,CV_BGR2RGB);
            //upsidedown
            {
                cv::Mat temp=img.clone();
                int rowsize=3*temp.cols;
                for(uchar* p=img.data,*dstP=temp.data+(temp.rows-1)*rowsize,*pend=temp.rows*rowsize+img.data;
                    p<pend;p+=rowsize,dstP-=rowsize)
                    memcpy(dstP,p,rowsize);
                img=temp;
            }
            item->update(QPixmap::fromImage(QImage(img.data,img.cols,img.rows,QImage::Format_RGB888)),tl,rb);
        }
        else
        {
            QImage dst(img.cols,img.rows,QImage::Format_ARGB32);
            for(int y=0;y<img.rows;y++)
            {
                pi::Byte<4>* Pdst=((pi::Byte<4>*)dst.bits())+y*img.cols;
                pi::Byte<3>* Psrc=((pi::Byte<3>*)img.data)+(img.rows-1-y)*img.cols;
                float* PsrcW=((float*)imgWeight.data)+(img.rows-1-y)*img.cols;
                for(int x=0;x<img.cols;x++)
                {
                    Pdst[x]=(pi::Byte<4>){Psrc[x].data[0],Psrc[x].data[1],Psrc[x].data[2],PsrcW[x]?255:0};//
                }
            }
            item->update(QPixmap::fromImage(dst),tl,rb);
        }
        pi::timer.leave("Map2DUpdate");
    }
    else if(cmd=="InsertGPSPoint")
    {
        Map2DItem* item=(Map2DItem*)ptr;
        double lng,lat;
        sst>>lng>>lat;
        item->insertGPSPoint(internals::PointLatLng(lat,lng));
    }
    else if(cmd=="SetPositionFromMap2D")
    {

    }
}

Map2DItem::Map2DItem(MapGraphicItem* _map, OPMapWidget* parent)
    :map(_map),mapwidget(parent)
{
    setParentItem(_map);
    setPos(0,0);
    scommand.RegisterCommand("MapWidget",Map2DItemHandle,this);
}

Map2DItem::~Map2DItem()
{
    scommand.UnRegisterCommand("MapWidget");
}

void Map2DItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
           QWidget *widget)
{
    pi::timer.enter("Map2DItem::paint");
    Q_UNUSED(option);
    Q_UNUSED(widget);
    for(std::map<internals::PointLatLng,Map2DElement>::iterator it=elements.begin();it!=elements.end();it++)
    {
        Map2DElement& ele=it->second;
        core::Point lt=map->FromLatLngToLocal(ele.lt);
        core::Point rb=map->FromLatLngToLocal(ele.rb);
        painter->drawPixmap(QRect(lt.X(),lt.Y(),rb.X()-lt.X(),rb.Y()-lt.Y()),ele.image);
    }

    if(gpsPoints.size()>=2)
    {
        //convert to lines
        std::vector<QPointF> pts;
        pts.reserve(gpsPoints.size());
        for(int i=0;i<gpsPoints.size();i++)
        {
            core::Point pt=map->FromLatLngToLocal(gpsPoints[i]);
            pts.push_back(QPointF(pt.X(),pt.Y()));
        }
        for(int i=1;i<pts.size();i++)
            painter->drawLine(pts[i-1],pts[i]);
    }

    pi::timer.leave("Map2DItem::paint");
}

QRectF Map2DItem::boundingRect()const
{
    return QRectF(10.,10.,256.,256.);
}

bool Map2DItem::update(const QPixmap& img,const pi::Point3d& _lt,const pi::Point3d& _rb)
{
    if(img.isNull())
    {
        cerr<<"Map2DItem::update :QPixmap is NULL!\n";
        return false;
    }

//    cout<<"Updating "<<_lt<<" "<<_rb<<endl;
    internals::PointLatLng lt(std::max(_lt.y,_rb.y),std::min(_lt.x,_rb.x));
    elements[lt]=Map2DElement(img,lt,internals::PointLatLng(std::min(_lt.y,_rb.y),std::max(_lt.x,_rb.x)));
    return true;
}

}
