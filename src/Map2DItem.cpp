#include "Map2DItem.h"
#include <opencv2/core/core.hpp>
#include <base/Svar/Scommand.h>
#include <string>
using namespace std;

namespace mapcontrol{

void Map2DItemHandle(void* ptr,std::string cmd,std::string para)
{
    stringstream sst(para);
    sst>>cmd;
    if(cmd=="Map2DUpdate")
    {
        sst>>cmd;//image name
        cv::Mat img=SvarWithType<cv::Mat>::instance()[cmd];
        if(img.empty()||(img.type()!=CV_8UC3)) return;
        pi::Point3d tl,rb;
        sst>>tl>>rb;
        Map2DItem* item=(Map2DItem*)ptr;
        item->update(QPixmap::fromImage(QImage(img.data,img.cols,img.rows,QImage::Format_RGB888)),tl,rb);
    }
}

Map2DItem::Map2DItem(MapGraphicItem* _map, OPMapWidget* parent)
    :map(_map),mapwidget(parent)
{
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
    Q_UNUSED(option);
    Q_UNUSED(widget);
    for(std::map<internals::PointLatLng,Map2DElement>::iterator it=elements.begin();it!=elements.end();it++)
    {
        Map2DElement& ele=it->second;
        core::Point lt=map->FromLatLngToLocal(ele.lt);
        core::Point rb=map->FromLatLngToLocal(ele.rb);
        painter->drawPixmap(QRect(lt.X(),lt.Y(),rb.X()-lt.X(),rb.Y()-lt.Y()),ele.image);
    }
}

QRectF Map2DItem::boundingRect()const
{
    return QRectF(10.,10.,256.,256.);
}

bool Map2DItem::update(const QPixmap& img,const pi::Point3d& _lt,const pi::Point3d& _rb)
{
    internals::PointLatLng lt(_lt.x,_lt.y);
    elements[lt]=Map2DElement(img,lt,internals::PointLatLng(_rb.x,_rb.y));
}

}
