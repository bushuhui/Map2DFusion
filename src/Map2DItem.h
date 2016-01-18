#ifndef MAP2DITEM_H
#define MAP2DITEM_H
#include <opmapcontrol/opmapcontrol.h>
#include <base/types/types.h>

namespace mapcontrol{

class Map2DElement
{
public:
    Map2DElement(){}
    Map2DElement(const QPixmap& img,const internals::PointLatLng& _lt,const internals::PointLatLng& _rb)
        :image(img),lt(_lt),rb(_rb)
    {
    }

    QPixmap image;
    internals::PointLatLng lt,rb;
};

class Map2DItem: public QObject, public QGraphicsItem
{
//    Q_OBJECT
//    Q_INTERFACES(QGraphicsItem)
public:
    enum { Type = UserType + 25 };
    Map2DItem(MapGraphicItem* _map, OPMapWidget* parent);
    virtual ~Map2DItem();

    virtual void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget);

    virtual QRectF boundingRect() const;

    bool update(const QPixmap& img,const pi::Point3d& _lt,const pi::Point3d& _rb);

    QRectF rectGPS;
    std::map<internals::PointLatLng,Map2DElement> elements;

    MapGraphicItem* map;
    OPMapWidget* mapwidget;
};

}

#endif // MAP2DITEM_H
