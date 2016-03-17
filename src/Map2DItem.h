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

/**
 * @brief The Map2DItem class does not contains any mutex and all thing should be done in GUI thread.
 */
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
    void insertGPSPoint(const internals::PointLatLng& pos){gpsPoints.push_back(pos);}

    QRectF rectGPS;
    std::map<internals::PointLatLng,Map2DElement> elements;
    std::vector<internals::PointLatLng>           gpsPoints;

    MapGraphicItem* map;
    OPMapWidget* mapwidget;
};

}

#endif // MAP2DITEM_H
