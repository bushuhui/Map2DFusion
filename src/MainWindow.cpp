#include "MainWindow.h"
#include "gui/controls/SvarTable.h"
#include "Map2DItem.h"
#include <base/Svar/Scommand.h>
using namespace std;

void GuiHandle(void *ptr,string cmd,string para)
{
    if(cmd=="setMapType")
    {
        MainWindow* mainwindow=(MainWindow*)ptr;
        mainwindow->setMapType(para);
    }
    else if(cmd=="show")
    {
        MainWindow* mainwindow=(MainWindow*)ptr;
        mainwindow->call("show");
    }
    else if(cmd=="SetCurrentPosition")
    {
        cout<<"SetCurrentPosition "<<para<<endl;
        mapcontrol::OPMapWidget* mapwidget=(mapcontrol::OPMapWidget*)ptr;
        pi::Point2d latlng;
        stringstream sst(para);
        sst>>latlng;
        mapwidget->SetCurrentPosition(internals::PointLatLng(latlng.y,latlng.x));
    }
    else if(cmd=="SetZoom")
    {
        mapcontrol::OPMapWidget* mapwidget=(mapcontrol::OPMapWidget*)ptr;
        int level;
        stringstream sst(para);
        sst>>level;
        mapwidget->SetZoom(level);
    }
}

MainWindow::MainWindow(QWidget *parent):QMainWindow(parent),win3d(NULL),mapwidget(NULL)
{
    // set window minimum size
    this->setMinimumSize(1000, 700);

    // window title
    setWindowTitle("Map2DFusion");

    // setup layout
    setupLayout();
    connect(this, SIGNAL(call_signal() ), this, SLOT(call_slot()) );
    scommand.RegisterCommand("setMapType",GuiHandle,this);
    scommand.RegisterCommand("show",GuiHandle,this);
}

int MainWindow::setupLayout(void)
{
    QWidget *wAll = new QWidget(this);

    // left pannel
    QTabWidget* m_tabWidget = new QTabWidget(this);
    win3d   = new pi::gl::Win3D(this);
    m_tabWidget->addTab(win3d,"Win3D");

    if(!mapwidget){
        mapwidget=new mapcontrol::OPMapWidget(this);
        scommand.RegisterCommand("SetCurrentPosition",GuiHandle,mapwidget);
        scommand.RegisterCommand("SetZoom",GuiHandle,mapwidget);
        m_tabWidget->addTab(mapwidget,"MapWidget");
    }

    {

        mapwidget->configuration->SetAccessMode(core::AccessMode::ServerAndCache);
        mapwidget->configuration->SetTileMemorySize(200);
        mapwidget->configuration->SetCacheLocation("data/map/");

        mapwidget->SetCurrentPosition(internals::PointLatLng(34.2457760,108.918389));
        mapwidget->SetZoom(svar.GetInt("MapWidget.DefaultZoom",18));
        mapwidget->SetMinZoom(4);
        mapwidget->SetMaxZoom(18);
        mapwidget->SetMapType(MapType::BingSatellite);
    }

    mapcontrol::Map2DItem* map2ditem=new mapcontrol::Map2DItem(mapwidget->GetMapItem(),mapwidget);

    SvarWidget* svarWidget=new SvarWidget(this);
    m_tabWidget->addTab(svarWidget,"SvarWidget");


    // overall layout
    QHBoxLayout *hl = new QHBoxLayout(wAll);
    wAll->setLayout(hl);

    hl->addWidget(m_tabWidget, 1);

    setCentralWidget(wAll);

    return 0;
}

bool MainWindow::setMapType(const std::string& MapType)
{
    mapwidget->SetMapType(MapType::TypeByStr(QString::fromStdString(MapType)));
    cout<<"Map type has changed to "<<MapType::StrByType(mapwidget->GetMapType()).toStdString();
    return true;
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    int     key;

    key  = event->key();
}

void MainWindow::mousePressEvent(QMouseEvent *event)
{
#if 0
    // 1 - left
    // 2 - right
    // 4 - middle
    printf("window pressed, %d, %d, %d\n", event->button(), event->pos().x(), event->pos().y());

    if( event->button() == 1 ) {

    }
#endif
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
}

void MainWindow::timerEvent(QTimerEvent *event)
{
}

void MainWindow::call(const std::string& cmd)
{
    cmds.push(cmd);
    emit call_signal();
}

void MainWindow::call_slot()
{
    if(cmds.size())
    {
        std::string& cmd=(cmds.front());
        if("show"==cmd) show();
        else
            scommand.Call(cmd);
        cmds.pop();
    }

}
