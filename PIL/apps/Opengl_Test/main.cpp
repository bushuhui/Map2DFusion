#include "opengl/Win3D.h"
#include "base/Svar/Svar_Inc.h"

using namespace pi;

int main(int argc,char** argv)
{
    QApplication application(argc,argv);
    svar.ParseMain(argc,argv);
    Win3D win3d;

    GL_Object airplane;

    cout<<"file loaded"<<airplane.LoadFromFile(svar.GetString("AirPlaneFile","airplane.off"))<<" elements.\n";

    win3d.insertObject(airplane);
    airplane.pose.get_rotation().FromAxis(Point3f(0,0,1),3.1415926/2);
    airplane.pose.get_translation()=Point3f(40,0,0);
    cout<<"AirPlane Pose="<<airplane.pose;
    win3d.setSceneRadius(1000);
    win3d.show();

     return application.exec();
}
