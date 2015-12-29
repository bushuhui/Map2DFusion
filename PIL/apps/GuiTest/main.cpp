#include <iostream>
#include <base/Svar/Svar.h>
#include <gui/MainWindowBase.h>
#include <base/time/Timer.h>
#include <gui/gl/Win3D.h>
#include <gui/gl/Object_3DS.h>

void luaTest()
{
    pi::MainWindowBase mainwindow;
    mainwindow.call("luaLoadFile gui.lua");
//    lua.load_file("gui.lua");
    mainwindow.call("show");

    pi::Rate rate(60);
    while(!mainwindow.shouldStop())
    {
        rate.sleep();
    }
}

int Win3DTest(int argc,char** argv)
{
    string file3ds = svar.GetString("Win3DTest.3dsFile","/data/zhaoyong/Linux/Program/Apps/PIS/PIL/apps/GuiTest/gear.3DS");

    pi::gl::Object_3DS* file=new pi::gl::Object_3DS(file3ds.c_str());
    pi::gl::GL_ObjectPtr ele(file);

    QApplication app(argc,argv);


    pi::gl::Win3D win3d;
    win3d.setSceneRadius(100);
    win3d.setBackgroundColor(QColor(0,128,255));
    win3d.setForegroundColor(QColor(0,128,255));
    win3d.qglColor(QColor(0,128,255));

    if(file->isOpened())
        win3d.insert(ele);

    win3d.show();
    cout<<win3d.backgroundColor().blue()<<endl;
    cout<<win3d.foregroundColor().blue()<<endl;

    return app.exec();
}

int main(int argc,char** argv)
{
    svar.ParseMain(argc,argv);
    return Win3DTest(argc,argv);
}
