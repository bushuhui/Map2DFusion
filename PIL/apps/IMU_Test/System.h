#ifndef SYSTEM_H
#define SYSTEM_H

#include <base/system/thread/ThreadBase.h>
#include <opengl/Win3D.h>
#include <hardware/IMU/IMU.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace pi;

class System : public EventHandle,public Draw_Opengl,public Thread
{
public:
    System();

    void Initial();
    virtual void run();
    void mainLoop();
    void StopAllThread();

    virtual bool KeyPressHandle(void *arg);
    virtual void Draw_Something();

    void TestOpengl();
    void CalibrateIMU();
    void RealtimePose();
    void DrawTexture();

protected:
    Win3D       win3d;              //this will catch the main system
    IMU         imu;                //imu has its own thread
    pthread_t   thread_System;      //this system will handle
    GLuint Texture;
    cv::Mat tex_img;

};
#endif // SYSTEM_H
