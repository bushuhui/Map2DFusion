#include "System.h"

#include <base/Svar/Svar_Inc.h>
#include <base/time/Timer.h>
#include <opengl/GL_Objects.h>
#include <stdio.h>

#include <iomanip>


System::System():win3d(NULL)
{

    Initial();
}

void System::Initial()
{
    win3d.SetEventHandle(this);
    //win3d.SetDraw_Opengl(this);
    Texture=0;
    win3d.show();
}


void System::run()
{
    mainLoop();
    svar.i["Pause"]=0;
}

void System::mainLoop()
{
    string act=svar.GetString("Act","RealtimePose");
    cout<<"Act="<<act<<endl;
    if(act=="CalibrateIMU") CalibrateIMU();
    if(act=="RealtimePose") RealtimePose();
    if(act=="DrawTexture")  DrawTexture();
    if(act=="TestOpengl")   TestOpengl();
    StopAllThread();
}

bool System::KeyPressHandle(void *arg)
{
    QKeyEvent* e=(QKeyEvent*)arg;
    switch(e->key())
    {
    case 'R':
        imu.CurrentFrame.pose.get_translation()=Point3d(0,0,0);
        imu.CurrentFrame.v=Point3d(0,0,0);
        return true;
    case 'P':
        if(svar.i["Pause"]) svar.i["Pause"]=0;
        else svar.i["Pause"]=1;
        return true;
    default:
        return false;

    }
}

void System::StopAllThread()
{
    win3d.stopAnimation();
    win3d.close();
    cout<<"System mainloop thread stoped manually.\n";
}

void System::Draw_Something()
{
//        cout<<"I'm drawing things in class system...\n";
    if(!Texture)
    {
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &Texture );
        cout<<"Generating.\n";
        glBindTexture(GL_TEXTURE_2D,Texture);
        //    glTexImage2D(GL_TEXTURE_2D,0,3,256,256,0,GL_BGR,GL_UNSIGNED_BYTE,tex_img.data);

        glTexImage2D(GL_TEXTURE_2D, 0,
                     GL_RGB, tex_img.cols,tex_img.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE,tex_img.data);
        //    glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_DECAL);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        //glTexEnvfv(GL_TEXUTRE_ENV,GL_TEXTURE_ENV_COLOR,&ColorRGBA);
        cout<<"Texture generated.\n";

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);


    }
    glBindTexture(GL_TEXTURE_2D, Texture);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex3f(-8.0f, -8.0f, 0.0f);
    glTexCoord2f(0.0f, 5.0f); glVertex3f(-8.0f, 8.0f, 0.0f);
    glTexCoord2f(5.0f, 5.0f); glVertex3f(8.0f, 8.0f, 0.0f);
    glTexCoord2f(5.0f, 0.0f); glVertex3f(8.0f, -8.0f, 0.0f);
    glEnd();
}



void System::TestOpengl()
{
    pi::AxisXYZ axis(10);
    win3d.insertObject(axis);
    pi::CameraSquare square(Point3f(-1,1,1),Point3f(1,1,1),
                            Point3f(-1,-1,1),Point3f(1,-1,1));
    win3d.insertObject(square);
    Rate rate(30);
    double yaw=0,roll=0,pitch=0;
    while(!shouldStop())
    {
        rate.sleep();
        if(svar.i["Pause"]) continue;
        win3d.ClearInfo();
        pi::SE3<float> pose;
        yaw+=0;
        roll+=0;
        pitch+=0.1;
        pose.get_rotation().FromEulerAngle(pitch,yaw,roll);
        square.SetPose(pose);
        axis.SetPose(pose);

        win3d.info<<"pitch:"<<pose.get_rotation().getPitch()
                 <<"yaw:"<<pose.get_rotation().getYaw()
                <<"roll:"<<pose.get_rotation().getRoll()<<endl;

        win3d.info<<"pose="<<pose<<endl;
        win3d.info<<(TooN::SE3<float>)pose;
        win3d.update();
    }
    sleep(100000);
}

void System::CalibrateIMU()
{
    imu.CalibrateIMU();
}

void System::RealtimePose()
{
    cout<<"testing the class IMU.\n";
    GL_Object airplane;
    airplane.LoadFromFile("airplane.off");
    airplane.ApplyScale(0.001);
    win3d.insertObject(airplane);
    pi::SE3<> se3_air;
    TooN::Matrix<3,3> R=rebuildRotation(-90.0,0.,0.);
    se3_air.get_rotation().fromMatrix(R);
//    se3_air.get_rotation().FromEuler(-3.1415926/2,0,0);
//    se3_air.get_rotation().FromAxis(Point3d(0,0,1),3.1415926/2.0);
    airplane.SetPose(se3_air);
//    sleep(100000);

    imu.drift.PrintParaments();
    Rate rate(svar.GetInt("ComputeRate",30));
    for(int i=0;i<200;)
    {
        rate.sleep();
        u_int64 time=tm_get_millis();
        win3d.ClearInfo();
        win3d.ClearTemp();

        pi::SE3<> pose=imu.ComputePose(time);
        win3d.info<<"TimeUsed:"<<tm_get_millis()-time<<"ms\n";
        pi::SE3<> airplanepose=pose;
        airplanepose.get_rotation()=airplanepose.get_rotation()*se3_air.get_rotation();
        airplane.SetPose(pose);

        //Draw axis
        TooN::Matrix<3,3> R;
        pose.get_rotation().getMatrixUnsafe(R);
        ColorfulLine lineX(Point3f(0,0,0),100*Point3f(R[0][0],R[1][0],R[2][0]),255,0,0);
        win3d.InsertLine(lineX);
        ColorfulLine lineY(Point3f(0,0,0),100*Point3f(R[0][1],R[1][1],R[2][1]),0,255,0);
        win3d.InsertLine(lineY);
        ColorfulLine lineZ(Point3f(0,0,0),100*Point3f(R[0][2],R[1][2],R[2][2]),0,0,255);
        win3d.InsertLine(lineZ);

        AHRS_Frame& frame=imu.CurrentFrame.frame;
        win3d.info<<"yaw:"<<frame.yaw<<" roll:"<<frame.roll<<" pitch:"<<frame.pitch<<endl;
        win3d.info<<R;

        //Draw path
        if(svar.GetInt("ShowPath",0))
        {
            ColorfulPoint point(pose.get_translation(),255,255,255);
            win3d.InsertPoint(point,true);
            Point3d &acc=imu.CurrentFrame.acc;
            Point3d v;
            if(svar.GetInt("UseFilter",1))
                v=imu.CurrentFrame.filted_v;
            else
                v=imu.CurrentFrame.v;

            //Draw acc & vesolity
            int n=7;
            ColorfulLine lineAcc(Point3f(0,0,0),acc,255,0,0);
            win3d.InsertLine(lineAcc);
            ColorfulLine lineV(Point3f(0,0,0),v,0,255,0);
            win3d.InsertLine(lineV);
            win3d.info<<"SystemState:\nacc="<<setiosflags(ios::fixed)<<setprecision(3)<<setfill(' ')<<setw(n)<<acc
                     <<"\nv="<<setw(n)<<v<<"\npose="<<setw(n)<<pose<<endl;
        }

        //Draw airplane
        TooN::SE3<> pp=((TooN::SE3<>)pose)*((TooN::SE3<>)se3_air);
        pi::SE3<>  zy_pp;
        zy_pp.get_rotation().fromMatrixUnsafe(pp.get_rotation());
        airplane.SetPose(pose*se3_air);
        win3d.update();
    }
}

using namespace cv;
void System::DrawTexture()
{
    tex_img=imread("texture.jpg");
    if(tex_img.empty())
    {
        fprintf(stderr, "Can not load image !\n");
        return;
    }
    else
        cout<<tex_img.rows<<tex_img.cols<<tex_img.elemSize1()<<endl;

//    imshow("texture",tex_img);
    unsigned char *p=tex_img.data,temp;
    for(int i=0;i<tex_img.cols*tex_img.rows;i++)
    {
        temp=p[i*3];
        p[i*3]=p[i*3+2];
        p[i*3+2]=temp;
    }
    win3d.SetDraw_Opengl(this);
    sleep(50000);
}

