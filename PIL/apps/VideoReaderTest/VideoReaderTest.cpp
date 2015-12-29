
#include <cv/highgui/VideoReader.h>
//#include <cv/img_proc/ledArray_detection.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <base/Svar/Svar_Inc.h>
#include <base/types/types.h>
#include <base/time/Global_Timer.h>
#include <base/time/DateTime.h>

//#define HasROSBag

#ifdef HasROSBag
#include <rosbag/bag.h>
#include<sensor_msgs/Image.h>
#include<std_msgs/Time.h>
#include<std_msgs/Header.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#endif

using namespace cv;
using namespace std;
using namespace pi;

//int TimeCalibrate()
//{
//    string      fn_in;

//    int         vfType = 0;
//    string      vfBase = "";
//    string      vfExt  = "MP4";

//    RVideoReader            vr;
//    RVideoFrameInfo         vfi;
//    Mat                     img, imgS;

//    rtk::RLEDArray_Detection     la_det;
//    rtk::RLEDArray_Marker_List   arrM;
//    int                     timeSynced = 0;

//    rtk::RDateTime               t, tm, ts;
//    ri64                    deltaTS = 0, tse=0;
//    ri64                    tsm=0, tss=0;
//    int                     markerFind;

//    int                     key = 0;
//    char                    buf[256];

//    // load parameters
//    fn_in = "/data/zhaoyong/Linux/Program/Data/uav_image/videos/GOPRO_2014.11/flight_06_00.MP4";
//    fn_in=svar.GetString("TimeCalibrate.video", fn_in);

//    vfType=svar.GetInt("TimeCalibrate.vfType", vfType);
//    vfBase=svar.GetString("vfBase", vfBase);
//    vfExt=svar.GetString("vfExt",  vfExt);


//    // load images
//    if( 0 == vfType )
//        vr.open(fn_in);
//    else
//        vr.open(vfBase, vfExt, 0);

//    if( !vr.isOpened() ) {
//        if( vfType == 0 )
//            rtk::dbg_pe("Can not open video file: %s\n", fn_in.c_str());
//        else
//            rtk::dbg_pe("Can not open video file: %s_%02d.%s\n", vfBase.c_str(), 0, vfExt.c_str());

//        return -1;
//    }

//    la_det.drawRes = 0;


//    while( 1 ) {
//        markerFind = 0;

//        if( key == 1113939 ) {
//            if( 0 != vr.ff(10*vr.fps) ) break;//down

//            key = 0;
//            continue;
//        } else if ( key == 1113940 ) {
//            if( 0 != vr.ff(30*vr.fps) ) break;//right

//            key = 0;
//            continue;
//        } else if ( key == 1048603 ) {
//            break;
//        }

//        // read a frame
//        timer.enter("GetFrame");
//        if( 0 != vr.read(img, &vfi) ) break;
//        timer.leave("GetFrame");

//        // get date/time
//        t.fromTimeStamp(vfi.timestamp);
//        /*
//        printf("[%6d] ts = %12lld, time = ", vfi.frameIdx, vfi.timestamp);
//        cout << t << ", ";
//        printf("pts = %15f\n", vfi.pts);
//        */

//        resize(img, imgS, cv::Size(), 0.5, 0.5);

//        // detect LED array
//        la_det.detect(imgS, arrM);

//        if( timeSynced ) {
//            tss = vfi.timestamp + deltaTS;
//            ts.fromTimeStamp(tss);
//        }

//        for(int i=0; i<arrM.size(); i++) {
//            rtk::RLEDArray_Marker &lm = arrM.at(i);

//            if( lm.isCorrect() == 0 && lm.tm_sf ) {
//                tm = t;
//                tm.min = lm.tm_m;
//                tm.sec = lm.tm_s;
//                tm.nano_sec = 0;
//                tsm = tm.toTimeStamp();

//                if( timeSynced == 0 ) {
//                    if( tsm <= vfi.timestamp )
//                        deltaTS = -(vfi.timestamp - tsm);
//                    else
//                        deltaTS = tsm - vfi.timestamp;

//                    fmt::print_colored(fmt::RED, "\ndeltaTS = {0}\n", deltaTS);

//                    tss = vfi.timestamp + deltaTS;
//                    ts.fromTimeStamp(tss);

//                    timeSynced = 1;
//                } else {
//                    tse = tss - tsm;
//                    printf("TS_e = %f\n", 1.0*tse/1000000);
//                }

//                markerFind = 1;

//                break;
//            }
//        }

//        // draw frame time into image
//        sprintf(buf, "Frame time = %4d-%02d-%02d %02d:%02d:%02d.%06d",
//                t.year, t.month, t.day, t.hour, t.min, t.sec,
//                t.nano_sec/1000);
//        putText(imgS, buf, Point(20, 30),
//                FONT_HERSHEY_PLAIN, 1.5,
//                Scalar(0x00, 0, 0xFF), 2);

//        sprintf(buf, "Syncd time = %4d-%02d-%02d %02d:%02d:%02d.%06d",
//                ts.year, ts.month, ts.day, ts.hour, ts.min, ts.sec,
//                ts.nano_sec/1000);
//        putText(imgS, buf, Point(20, 60),
//                FONT_HERSHEY_PLAIN, 1.5,
//                Scalar(0xFF, 0, 0x00), 2);

//        sprintf(buf, "TS error    = %f", 1.0*tse/1000000);
//        putText(imgS, buf, Point(20, 90),
//                FONT_HERSHEY_PLAIN, 1.5,
//                Scalar(0xFF, 0, 0xFF), 2);

//        int pm, ps, ttm, tts;
//        ps = (vfi.timestamp-vr.tsBeg)/1000000;
//        pm = ps / 60;
//        ps = ps % 60;
//        ttm = vr.duration / 60;
//        tts = vr.duration - ttm*60;
//        sprintf(buf, "Play time   = %02d:%02d (%02d:%02d) [%6d]",
//                pm, ps, ttm, tts, vfi.frameIdx);
//        putText(imgS, buf, Point(20, 120),
//                FONT_HERSHEY_PLAIN, 1.5,
//                Scalar(0x00, 0xFF, 0x00), 2);

//        imshow("video", imgS);
//        if( markerFind ) key = waitKey(10);
//        else             key = waitKey(10);
//    }
//    timer.dumpAllStats();
//    return 0;
//}

int main(int argc,char** argv)
{
    svar.ParseMain(argc,argv);
    string act=svar.GetString("Act","Reader");
    if(act=="TimeCalibrate")
    {
//        TimeCalibrate();
        return 0;
    }
    string          fn_in;

    RVideoReader    vr;
    RVideoFrameInfo vfi;
    Mat             img, imgS;

    DateTime       t;

    char             key;

    char            buf[256];

    // load parameters
    fn_in = "/data/zhaoyong/Linux/Program/Apps/GSLAM/Data/npu_playground.avi";
    svar.GetString("fn_in", fn_in);

    // load images
    vr.open(fn_in);
    if( !vr.isOpened() ) {
    printf("Can not open video file: %s\n", fn_in.c_str());
        return -1;
    }

    bool save2file=false;

    string outPath=svar.GetString("OutPath","");
    if(outPath!="")
    {
        save2file=true;
    }
    int index=0;

#ifdef HasROSBag

    rosbag::Bag bag_out(svar.GetString("BagName","out.bag"),
                        rosbag::bagmode::Write);
    ros::Time::init();
#endif
    while( 0 == vr.read(img, &vfi) ) {
        if( key == 1113939 ) {
            if( 0 != vr.ff(10*vr.fps) ) break;
            key = 0;
            continue;
        } else if ( key == 1113940 ) {
            //if( 0 != vr.ff(30*vr.fps) ) break;
            vr.read(img, &vfi,vfi.pts+300);
            key = 0;
            continue;
        } else if ( key == 1048603 ) {
            break;
        }

        t.fromTimeStamp(vfi.timestamp);
        fmt::printf("[%6d] ts = %12lld, time = ", vfi.frameIdx, vfi.timestamp);
        cout << t << ", ";
        fmt::printf("pts = %15f\n", vfi.pts);

        if(save2file)
        {
            index++;
//            if(index%30==0)
            {
//                resize(img, imgS, cv::Size(), 0.25, 0.25);
                imgS=img;
#ifdef HasROSBag
                cv_bridge::CvImage cvImage;
                cvImage.image=imgS;
                cvImage.encoding=sensor_msgs::image_encodings::RGB8;
                cvImage.header.stamp=ros::Time((double)vfi.timestamp/1000000.0+ros::Time::now().toSec());
                bag_out.write("/camera/image_raw", cvImage.header.stamp,cvImage.toImageMsg());
#else
                cv::Mat img_out;
                cv::cvtColor(imgS,img_out,CV_BGR2GRAY);
                imwrite(outPath+"/"+to_str(vfi.timestamp)+".png",img_out);
#endif
                // draw frame time into image
                sprintf(buf, "Frame time = %4d-%02d-%02d %02d:%02d:%02d.%06d",
                        t.year, t.month, t.day, t.hour, t.min, t.sec,
                        t.nano_sec/1000);
                putText(imgS, buf, Point(10, 20),
                        FONT_HERSHEY_PLAIN, 1.0,
                        Scalar(0xFF, 0, 0), 1);

                imshow("video", imgS);
            }
        }
        key = waitKey(10);
        if(key==27) break;
    }

#ifdef HasROSBag
    bag_out.close();
#endif
    return 0;
}
