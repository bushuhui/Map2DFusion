
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef HAS_CVD
#include <cvd/Linux/v4lbuffer.h>
#include <cvd/colourspace_convert.h>
#include <cvd/colourspaces.h>
#endif

#include "VideoReader.h"
#include "VideoPOS_Manager.h"


namespace pi {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class VideoReader_Base
{
public:
    VideoReader_Base(std::string confName) {
        m_confName = m_confName;

        videoFile   = svar.GetString(confName + ".File", "");
        videoType   = svar.GetString(confName + ".Type", "FILE");

        FPS         = svar.GetDouble(confName + ".FPS", 60);
        ImgSkip     = &svar.GetInt(confName + ".ImgSkip", 10);
        timeDelay   = &svar.GetDouble(confName + ".timeDelay", 0.0);

        camInName   = svar.GetString(confName + ".CameraInName", "GoProIdeaM960");
        camOutName  = svar.GetString(confName + ".CameraOutName", "GoProIdeaM960");

        camIn = SPtr<Camera>(GetCameraFromName(camInName));
        if( camInName == camOutName ) {
            camOut = camIn;
        } else {
            camOut = SPtr<Camera>(GetCameraFromName(camOutName));
        }

        imgUndistort = svar.GetInt(confName + ".Undistorter", 0);
        if( imgUndistort )
            camUndis = SPtr<Undistorter>(new Undistorter(GetCopy(camIn.get()), GetCopy(camOut.get())));
    }

    virtual ~VideoReader_Base() {}

    virtual int open(std::string fn="") = 0;
    virtual int close(void) = 0;

    virtual int grabImage(VideoData &img) = 0;

public:
    std::string                 m_confName;
    std::string                 videoType;
    std::string                 videoFile;

    double                      FPS;
    int                         *ImgSkip;
    double                      *timeDelay;

    std::string                 camInName, camOutName;
    SPtr<Camera>                camIn, camOut;
    int                         imgUndistort;
    SPtr<Undistorter>           camUndis;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class VideoReader_FILE : public VideoReader_Base
{
public:
    VideoReader_FILE(std::string confName) : VideoReader_Base(confName) {

    }
    virtual ~VideoReader_FILE() {
        close();
    }

    virtual int open(std::string fn="") {
        // close previous video
        if( m_reader.isOpened() ) close();

        if( fn == "" )
            return m_reader.open(videoFile);
        else
            return m_reader.open(fn);
    }

    virtual int close(void) {
        return m_reader.close();
    }

    virtual int grabImage(VideoData &img) {
        int ret = 0;
        RVideoFrameInfo videoinfo;

        // skip some frames
        if( *ImgSkip > 10 ) {
            ret = ff(*ImgSkip-10);

            for(int i=0; ret == 0 && i<10; i++) ret = read(img.img, &videoinfo);
        } else {
            if( *ImgSkip > 0 )
                for(int i=0; ret == 0 && i<*ImgSkip; i++) ret = read(img.img, &videoinfo);
        }
        if( ret < 0 ) return -1;

        // read the image
        ret = read(img.img, &videoinfo);
        if( ret < 0 ) return -1;

        if( img.img.empty() ) {
            //dbg_pe("get videoframe failed\n");
            return -1;
        }

        img.timestamp = videoinfo.timestamp;

        return 0;
    }

    int read(cv::Mat &img, RVideoFrameInfo *vi) {
        int ret = m_reader.read(img, vi);

        if( ret < 0 ) m_reader.close();

        return ret;
    }

    int ff(int n) {
        int ret = m_reader.ff(n);

        if( ret < 0 ) m_reader.close();

        return ret;
    }


protected:
    RVideoReader        m_reader;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#ifdef HAS_CVD
class VideoReader_V4L : public VideoReader_Base
{
public:
    VideoReader_V4L(std::string confName) : VideoReader_Base(confName) {
        pvb = NULL;
    }

    virtual ~VideoReader_V4L() {
        if( pvb != NULL ) {
            delete pvb;
            pvb = NULL;
        }
    }


    virtual int open(std::string fn="") {
        using namespace CVD;

        m_bFirstFrame = 1;

        string  QuickCamFile;
        int     imgW, imgH;

        if( fn == "" )  QuickCamFile = videoFile;
        else            QuickCamFile = fn;

        imgW = svar.GetInt(m_confName + ".imgW", 1920);
        imgH = svar.GetInt(m_confName + ".imgH", 1080);

        dbg_pt("Opening V4L video: %s\n", QuickCamFile.c_str());

        ImageRef irSize = ImageRef(imgW, imgH);
        int nFrameRate =  FPS;

        try {
            pvb = new V4LBuffer<yuv422>(QuickCamFile, irSize, -1, false, nFrameRate);
        } catch (...) {
            pvb = NULL;

            dbg_pe("Can not open video: %s\n", fn.c_str());
            return -1;
        }

        if( !pvb ) {
            dbg_pe("Can not open video: %s\n", fn.c_str());
            return -1;
        } else {
            return 0;
        }

        return 0;
    }

    virtual int close(void) {

    }

    virtual int grabImage(VideoData &img) {
        using namespace CVD;

        if( !pvb ) return -1;

        ImageRef irSize = pvb->size();

        Image<Rgb<byte> > imRGB;
        imRGB.resize(irSize);
        CVD::VideoFrame<yuv422> *pVidFrame = pvb->get_frame();

        //img.timestamp = pVidFrame->timestamp();
        img.timestamp = tm_getTimeStamp() + *timeDelay;
        //dbg_pt("timeStamp = %f\n", img.timestamp);

        convert_image(*pVidFrame, imRGB);
        pvb->put_frame(pVidFrame);              // release frame buffer

        cv::cvtColor(cv::Mat(irSize.y, irSize.x,
                     CV_8UC3, imRGB.data()),
                     img.img, CV_BGR2RGB);

        return 0;
    }

protected:
    CVD::V4LBuffer<CVD::yuv422>     *pvb;

    int                             m_bFirstFrame;      ///< first frame
    double                          m_tsBeg;            ///< timeStamp for begin
};
#else
typedef VideoReader_FILE VideoReader_V4L;
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

VideoReader_Base* GetVideoReaderByName(std::string confName)
{
    VideoReader_Base *vr = NULL;

    std::string videoType = svar.GetString(confName + ".Type", "FILE");

    if( videoType == "FILE" ) {
        vr = new VideoReader_FILE(confName);
    } else if ( videoType == "LIVE" ) {
        vr = new VideoReader_V4L(confName);
    }

    return vr;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


VideoPOS_Manager::VideoPOS_Manager(std::string confName)
{
    SvarWithType<VideoPOS_Manager*>::instance()["VideoPOS_Manager.ptr"] = this;

    m_confName = confName;

    m_vpt = SPtr<VideoPOS_Transfer>(createVPT_slot());
    m_pdm = SPtr<POS_DataManager>(new POS_DataManager);

    // get configures
    m_useIPC  = svar.GetInt(confName + ".IPC", 1);
    string szType = svar.GetString(confName + ".Type", "SERVER");
    if( szType == "SERVER" ) m_vpmType = 1;
    else                     m_vpmType = 0;
    string nodeName = svar.GetString(confName + ".NodeName", "Master");

    // start VideoPOS transfer system
    if( m_vpmType == 1 ) {
        m_vpt->setMasterNodeName(nodeName);

        m_vpt->start(1, m_useIPC==0);
    } else {
        m_vpt->setReceiverNodeName(nodeName);

        m_vpt->start(0, m_useIPC==0);
    }
}

VideoPOS_Manager::~VideoPOS_Manager()
{
    close();

    m_vpt->stop();

    SvarWithType<VideoPOS_Manager*>::instance()["VideoPOS_Manager.ptr"] = NULL;
}



int VideoPOS_Manager::videoIncome_slot(VideoData *img, POS_DataManager *pDM)
{
    return 0;
}


VideoPOS_Transfer* VideoPOS_Manager::createVPT_slot(void)
{
    return (new VideoPOS_Transfer());
}


int VideoPOS_Manager::getCameras(Camera **camIn, Camera **camOut, Undistorter **undistort)
{
    if( m_videoReader.get() ) {
        *camIn  = m_videoReader->camIn.get();
        *camOut = m_videoReader->camOut.get();
        *undistort = m_videoReader->camUndis.get();

        return 0;
    }

    return -1;
}


int VideoPOS_Manager::thread_func(void *arg)
{
    m_playTS = &svar.GetDouble("IPC.PlayTS", -1.0);
    int *undisType = &svar.GetInt("VideoPOS_Manager.ImageUndistortType", 0);

    Undistorter *undis = NULL;
    if( m_videoReader.get() ) undis = m_videoReader->camUndis.get();

    int queueSize = svar.GetInt("VideoPOS_Manager.queueSize", 20);
    int &pause=svar.GetInt("Pause", 0);
    while( getAlive() ) {
        if( NULL == m_videoReader.get() || pause) {
            tm_sleep(80);
            continue;
        }

        // get a video frame
        spVideoData vd(new VideoData);
        if( 0 != m_videoReader->grabImage(*vd) ) {
            tm_sleep(80);
            continue;
        }

        // undistort input image
        if( undis ) {
            cv::Mat img;

            if( undisType == 0 ) undis->undistortFast(vd->img, img);
            else                 undis->undistort(vd->img, img);

            vd->img = img;
        }

        // sync to given time stamp
        //  or keep playspeed
        syncTimeStamp(vd);

        // call video income slot and then send current video frame to network
        videoIncome_slot(vd.get(), m_pdm.get());
        m_vpt->sendVideo(vd);

        // FIXME: if no receiver then remove some video
        if( m_vpt->sizeVideo() > queueSize ) {
            for(int i=0; i<queueSize-m_vpt->sizeVideo(); i++) m_vpt->popVideoData();
        }

        // push to image queue for FGCS use
        push(vd);

        // FIXME: if FGCS do not use the image data then clear it
        if( m_videoQueue.size() > queueSize ) {
            for(int i=0; i<queueSize-m_videoQueue.size(); i++) m_videoQueue.pop_front();
        }
    }

    return 0;
}

void VideoPOS_Manager::syncTimeStamp(SPtr<VideoData> &vd)
{
    if( *m_playTS < 0 ) {
        // sync to actual time or GPS time
        if( m_tm0 == -1 ) {
            m_tm0 = vd->timestamp;
            m_st0 = tm_get_us() / 1e6;
        } else {
            m_tm1 = vd->timestamp;
            m_st1 = tm_get_us() / 1e6;

            double tm_dt = m_tm1 - m_tm0;
            if( tm_dt > 1e-3 ) {
                if( tm_dt < 10.0 && tm_dt > 1e-6 ) {
                    ri64 dt = 1e6*(tm_dt - (m_st1 - m_st0));

                    if( dt > 1 ) tm_sleep_us(dt);

                    m_tm0 = m_tm1;
                    m_st0 = m_st1;
                }

                // reset begin time
                if( tm_dt >= 10.0 ) {
                    m_tm0 = m_tm1;
                    m_st0 = m_st1;
                }
            }
        }
    } else {
        // sync to given PTS
        while ( 1 ) {
            double dt = vd->timestamp - *m_playTS;

            /*
            printf("ts_video = %f, ts_sys = %f, dt(ts_video - ts_sys) = %f\n",
                   vd->timestamp, *m_playTS, dt);
            */

            if( dt > 0 ) tm_sleep_us(2000);
            else break;
        }
    }
}

int VideoPOS_Manager::open(std::string videoFN, std::string videoConf)
{
    int ret = -1;

    m_tm0 = -1;
    m_st0 = -1;

    m_vpt->clearPOS();
    m_vpt->clearVideo();

    // creat video reader
    if( videoConf == "" ) {
        string videoName = svar.GetString(m_confName + ".Video", "Video_Default");
        m_videoReader = SPtr<VideoReader_Base>(GetVideoReaderByName(videoName));
    } else {
        m_videoReader = SPtr<VideoReader_Base>(GetVideoReaderByName(videoConf));
    }

    if( m_videoReader.get() == NULL ) return -1;

    if( videoFN == "" )
        ret = m_videoReader->open();
    else
        ret = m_videoReader->open(videoFN);

    // start reading thread
    if( ret == 0 ) start();

    return ret;
}

int VideoPOS_Manager::close(void)
{
    // stop thread
    setAlive(0);
    wait(20);
    kill();

    m_vpt->clearPOS();
    m_vpt->clearVideo();

    return 0;
}


int VideoPOS_Manager::size(void)
{
    RMutex m(&m_mutex);

    return m_videoQueue.size();
}

int VideoPOS_Manager::push(SPtr<VideoData> vd)
{
    RMutex m(&m_mutex);

    m_videoQueue.push_back(vd);

    return 0;
}

SPtr<VideoData> VideoPOS_Manager::pop(void)
{
    RMutex m(&m_mutex);

    SPtr<VideoData> d;

    if( m_videoQueue.size() > 0 ) {
        d = m_videoQueue.front();
        m_videoQueue.pop_front();
    }

    return d;
}

int VideoPOS_Manager::clear(void)
{
    RMutex m(&m_mutex);

    m_videoQueue.clear();

    return 0;
}

int VideoPOS_Manager::sendPOS(spPOSData pd)
{
    m_pdm->addData(*pd);
    m_vpt->sendPOS(pd);

    return 0;
}


} // end of namespace pi
