#ifndef ONLINEFUSIONOBJECT_H
#define ONLINEFUSIONOBJECT_H

#include "fastfusion/fusion/geometryfusion_mipmap_cpu.hpp"
#include "fastfusion/fusion/mesh.hpp"

#include "gui/gl/GL_Object.h"
#include "base/system/thread/ThreadBase.h"
#include <base/types/SE3.h>

//#include <GL/glx.h>
#include <GL/glu.h>

class OnlineFusionObject:public pi::gl::GL_Object
{
public:
    OnlineFusionObject(float offsetX, float offsetY, float offsetZ,
                                      float scale, float distanceThreshold,
                                      sidetype n = 0, bool color = true);

    virtual ~OnlineFusionObject();

    virtual void draw();

    CameraInfo fromSE3(pi::SE3d pose,float fx,float fy,float cx,float cy);

    int addFrame(cv::Mat depth, CameraInfo caminfo,
                  std::vector<cv::Mat> rgb = std::vector<cv::Mat>(3),bool updateMesh=true);

    int addFrame(cv::Mat depth, CameraInfo caminfo, cv::Mat rgb,
                 float scaling=1.0, float maxcamdistance=10,bool updateMesh=true);

    std::vector<int> addFrame(const std::vector<cv::Mat>& depths,
                              const std::vector<CameraInfo>& caminfos,
                              const std::vector<cv::Mat>& rgbs,
                              float scaling=1.0,float maxcamdistance=10,bool updateMesh=true);

    int updateMesh();

    void setThreadMeshing(bool use_thread);

    void setVerbose(bool verbose);

    bool save2ply(const std::string& filename,bool bin=1);
    bool save2Obj(const std::string& filename);

    void clear();

    bool _colorEnabled,_updatingMesh;
    int _displayMode;
protected:
    void generateBuffers();

    MeshInterleaved *_currentMeshInterleaved;
    FusionMipMapCPU *fusion;
    unsigned int _currentNV,_currentNF;

    GLuint _vertexBuffer;
    GLuint _faceBuffer;
    GLuint _edgeBuffer;
    GLuint _normalBuffer;
    GLuint _colorBuffer;

    pi::Mutex _mutex;

};


#endif // ONLINEFUSIONOBJECT_H
