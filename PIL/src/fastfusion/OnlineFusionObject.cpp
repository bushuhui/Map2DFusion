#include <GL/glew.h>
//#include <GL/glx.h>

#include "OnlineFusionObject.h"
#include "gui/gl/SignalHandle.h"
#include <base/Svar/Svar.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <base/time/Global_Timer.h>

OnlineFusionObject::OnlineFusionObject(float offsetX, float offsetY, float offsetZ,
                                  float scale, float distanceThreshold,
                                  sidetype n , bool color )

    :fusion(NULL),
      _currentMeshInterleaved(NULL),_currentNV(0),_currentNF(0),_vertexBuffer(0),
      _colorEnabled(true),_displayMode(2),_updatingMesh(false)
{
    fusion=new FusionMipMapCPU(offsetX,offsetY,offsetZ,scale,distanceThreshold,n,color);
    setThreadMeshing(true);
    fusion->setVerbose(false);
//    fusion.setLoopClosureEnable(svar.GetInt("Fusion.LoopClosureEnabled",1));
//    fusion.setLoopClosureLogging(svar.GetInt("Fusion.LoopClosureLogging",1));
//    fusion._loopClosureMode=svar.GetInt("Fusion.LoopClosureMode",fusion._loopClosureMode);
}

OnlineFusionObject::~OnlineFusionObject()
{
    pi::gl::Signal_Handle::instance().delete_buffer(_vertexBuffer);
    pi::gl::Signal_Handle::instance().delete_buffer(_faceBuffer);
    pi::gl::Signal_Handle::instance().delete_buffer(_edgeBuffer);
    pi::gl::Signal_Handle::instance().delete_buffer(_normalBuffer);
    pi::gl::Signal_Handle::instance().delete_buffer(_colorBuffer);
    delete fusion;
}



void OnlineFusionObject::setThreadMeshing(bool use_thread)
{
    fusion->setThreadMeshing(use_thread);
}

void OnlineFusionObject::setVerbose(bool verbose)
{
    fusion->setVerbose(verbose);
}

void OnlineFusionObject::draw(){
    pi::ScopedMutex lock(_mutex);
    if(!_currentMeshInterleaved) return;

    glColor3f(1,1,1);
    if(_currentMeshInterleaved->faces.size()<1) return;

    if(_currentMeshInterleaved->vertices.size() != _currentNV ||
            _currentMeshInterleaved->faces.size() != _currentNF){
        eprintf("\nReassigning Buffers for interleaved Mesh");
        if(!_vertexBuffer){
            generateBuffers();
        }
        _currentNV = _currentMeshInterleaved->vertices.size();
        _currentNF = _currentMeshInterleaved->faces.size();

        glBindBuffer(GL_ARRAY_BUFFER,_vertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,_currentMeshInterleaved->vertices.size()*3*sizeof(float),_currentMeshInterleaved->vertices.data(), GL_STATIC_DRAW);
        if(_currentMeshInterleaved->colors.size()){
            glBindBuffer(GL_ARRAY_BUFFER,_colorBuffer);
            glBufferData(GL_ARRAY_BUFFER,_currentMeshInterleaved->colors.size()*3,_currentMeshInterleaved->colors.data(), GL_STATIC_DRAW);
        }

        if(_currentMeshInterleaved->faces.size()){
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,_faceBuffer);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, _currentMeshInterleaved->faces.size()*sizeof(unsigned int),
                         _currentMeshInterleaved->faces.data(), GL_STATIC_DRAW);
        }

        eprintf("\nChecking Mesh...");
        std::vector<bool> checks(_currentMeshInterleaved->vertices.size(),false);
        for(size_t i=0;i<_currentMeshInterleaved->faces.size();i++)
            checks[_currentMeshInterleaved->faces[i]] = true;

        bool loneVertex = false;
        for(size_t i=0;i<checks.size();i++) loneVertex |= !checks[i];
        if(loneVertex){
            fprintf(stderr,"\nThere were lone Vertices!");
        }
        eprintf("\nMesh Check done");
    }


    glBindBuffer(GL_ARRAY_BUFFER,_vertexBuffer);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    if(_currentMeshInterleaved->colors.size()){
        glBindBuffer(GL_ARRAY_BUFFER,_colorBuffer);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
    }
    else{
        glColor3f(0.5f,0.5f,0.5f);
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    if(_colorEnabled) {
        glEnableClientState(GL_COLOR_ARRAY);
    }
    else{
        glColor3f(0.5f,0.5f,0.5f);
    }


    if(_displayMode==1){
        glPolygonMode(GL_FRONT, GL_LINE);
        glPolygonMode(GL_BACK, GL_LINE);
        glLineWidth(0.5f);
    }
    else{
        glPolygonMode(GL_FRONT, GL_FILL);
        glPolygonMode(GL_BACK, GL_FILL);
    }

    if(_displayMode==2){
        glPointSize(2.0);
        glBindBuffer(GL_ARRAY_BUFFER,_vertexBuffer);
        glDrawArrays(GL_POINTS,0,_currentMeshInterleaved->vertices.size());
    }
    else{
        glDrawElements(GL_TRIANGLES, _currentMeshInterleaved->faces.size(), GL_UNSIGNED_INT,0);
    }

    if(_colorEnabled) glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

int OnlineFusionObject::addFrame(cv::Mat depth, CameraInfo caminfo,
              std::vector<cv::Mat> rgb ,bool updateMesh)
{
    fusion->addMap(depth,caminfo,rgb);
    if(!updateMesh) return 0;

    this->updateMesh();
}

void OnlineFusionObject::clear()
{
    FusionMipMapCPU* tmp=fusion;
    fusion=new FusionMipMapCPU(0,0,0,0.005,0.005);
    fusion->setThreadMeshing(svar.GetInt("Fusion.UseThread",1));
    fusion->setVerbose(svar.GetInt("Fusion.Verbose",0));
    delete tmp;
}

//void OnlineFusionObject::free()
//{
//    FusionMipMapCPU
//}

int OnlineFusionObject::addFrame(cv::Mat depth, CameraInfo caminfo, cv::Mat rgb,float scaling, float maxcamdistance,bool updateMesh)
{
    fusion->addMap(depth,caminfo,rgb,scaling,maxcamdistance);
    if(!updateMesh||_updatingMesh) return 0;

    this->updateMesh();
}

std::vector<int> OnlineFusionObject::addFrame(const std::vector<cv::Mat>& depths,
                          const std::vector<CameraInfo>& caminfos,
                          const std::vector<cv::Mat>& rgbs,
                          float scaling,float maxcamdistance,bool updateMesh)
{
    std::vector<int> ret;
    pi::timer.enter("FusionObject::clear");
    clear();
    pi::timer.leave("FusionObject::clear");
    std::cerr<<"ReFusioning "<<depths.size()<<" frames.\n";
    pi::timer.enter("FusionObject::ReFusion");
    for(int i=0;i<depths.size();i++)
        addFrame(depths[i],caminfos[i],rgbs[i],scaling,maxcamdistance);
    pi::timer.leave("FusionObject::ReFusion");

    if(!updateMesh||_updatingMesh) return ret;
    this->updateMesh();
    return ret;
}

int OnlineFusionObject::updateMesh()
{
    if(!fusion) return -1;
    _updatingMesh=true;
    fusion->updateMeshes();
    if(!_currentMeshInterleaved) _currentMeshInterleaved = new MeshInterleaved(3);
    {
        pi::ScopedMutex lock(_mutex);
        *_currentMeshInterleaved=fusion->getMeshInterleavedMarchingCubes();
    }
    _updatingMesh=false;
    return 0;
}

void OnlineFusionObject::generateBuffers(){
    glewInit();
    glGenBuffers(1, &_vertexBuffer);
    glGenBuffers(1, &_faceBuffer);
    glGenBuffers(1, &_colorBuffer);
    glGenBuffers(1, &_normalBuffer);
}

bool OnlineFusionObject::save2ply(const std::string& filename,bool bin)
{
    pi::ScopedMutex lock(_mutex);
    _currentMeshInterleaved->writePLY(filename,bin);
}

bool OnlineFusionObject::save2Obj(const std::string& filename)
{
    pi::ScopedMutex lock(_mutex);
    _currentMeshInterleaved->writeOBJ(filename);
}


CameraInfo OnlineFusionObject::fromSE3(pi::SE3d pose,float fx,float fy,float cx,float cy)
{
    CameraInfo result;
    cv::Mat intrinsic = cv::Mat::eye(3,3,cv::DataType<double>::type);
    //Kinect Intrinsic Parameters
    intrinsic.at<double>(0,0) = fx;
    intrinsic.at<double>(1,1) = fy;
    intrinsic.at<double>(0,2) = cx;
    intrinsic.at<double>(1,2) = cy;

    result.setIntrinsic(intrinsic);
    cv::Mat rotation2 = cv::Mat::eye(3,3,cv::DataType<double>::type);
//    for(int i=0;i<3;i++) for(int j=0;j<3;j++) rotation2.at<double>(i,j) = rotation(i,j);
    pose.get_rotation().getMatrix((double*)rotation2.data);
    result.setRotation(rotation2);

    pi::Point3d trans=pose.get_translation();
    cv::Mat translation2 = cv::Mat::zeros(3,1,cv::DataType<double>::type);
    translation2.at<double>(0,0) = trans.x;
    translation2.at<double>(1,0) = trans.y;
    translation2.at<double>(2,0) = trans.z;
    result.setTranslation(translation2);
    return result;
}
