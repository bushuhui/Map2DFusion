#include "BufferObject.h"
#include <GL/glew.h>
#include <gui/gl/glHelper.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

void BufferObject::generateBuffers()
{
    glewInit();
    glGenBuffers(1, &_vertexBuffer);
    glGenBuffers(1, &_faceBuffer);
    glGenBuffers(1, &_colorBuffer);
    glGenBuffers(1, &_normalBuffer);
}

void BufferObject::draw()
{
//    glColor3f(1,1,1);
//    if(faces.size()<1) return;

//    if(_currentMeshInterleaved->vertices.size() != _currentNV ||
//            _currentMeshInterleaved->faces.size() != _currentNF){
//        eprintf("\nReassigning Buffers for interleaved Mesh");
//        if(!_vertexBuffer){
//            generateBuffers();
//        }
//        _currentNV = _currentMeshInterleaved->vertices.size();
//        _currentNF = _currentMeshInterleaved->faces.size();

//        glBindBuffer(GL_ARRAY_BUFFER,_vertexBuffer);
//        glBufferData(GL_ARRAY_BUFFER,_currentMeshInterleaved->vertices.size()*3*sizeof(float),_currentMeshInterleaved->vertices.data(), GL_STATIC_DRAW);
//        if(_currentMeshInterleaved->colors.size()){
//            glBindBuffer(GL_ARRAY_BUFFER,_colorBuffer);
//            glBufferData(GL_ARRAY_BUFFER,_currentMeshInterleaved->colors.size()*3,_currentMeshInterleaved->colors.data(), GL_STATIC_DRAW);
//        }

//        if(_currentMeshInterleaved->faces.size()){
//            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,_faceBuffer);
//            glBufferData(GL_ELEMENT_ARRAY_BUFFER, _currentMeshInterleaved->faces.size()*sizeof(unsigned int),
//                         _currentMeshInterleaved->faces.data(), GL_STATIC_DRAW);
//        }

//        eprintf("\nChecking Mesh...");
//        std::vector<bool> checks(_currentMeshInterleaved->vertices.size(),false);
//        for(size_t i=0;i<_currentMeshInterleaved->faces.size();i++)
//            checks[_currentMeshInterleaved->faces[i]] = true;

//        bool loneVertex = false;
//        for(size_t i=0;i<checks.size();i++) loneVertex |= !checks[i];
//        if(loneVertex){
//            fprintf(stderr,"\nThere were lone Vertices!");
//        }
//        eprintf("\nMesh Check done");
//    }


//    glBindBuffer(GL_ARRAY_BUFFER,_vertexBuffer);
//    glVertexPointer(3, GL_FLOAT, 0, 0);
//    if(_currentMeshInterleaved->colors.size()){
//        glBindBuffer(GL_ARRAY_BUFFER,_colorBuffer);
//        glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
//    }
//    else{
//        glColor3f(0.5f,0.5f,0.5f);
//    }

//    glEnableClientState(GL_VERTEX_ARRAY);
//    if(_colorEnabled) {
//        glEnableClientState(GL_COLOR_ARRAY);
//    }
//    else{
//        glColor3f(0.5f,0.5f,0.5f);
//    }


//    if(_displayMode==1){
//        glPolygonMode(GL_FRONT, GL_LINE);
//        glPolygonMode(GL_BACK, GL_LINE);
//        glLineWidth(0.5f);
//    }
//    else{
//        glPolygonMode(GL_FRONT, GL_FILL);
//        glPolygonMode(GL_BACK, GL_FILL);
//    }

//    if(_displayMode==2){
//        glPointSize(2.0);
//        glBindBuffer(GL_ARRAY_BUFFER,_vertexBuffer);
//        glDrawArrays(GL_POINTS,0,_currentMeshInterleaved->vertices.size());
//    }
//    else{
//        glDrawElements(GL_TRIANGLES, _currentMeshInterleaved->faces.size(), GL_UNSIGNED_INT,0);
//    }

//    if(_colorEnabled) glDisableClientState(GL_COLOR_ARRAY);
//    glDisableClientState(GL_VERTEX_ARRAY);
}
