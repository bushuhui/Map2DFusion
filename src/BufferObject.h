#ifndef BUFFEROBJECT_H
#define BUFFEROBJECT_H
#include <vector>
#include <base/types/types.h>
#include <gui/gl/GL_Object.h>

typedef unsigned char uchar;
typedef pi::Point3f   Vertex3f;
typedef pi::Point3ub  Color3b;
typedef pi::Point2f   Vertex2f;

class BufferObject
{
public:
    BufferObject(unsigned int verticesPerFace)
        :_verticesPerFace(verticesPerFace)
    {}

    void draw();

    uchar _verticesPerFace;

    std::vector<Vertex3f> vertices;
    std::vector<unsigned int> faces;
    std::vector<Vertex3f> normals;
    std::vector<Color3b> colors;
    std::vector<unsigned int> edges;

    //Texture Coordinates
    std::vector<Vertex2f> texcoords;

    unsigned int _vertexBuffer,_faceBuffer,_colorBuffer,_normalBuffer;
protected:
    void generateBuffers();
    std::vector<unsigned int> materialIndices;
//    std::vector<cv::Mat>      textures;
};

#endif // BUFFEROBJECT_H
