#pragma once

#include <GlutFramework.h>
#include <TH.h>
#include <vector>
using namespace glutFramework;

struct GLvector
{
        GLfloat fX;
        GLfloat fY;
        GLfloat fZ;     
};

class cVolume: public Object{
public:
	cVolume(THFloatTensor *tensor);
        cVolume(THFloatTensor *tensor, int atom_type);
	~cVolume();
	void display();

        GLfloat   fTargetValue;
        GLfloat   fStepSize;
        GLint     iDataSetSize;
        GLenum    ePolygonMode;

private:
        GLvoid vMarchingCubes(int at);
        GLvoid vMarchCube(int at, GLfloat fX, GLfloat fY, GLfloat fZ, GLfloat fScale);

        GLvoid vNormalizeVector(GLvector &rfVectorResult, GLvector &rfVectorSource);
        GLvoid vGetColor(GLvector &rfColor, GLvector &rfPosition, GLvector &rfNormal);
        GLvoid vGetNormal(int at, GLvector &rfNormal, GLfloat fX, GLfloat fY, GLfloat fZ);

        GLfloat fSample(int at, GLfloat fX, GLfloat fY, GLfloat fZ);

        THFloatTensor *data;

        GLuint dataList, dataList1;

        std::vector<GLvector> manualVertexList;
        std::vector<GLvector> manualNormalList;
        std::vector<GLvector> manualColorList;


};




