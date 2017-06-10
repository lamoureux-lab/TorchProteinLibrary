#include "cMarchingCubes.h"
#include <math.h>
#include "marchingCubesDef.h"
#include <algorithm>

//fGetOffset finds the approximate point of intersection of the surface
// between two points with the values fValue1 and fValue2
GLfloat fGetOffset(GLfloat fValue1, GLfloat fValue2, GLfloat fValueDesired)
{
        GLdouble fDelta = fValue2 - fValue1;

        if(fDelta == 0.0)
        {
                return 0.5;
        }
        return (fValueDesired - fValue1)/fDelta;
}


//vGetColor generates a color from a given position and normal of a point
GLvoid cVolume::vGetColor(GLvector &rfColor, GLvector &rfPosition, GLvector &rfNormal)
{
        GLfloat fX = rfNormal.fX;
        GLfloat fY = rfNormal.fY;
        GLfloat fZ = rfNormal.fZ;
        rfColor.fX = (fX > 0.0 ? fX : 0.0) + (fY < 0.0 ? -0.5*fY : 0.0) + (fZ < 0.0 ? -0.5*fZ : 0.0);
        rfColor.fY = (fY > 0.0 ? fY : 0.0) + (fZ < 0.0 ? -0.5*fZ : 0.0) + (fX < 0.0 ? -0.5*fX : 0.0);
        rfColor.fZ = (fZ > 0.0 ? fZ : 0.0) + (fX < 0.0 ? -0.5*fX : 0.0) + (fY < 0.0 ? -0.5*fY : 0.0);
}

GLvoid cVolume::vNormalizeVector(GLvector &rfVectorResult, GLvector &rfVectorSource)
{
        GLfloat fOldLength;
        GLfloat fScale;

        fOldLength = sqrtf( (rfVectorSource.fX * rfVectorSource.fX) +
                            (rfVectorSource.fY * rfVectorSource.fY) +
                            (rfVectorSource.fZ * rfVectorSource.fZ) );

        if(fOldLength == 0.0)
        {
                rfVectorResult.fX = rfVectorSource.fX;
                rfVectorResult.fY = rfVectorSource.fY;
                rfVectorResult.fZ = rfVectorSource.fZ;
        }
        else
        {
                fScale = 1.0/fOldLength;
                rfVectorResult.fX = rfVectorSource.fX*fScale;
                rfVectorResult.fY = rfVectorSource.fY*fScale;
                rfVectorResult.fZ = rfVectorSource.fZ*fScale;
        }
}

//vGetNormal() finds the gradient of the scalar field at a point
//This gradient can be used as a very accurate vertx normal for lighting calculations
GLvoid cVolume::vGetNormal(int at, GLvector &rfNormal, GLfloat fX, GLfloat fY, GLfloat fZ)
{
        rfNormal.fX = fSample(at, fX-0.01, fY, fZ) - fSample(at, fX+0.01, fY, fZ);
        rfNormal.fY = fSample(at, fX, fY-0.01, fZ) - fSample(at, fX, fY+0.01, fZ);
        rfNormal.fZ = fSample(at, fX, fY, fZ-0.01) - fSample(at, fX, fY, fZ+0.01);
        vNormalizeVector(rfNormal, rfNormal);
}

//vMarchCube1 performs the Marching Cubes algorithm on a single cube
GLvoid cVolume::vMarchCube(int at, GLfloat fX, GLfloat fY, GLfloat fZ, GLfloat fScale)
{
        extern GLint aiCubeEdgeFlags[256];
        extern GLint a2iTriangleConnectionTable[256][16];

        GLint iCorner, iVertex, iVertexTest, iEdge, iTriangle, iFlagIndex, iEdgeFlags;
        GLfloat fOffset;
        GLvector sColor;
        GLfloat afCubeValue[8];
        GLvector asEdgeVertex[12];
        GLvector asEdgeNorm[12];

        //Make a local copy of the values at the cube's corners
        for(iVertex = 0; iVertex < 8; iVertex++)
        {
                afCubeValue[iVertex] = fSample(at, fX + a2fVertexOffset[iVertex][0]*fScale,
                                                   fY + a2fVertexOffset[iVertex][1]*fScale,
                                                   fZ + a2fVertexOffset[iVertex][2]*fScale);
        }

        //Find which vertices are inside of the surface and which are outside
        iFlagIndex = 0;
        for(iVertexTest = 0; iVertexTest < 8; iVertexTest++)
        {
                if(afCubeValue[iVertexTest] <= fTargetValue) 
                        iFlagIndex |= 1<<iVertexTest;
        }

        //Find which edges are intersected by the surface
        iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

        //If the cube is entirely inside or outside of the surface, then there will be no intersections
        if(iEdgeFlags == 0) 
        {
                return;
        }

        //Find the point of intersection of the surface with each edge
        //Then find the normal to the surface at those points
        for(iEdge = 0; iEdge < 12; iEdge++)
        {
                //if there is an intersection on this edge
                if(iEdgeFlags & (1<<iEdge))
                {
                        fOffset = fGetOffset(afCubeValue[ a2iEdgeConnection[iEdge][0] ], 
                                                     afCubeValue[ a2iEdgeConnection[iEdge][1] ], fTargetValue);

                        asEdgeVertex[iEdge].fX = fX + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][0]  +  fOffset * a2fEdgeDirection[iEdge][0]) * fScale;
                        asEdgeVertex[iEdge].fY = fY + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][1]  +  fOffset * a2fEdgeDirection[iEdge][1]) * fScale;
                        asEdgeVertex[iEdge].fZ = fZ + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][2]  +  fOffset * a2fEdgeDirection[iEdge][2]) * fScale;

                        vGetNormal(at, asEdgeNorm[iEdge], asEdgeVertex[iEdge].fX, asEdgeVertex[iEdge].fY, asEdgeVertex[iEdge].fZ);
                }
        }


        //Draw the triangles that were found.  There can be up to five per cube
        for(iTriangle = 0; iTriangle < 5; iTriangle++)
        {
                if(a2iTriangleConnectionTable[iFlagIndex][3*iTriangle] < 0)
                        break;

                for(iCorner = 0; iCorner < 3; iCorner++)
                {
                        iVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+iCorner];
                        //vGetColor(sColor, asEdgeVertex[iVertex], asEdgeNorm[iVertex]);
                        GLvector vertex3f, color3f, normal3f;
                        vertex3f.fX = asEdgeVertex[iVertex].fX;
                        vertex3f.fY = asEdgeVertex[iVertex].fY;
                        vertex3f.fZ = asEdgeVertex[iVertex].fZ;
                        switch(at){
                            case -1:
                                sColor.fX = 1.0; sColor.fY = 1.0; sColor.fZ = 1.0;
                                break;
                            case 0:
                                sColor.fX = 0.5; sColor.fY = 0.5; sColor.fZ = 0.5;
                                break;
                            case 1:
                                sColor.fX = 0; sColor.fY = 0; sColor.fZ = 1;
                                break;
                            case 2:
                                sColor.fX = 1; sColor.fY = 0; sColor.fZ = 0;
                                break;
                            case 3:
                                sColor.fX = 1; sColor.fY = 1; sColor.fZ = 1;
                                break;
                            case 4:
                                sColor.fX = 0.4; sColor.fY = 0.2; sColor.fZ = 0.6;
                                break;
                            case 5:
                                sColor.fX = 0.7; sColor.fY = 0.3; sColor.fZ = 0.1;
                                break;
                            case 6:
                                sColor.fX = 0.7; sColor.fY = 0.8; sColor.fZ = 0.1;
                                break;
                            case 7:
                                sColor.fX = 0.2; sColor.fY = 0.8; sColor.fZ = 0.1;
                                break;
                            case 8:
                                sColor.fX = 0.2; sColor.fY = 0.8; sColor.fZ = 0.9;
                                break;
                            case 9:
                                sColor.fX = 0.5; sColor.fY = 0.8; sColor.fZ = 0.9;
                                break;
                            case 10:
                                sColor.fX = 0.5; sColor.fY = 0.1; sColor.fZ = 0.9;
                                break;
                            case 11:
                                sColor.fX = 0.5; sColor.fY = 0.1; sColor.fZ = 0.2;
                                break;
                            default:
                                sColor.fX = 0; sColor.fY = 0; sColor.fZ = 0;
                        };
                        manualVertexList.push_back(vertex3f);
                        color3f.fX = sColor.fX;
                        color3f.fY = sColor.fY;
                        color3f.fZ = sColor.fZ;
                        manualColorList.push_back(color3f);
                        normal3f.fX = asEdgeNorm[iVertex].fX;
                        normal3f.fY = asEdgeNorm[iVertex].fY;
                        normal3f.fZ = asEdgeNorm[iVertex].fZ;
                        manualNormalList.push_back(normal3f);
                }
        }
}

//vMarchingCubes iterates over the entire dataset, calling vMarchCube on each cube
GLvoid cVolume::vMarchingCubes(int at)
{
        GLint iX, iY, iZ;
        for(iX = 0; iX < iDataSetSize; iX++)
        for(iY = 0; iY < iDataSetSize; iY++)
        for(iZ = 0; iZ < iDataSetSize; iZ++)
        {
                vMarchCube(at, iX*fStepSize, iY*fStepSize, iZ*fStepSize, fStepSize);
        }
}

GLfloat cVolume::fSample(int at, GLfloat fX, GLfloat fY, GLfloat fZ){
    int ix, iy, iz;
    if(at>=0){
        ix=fmax(0,fmin(floor(fX),data->size[1]-1));
        iy=fmax(0,fmin(floor(fY),data->size[2]-1));
        iz=fmax(0,fmin(floor(fZ),data->size[3]-1));
        float res = THFloatTensor_get4d(data, at, ix, iy, iz);
        return res;
    }else{
        ix=fmax(0,fmin(floor(fX),data->size[0]-1));
        iy=fmax(0,fmin(floor(fY),data->size[1]-1));
        iz=fmax(0,fmin(floor(fZ),data->size[2]-1));
        float res = THFloatTensor_get3d(data, ix, iy, iz);
        return res;
    }
}

cVolume::cVolume(THFloatTensor *tensor){
    fStepSize = 1.0;
    //fTargetValue = 0.5*(THFloatTensor_maxall(tensor) - THFloatTensor_minall(tensor));
    fTargetValue = 0.5;
    std::cout<<"Target value = "<<fTargetValue<<"\n";

    iDataSetSize = tensor->size[1];
    ePolygonMode = GL_LINE;

    if(tensor->nDimension == 4){
        data = tensor;  
        for(int i=0;i<data->size[0];i++)
            vMarchingCubes(i);
        std::cout<<"built gl list\n";
    }else if(tensor->nDimension == 3){
        data = tensor;
        vMarchingCubes(-1);
    }
}

cVolume::cVolume(THFloatTensor *tensor, int atom_type){
    fStepSize = 1.0;
    fTargetValue = 0.5*(THFloatTensor_maxall(tensor) - THFloatTensor_minall(tensor));
    std::cout<<"Target value = "<<fTargetValue<<"\n";

    iDataSetSize = tensor->size[1];
    ePolygonMode = GL_LINE;
    data = tensor;
    vMarchingCubes(atom_type);
}

cVolume::~cVolume(){
}

void cVolume::display(){
    
    glPushAttrib(GL_LIGHTING_BIT);
        glPolygonMode(GL_FRONT, GL_LINE);
        glPolygonMode(GL_BACK, GL_LINE);
        glDisable(GL_LIGHTING);
        glLineWidth(1);
        glBegin(GL_TRIANGLES);
            for(int i=0;i<manualVertexList.size();i++){
                glColor3f(manualColorList[i].fX,manualColorList[i].fY,manualColorList[i].fZ);
                glNormal3f(manualNormalList[i].fX,manualNormalList[i].fY,manualNormalList[i].fZ);
                glVertex3f(manualVertexList[i].fX,manualVertexList[i].fY,manualVertexList[i].fZ);
            }
        glEnd();
        glPolygonMode(GL_FRONT, GL_FILL);
        glPolygonMode(GL_BACK, GL_FILL);
    glPopAttrib(); 

}

