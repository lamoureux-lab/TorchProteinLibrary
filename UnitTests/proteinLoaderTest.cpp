#include <TH/TH.h>
#include <GlutFramework.h>
#include <cProteinLoader.h>
#include <cVector3.h>
#include <cMarchingCubes.h>

using namespace glutFramework;

int main(int argc, char** argv)
{
	GlutFramework framework;

    int spatial_dim = 120;
	cProteinLoader pL;
	THFloatTensor *grid = THFloatTensor_newWithSize4d(4,spatial_dim,spatial_dim,spatial_dim);
	int res = pL.loadPDB("/home/lupoglaz/ProteinsDataset/CASP_SCWRL/T0653/FALCON-TOPO-X_TS5");
	pL.res=1.0;
	
	pL.computeBoundingBox();
	pL.shiftProtein( -0.5*(pL.b0 + pL.b1) );
	pL.shiftProtein( 0.5*cVector3(spatial_dim, spatial_dim, spatial_dim)*pL.res ); 
	pL.assignAtomTypes(1);
	pL.projectToTensor(grid);
	
    cVolume v(grid);
    Vector<double> lookAtPos(60,60,60);
    framework.setLookAt(120, 120.0, 120.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
	framework.addObject(&v);
    framework.startFramework(argc, argv);
	
	return 0;
}