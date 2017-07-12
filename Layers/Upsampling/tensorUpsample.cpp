#include <tensorUpsample.h>
#include <cVector3.h>
#include <math.h>

typedef struct{
	int x, y, z;
} cVector3Idx;

float bilinear(const float tx, const float ty, const float c00, const float c10, const float c01, const float c11)
{ 
	float a = c00 * (1 - tx) + c10 * tx;
	float b = c01 * (1 - tx) + c11 * tx;
	return a * (1 - ty) + b * ty;
}

cVector3Idx getTensorIndexes(cVector3 r){
	cVector3Idx ri;
	ri.x = int(r.v[0]);
	ri.y = int(r.v[1]);
	ri.z = int(r.v[2]);

	return ri;
}

float getValue(THFloatTensor *data, int ix, int iy, int iz){
	if( (ix>=0)&&(ix<data->size[0]) && (iy>=0)&&(iy<data->size[1]) && (iz>=0)&&(iz<data->size[2]) ){
		return THFloatTensor_get3d(data, ix,iy,iz);
	}else{
		return 0.0;
	}
}

float interpolate(THFloatTensor *data, const cVector3& location) 
{ 
	float gx, gy, gz, tx, ty, tz;
	unsigned gxi, gyi, gzi; 
	// remap point coordinates to grid coordinates
	cVector3Idx gri = getTensorIndexes(location);
	
	tx = location.v[0] - gri.x; 
	ty = location.v[1] - gri.y; 
	tz = location.v[2] - gri.z; 
	
	const float c000 = getValue(data, gri.x, gri.y, gri.z);//data[IX(gei, gyi, gzi)]; 
	const float c100 = getValue(data, gri.x+1, gri.y, gri.z);//data[IX(gxi + 1, gyi, gzi)]; 
	const float c010 = getValue(data, gri.x, gri.y+1, gri.z);//data[IX(gxi, gyi + 1, gzi)]; 
	const float c110 = getValue(data, gri.x+1, gri.y+1, gri.z);//data[IX(gxi + 1, gyi + 1, gzi)]; 
	const float c001 = getValue(data, gri.x, gri.y, gri.z+1);//data[IX(gxi, gyi, gzi + 1)]; 
	const float c101 = getValue(data, gri.x+1, gri.y, gri.z+1);//data[IX(gxi + 1, gyi, gzi + 1)]; 
	const float c011 = getValue(data, gri.x, gri.y+1, gri.z+1);//data[IX(gxi, gyi + 1, gzi + 1)]; 
	const float c111 = getValue(data, gri.x+1, gri.y+1, gri.z+1);//data[IX(gxi + 1, gyi + 1, gzi + 1)]; 

	float e = bilinear(tx, ty, c000, c100, c010, c110); 
	float f = bilinear(tx, ty, c001, c101, c011, c111); 
	return e * ( 1 - tz) + f * tz;

} 
	 
void interpolateTensor(THFloatTensor *src, THFloatTensor *dst){
	cVector3 dst_r, src_r;
	double scale_x = (float(src->size[0])/float(dst->size[0]));
	double scale_y = (float(src->size[1])/float(dst->size[1]));
	double scale_z = (float(src->size[2])/float(dst->size[2]));
	
	for(int ix=0; ix<dst->size[0];ix++){
		for(int iy=0; iy<dst->size[1];iy++){
			for(int iz=0; iz<dst->size[2];iz++){
				dst_r = cVector3(ix,iy,iz);
				src_r.v[0] = dst_r.v[0]*scale_x;
				src_r.v[1] = dst_r.v[1]*scale_y;
				src_r.v[2] = dst_r.v[2]*scale_z;
				float interpolated_value = interpolate(src, src_r);
				THFloatTensor_set3d(dst, ix, iy, iz, interpolated_value);
			}
		}
	}
}