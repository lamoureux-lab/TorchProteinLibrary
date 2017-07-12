#include <tensorUpsample.h>
#include <iostream>
#include <math.h>
using namespace std;

float getDist(float x, float y, float z, float center){
	return sqrt( (x-center)*(x-center) + (y-center)*(y-center) + (z-center)*(z-center) );
}

float f1(int x, int y, int z){
	float center = 50;
	float r = 25; 
	if(getDist(x,y,z,center)< r){
		return 1.0;
	}
	return 0.0;
}

float f2(int x, int y, int z){
	float center = 25;
	float r = 12.5; 
	if(getDist(x,y,z,center)< r){
		return 1.0;
	}
	return 0.0;
}

void fillTensor(THFloatTensor *tensor, float (*f)(int, int, int)){
	for(int ix=0; ix<tensor->size[0];ix++){
		for(int iy=0; iy<tensor->size[1];iy++){
			for(int iz=0; iz<tensor->size[2];iz++){
				THFloatTensor_set3d(tensor, ix, iy, iz, f(ix, iy, iz));
	}}}
}

int main(void){

	cout<<"Tensor math utils test\n";

	THFloatTensor *Tref = THFloatTensor_newWithSize3d(100,100,100);
	fillTensor(Tref, f1);

	THFloatTensor *Tsrc = THFloatTensor_newWithSize3d(50,50,50);
	fillTensor(Tsrc, f2);

	THFloatTensor *Tdst = THFloatTensor_newWithSize3d(100,100,100);
	interpolateTensor(Tsrc, Tdst);

	THFloatTensor *Terror = THFloatTensor_newWithSize3d(100,100,100);
	

	THFloatTensor_cadd(Terror, Tdst, -1.0, Tref);
	THFloatTensor_abs(Tdst, Terror);
	float error = THFloatTensor_meanall(Tdst);
	cout<<"Scaling error: "<<error<<"\n";

	return 1;
}