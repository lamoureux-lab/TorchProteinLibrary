#include "cProteinLoader.h"
#include <iostream>
#include <string>
#include "TH/TH.h"
#include <math.h>
extern "C"{
	void loadProtein(const char* proteinPath, THFloatTensor *grid, bool shift, bool rot, float resolution, THGenerator *gen, bool binary){
		bool destroy_generator = false;
		if(gen==NULL){
			gen = THGenerator_new();
 			THRandom_seed(gen);
 			destroy_generator=true;
		}
		cProteinLoader pL;
		if(binary){
			pL.load_binary(proteinPath);
		}else{
		
			pL.loadPDB(proteinPath);
			if(grid->size[0]==4){
				pL.assignAtomTypes(1);
			}else if(grid->size[0]==11){
				pL.assignAtomTypes(2);
			}else{
				std::cout<<"Wrong number of input features: "<<grid->size[0]<<"\n";
				return;
			}
		}
		pL.res = resolution;
		pL.computeBoundingBox();
		//placing center of the bbox to the origin
		pL.shiftProtein( -0.5*(pL.b0 + pL.b1) ); 
		if(rot){
			float alpha = THRandom_uniform(gen,0,2.0*M_PI);
 			float beta = THRandom_uniform(gen,0,2.0*M_PI);
 			float theta = THRandom_uniform(gen,0,2.0*M_PI);
 			cMatrix33 random_rotation = cMatrix33::rotationXYZ(alpha,beta,theta);
			pL.rotateProtein(random_rotation);
		}
		if(shift){
			float dx_max = fmax(0, grid->size[1]*pL.res/2.0 - (pL.b1[0]-pL.b0[0])/2.0)*0.5;
			float dy_max = fmax(0, grid->size[2]*pL.res/2.0 - (pL.b1[1]-pL.b0[1])/2.0)*0.5;
			float dz_max = fmax(0, grid->size[3]*pL.res/2.0 - (pL.b1[2]-pL.b0[2])/2.0)*0.5;
			float dx = THRandom_uniform(gen,-dx_max,dx_max);
		 	float dy = THRandom_uniform(gen,-dy_max,dy_max);
		 	float dz = THRandom_uniform(gen,-dz_max,dz_max);
		 	pL.shiftProtein(cVector3(dx,dy,dz));
		}
		// placing center of the protein to the center of the grid
		pL.shiftProtein( 0.5*cVector3(grid->size[1],grid->size[2],grid->size[3])*pL.res ); 
		pL.projectToTensor(grid);

		if(destroy_generator){
			THGenerator_free(gen);
		}
	}

	typedef struct{
		char **strings;
		size_t len, ind;
		THFloatTensor **grids4D;
	} batchInfo;

	batchInfo* createBatchInfo(int batch_size){
		//std::cout<<"Creating batch info of size = "<<batch_size<<"\n";
		batchInfo *binfo;
		binfo = new batchInfo;
		binfo->strings = new char*[batch_size];
		binfo->grids4D = new THFloatTensor*[batch_size];
		binfo->len = batch_size;
		binfo->ind = 0;
		return binfo;
	}

	void deleteBatchInfo(batchInfo* binfo){
		for(int i=0;i<binfo->len;i++){
			delete [] binfo->strings[i];
		}
		binfo->len=0;
		binfo->ind=0;
		delete binfo;
	}

	void pushProteinToBatchInfo(const char* filename, THFloatTensor *grid4D, batchInfo* binfo){
		std::string str(filename);
		//std::cout<<"Pushing the string "<<str<<" to the position "<<pos<<"\n";
		//std::cout<<grid4D->nDimension<<"\n";

		binfo->strings[binfo->ind] = new char[str.length()+1];
		for(int i=0; i<str.length(); i++){
			binfo->strings[binfo->ind][i] = str[i];
		}
		binfo->strings[binfo->ind][str.length()]='\0';
		binfo->grids4D[binfo->ind] = grid4D;
		binfo->ind += 1;
		//std::cout<<binfo->grids4D[pos]->nDimension<<"\n";
	}

	void printBatchInfo(batchInfo* binfo){
		for(int i=0;i<binfo->len;i++){
			std::cout<<binfo->strings[i]<<"\n";
			std::cout<<binfo->grids4D[i]->nDimension<<"\n";
		}
	}

	void loadProteinOMP(batchInfo* batch, bool shift, bool rot, float resolution, bool binary){
		THGenerator *gen = THGenerator_new();
 		THRandom_seed(gen);

		#pragma omp parallel for num_threads(10)
		for(int i=0; i<batch->len; i++){
			loadProtein(batch->strings[i], batch->grids4D[i], shift, rot, resolution, gen, binary);
		}
		THGenerator_free(gen);
	}
}