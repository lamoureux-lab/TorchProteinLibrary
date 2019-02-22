#include "cSO3Sampler.h"

//This code adapted from https://github.com/mlund/situs.git

#define ROT_CONV (M_PI/180.0)
void cSO3Sampler::remap_eulers (double *psi_out, double *theta_out, double *phi_out,
					 double psi_in, double theta_in, double phi_in,
					 double psi_ref, double theta_ref, double phi_ref) {
	double curr_psi, curr_theta, curr_phi;
	double new_psi, new_theta, new_phi;

	
	/* bring psi, theta, phi, within 2 M_PI of reference */
	curr_psi = psi_in - psi_ref; 
	if (curr_psi >= 0) new_psi = fmod(curr_psi,2*M_PI) + psi_ref;
	else new_psi = 2*M_PI - fmod(-curr_psi,2*M_PI) + psi_ref;
	 
	curr_theta = theta_in - theta_ref;
	if (curr_theta >= 0) new_theta = fmod(curr_theta,2*M_PI) + theta_ref;
	else new_theta = 2*M_PI - fmod(-curr_theta,2*M_PI) + theta_ref;
	
	curr_phi = phi_in - phi_ref;
	if (curr_phi >= 0) new_phi = fmod(curr_phi,2*M_PI) + phi_ref;
	else new_phi = 2*M_PI - fmod(-curr_phi,2*M_PI) + phi_ref;
	
	/* if theta is not within M_PI, we use invariant transformations */
	/* and attempt to map to above intervals */
	/* this works in most cases even if the reference is not zero*/
	if (new_theta - theta_ref > M_PI) { /* theta overflow */
		/* theta . 2 M_PI - theta */
		if (new_theta >= 0) curr_theta = fmod(new_theta,2*M_PI);
		else curr_theta = 2*M_PI - fmod(-new_theta,2*M_PI);
		new_theta -= 2 * curr_theta;
	
		/* remap to [0, 2 M_PI] interval */
		curr_theta = new_theta - theta_ref;
		if (curr_theta >= 0) new_theta = fmod(curr_theta,2*M_PI) + theta_ref;
		else new_theta = 2*M_PI - fmod(-curr_theta,2*M_PI) + theta_ref;
	
		/* we have flipped theta so we need to flip psi and phi as well */
		/* to keep rot-matrix invariant */
		/* psi . psi + M_PI */
		if (new_psi - psi_ref > M_PI) new_psi -= M_PI;
		else new_psi += M_PI;
		
		/* phi . phi + M_PI */
		if (new_phi - phi_ref > M_PI) new_phi -= M_PI;
		else new_phi += M_PI;
	}
	
	*psi_out = new_psi;
	*theta_out = new_theta;
	*phi_out = new_phi;
}

void cSO3Sampler::eu_spiral (double eu_range[3][2], double delta, unsigned long *eu_count, float  *&eu_store) {
	unsigned long i,j;
	int phi_steps,n,k;
	double phi, phi_tmp, psi_old, psi_new, psi_tmp, theta, theta_tmp, h;
	//char *program = "lib_eul";
	
	/* rough estimate of number of points on sphere that give a surface*/
	/* density that is the squared linear density of angle increments*/
	n=(int)ceil(360.0*360.0/(delta*delta*M_PI));
	
	/* total nr. points = nr. of points on the sphere * phi increments */
	theta = M_PI;
	psi_new = (eu_range[0][1]+eu_range[0][0])*0.5*ROT_CONV;
	phi_steps = 0;
	remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0,
				  eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
	if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) {
		for (phi=eu_range[2][0];phi<=eu_range[2][1];phi+=delta) {
			if (phi >= 360) break;
			remap_eulers (&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi*ROT_CONV,
						  0.0, 0.0, 0.0);
			phi_steps++;
		}
	} else {

		phi_steps = ceil((eu_range[2][1]-eu_range[2][0])/delta);
	}
	
//	std::cout<<eu_range[2][1]<<" "<<eu_range[2][0]<<" "<<delta<<" "<<(double(eu_range[2][1]-eu_range[2][0]))/delta<<endl;

	if (phi_steps<1) {
//		error_negative_euler(15080, program);
        exit(-1);
	}
	
	/* count number of points on the (theta,psi) sphere */
	j=0;
	
	/* lower pole on (theta,psi) sphere, k=0 */
	theta = M_PI;
	psi_new = (eu_range[0][1]+eu_range[0][0])*0.5*ROT_CONV;
	remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0, 
			eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
	if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) j++;
	
//	std::cout<<"Lower pole "<<j<<" elements"<<endl;
//	int lJ=j;
	
        /* intermediate sphere latitudes theta */
	psi_old = 0; /* longitude */
	for (k=1;k<n-1;k++) {
		h = -1 + 2 * k / (n-1.0); /* linear distance from pole */
		theta = acos(h);
		psi_new = psi_old + 3.6 / (sqrt ((double)n * (1-h*h))); 
		psi_old = psi_new;
		
		remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0, 
			eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
		
		if (eu_range[0][0]*ROT_CONV <= psi_new && eu_range[0][1]*ROT_CONV >= psi_new && eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) j++;
	}
	
//	std::cout<<"Intermediate sphere latitudes "<<j-lJ<<" elements"<<endl;
//	int iJ=j;
	
	 
	/* upper pole on (theta,psi) sphere, k=n-1 */
	theta = 0.0;
	psi_new = (eu_range[0][1]+eu_range[0][0])*0.5*ROT_CONV;
	remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0, 
			eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
	if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) j++;
		
//	std::cout<<"Upper pole "<<j-iJ<<" elements"<<endl;
//	int uJ=j;
//	std::cout<<"phi_steps = "<<phi_steps<<endl;

	i=phi_steps*j;
	*eu_count=i;
	// printf("Spiral Euler angle distribution, total number %lu (delta = %f deg.)\n",i,delta);
	
        /* allocate memory */

//	*eu_store = (float *) malloc(i * 3 * sizeof(float));
    eu_store = new float[i * 3];
//	std::cout<<"Allocate "<<i*3<<" elements"<<endl;
	if (eu_store == NULL) {
		throw "Failed to allocate euler angles storage";
//		error_memory_allocation(15110, program);
        // exit(-1);
	}
	
	j=0;
	/* lower pole on (theta,psi) sphere, k=0 */
	theta = M_PI;
	psi_new = (eu_range[0][1]+eu_range[0][0])*0.5*ROT_CONV;
	remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0, 
			eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
	if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) {
		for (phi=eu_range[2][0];phi<=eu_range[2][1];phi+=delta) {
                        if (phi >= 360) break; 
			remap_eulers (&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi*ROT_CONV, 
							0.0, 0.0, 0.0);
			*(eu_store+j+0)=psi_tmp;
			*(eu_store+j+1)=theta_tmp;
			*(eu_store+j+2)=phi_tmp;
			//std::cout<<j+2<<endl;
			j+=3;
		}
	}
//	std::cout<<"Lower pole "<<(j)/3<<" elements"<<endl;
//	lJ=j;

	
	/* intermediate sphere latitudes theta */
	psi_old = 0; /* longitude */
	for (k=1;k<n-1;k++) {
		h = -1 + 2 * k / (n-1.0); /* linear distance from pole */
		theta = acos(h);
		psi_new = psi_old + 3.6 / (sqrt ((double)n * (1-h*h))); 
		psi_old = psi_new; 
		remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0, 
			eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
		if (eu_range[0][0]*ROT_CONV <= psi_new && eu_range[0][1]*ROT_CONV >= psi_new && eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) {
			for (phi=eu_range[2][0];phi<=eu_range[2][1];phi+=delta) {
		                if (phi >= 360) break; 
		       		remap_eulers (&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi*ROT_CONV, 
								0.0, 0.0, 0.0);
				*(eu_store+j+0)=psi_tmp;
				*(eu_store+j+1)=theta_tmp;
				*(eu_store+j+2)=phi_tmp;
				//std::cout<<j+2<<endl;
				j+=3;
			}
		}
	}
//	std::cout<<"Intermediate sphere latitudes "<<(j-lJ)/3<<" elements"<<endl;
//	iJ=j;
	
	 
	/* upper pole on (theta,psi) sphere, k=n-1 */
	theta = 0.0;
	psi_new = (eu_range[0][1]+eu_range[0][0])*0.5*ROT_CONV;
	remap_eulers (&psi_new, &theta, &phi_tmp, psi_new, theta, 0.0, 
			eu_range[0][0]*ROT_CONV, eu_range[1][0]*ROT_CONV, eu_range[2][0]*ROT_CONV);
	if (eu_range[1][0]*ROT_CONV <= theta && eu_range[1][1]*ROT_CONV >= theta ) {
		for (phi=eu_range[2][0];phi<=eu_range[2][1];phi+=delta) {
			if (phi >= 360) break; 
		       	remap_eulers (&psi_tmp, &theta_tmp, &phi_tmp, psi_new, theta, phi*ROT_CONV, 
							0.0, 0.0, 0.0);
			*(eu_store+j+0)=psi_tmp;
			*(eu_store+j+1)=theta_tmp;
			*(eu_store+j+2)=phi_tmp;
			j+=3;
		}
	}
//	std::cout<<j-1<<endl;
//	std::cout<<"Upper pole "<<(j-iJ)/3<<" elements"<<endl;

}


cMatrix33 cSO3Sampler::convertZXZtoU (double psi, double theta, double phi) {
	double sin_psi = sin( psi );
	double cos_psi = cos( psi );
	double sin_theta = sin( theta );
	double cos_theta = cos( theta );
	double sin_phi = sin( phi);
	double cos_phi = cos( phi );
	
	cMatrix33 _U;
	
	/* use Goldstein convention */
	_U.m[0][0] = cos_psi * cos_phi - cos_theta * sin_phi * sin_psi;
	_U.m[0][1] = cos_psi * sin_phi + cos_theta * cos_phi * sin_psi;
	_U.m[0][2] = sin_psi * sin_theta;
	_U.m[1][0] = -sin_psi * cos_phi- cos_theta * sin_phi * cos_psi;
	_U.m[1][1] = -sin_psi * sin_phi+ cos_theta * cos_phi * cos_psi;
	_U.m[1][2] = cos_psi * sin_theta;
	_U.m[2][0] = sin_theta * sin_phi;
	_U.m[2][1] = -sin_theta * cos_phi;
	_U.m[2][2] =cos_theta;
	return _U;
}


cSO3Sampler::cSO3Sampler(double dAngle){
    double g_eu_range[3][2];
    g_eu_range[0][0]=0.;  g_eu_range[1][0]=0.; g_eu_range[2][0]=0.;
    g_eu_range[0][1]=360.;  g_eu_range[1][1]=180.; g_eu_range[2][1]=360.; 
    float *g_eulers = NULL;                      /* Euler angle matrix */
    unsigned long g_eulers_count;         /* number of Euler angles */
	//double *tmp = new double[13*13*13];

    eu_spiral(g_eu_range, dAngle, &g_eulers_count, g_eulers);
//	double *tmp1 = new double[13*13*13];
    for(int i=0;i<g_eulers_count;i++){
        U.push_back(convertZXZtoU(g_eulers[3*i], g_eulers[3*i+1], g_eulers[3*i+2]));
    }
    if (g_eulers)
		delete [] g_eulers;
    
}
