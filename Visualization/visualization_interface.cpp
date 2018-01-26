#include <TH/TH.h>
#include <cUtil.h>
#include <PracticalSocket.h>
#include <GlutFramework.h>
#include <cVector3.h>
#include <cRigidGroup.h>
#include <cConformation.h>
#include <cPDBLoader.h>
#include <string>
#include <iostream>
#include <thread>
#include <cUtil.h>
#include <iostream>
#include <string>
#include <thread>

const int RCVBUFSIZE = 128;
unsigned short echoServPort = 8080;
std::string servAddress = "127.0.0.1";
using namespace glutFramework;


void thread_tcp_listen(double *th_angles, int total_angles_length){
    TCPSocket *clntSock = NULL;
    TCPServerSocket servSock(echoServPort);
    while(1){
        try{ 
            clntSock = servSock.accept();
            double buffer[RCVBUFSIZE];
            int index = 0, totalBytesReceived=0, bytesReceived;
            while (totalBytesReceived < total_angles_length) {
                bytesReceived = clntSock->recv(buffer, RCVBUFSIZE*sizeof(double));
                if (bytesReceived <= 0) {
                    std::cerr << "Unable to read" << std::endl;;
                }
                memcpy(th_angles+totalBytesReceived, buffer, bytesReceived);
                totalBytesReceived += bytesReceived;
            }
            delete clntSock;
        } catch (SocketException &e) {
            std::cerr << e.what() << std::endl;
        }
    }
    if(clntSock!=NULL)
        delete clntSock;
}


void thread_visualization(std::string sequence){
    GlutFramework framework;
    cPDBLoader pdb;

    int length = sequence.length();
    int num_angles = 7;
    int num_atoms = pdb.getNumAtoms(sequence);
    uint total_angles_length = length*num_angles;
    double th_angles[total_angles_length];
    double th_angles_grad[length*num_angles];
    double th_atoms[num_atoms*3];

    for(int i=0;i<length;i++){
        for(int j=0;j<num_angles;j++){
            th_angles[i + length*j] = 0.0;
    }}
    
    cConformation conf(sequence, th_angles, th_angles_grad, length, th_atoms);
    ConformationVis pV(&conf);
    ConformationUpdate cU(&conf);
            
    Vector<double> lookAtPos(0,0,0);
    framework.setLookAt(20.0, 20.0, 20.0, lookAtPos.x, lookAtPos.y, lookAtPos.z, 0.0, 1.0, 0.0);
    
	framework.addObject(&pV);
    framework.addObject(&cU);

    std::thread t_listen(&thread_tcp_listen, &th_angles[0], total_angles_length);
    char ** argv;
    int argc = 0;
    framework.startFramework(argc, argv);
    t_listen.join();
}

extern "C" {
    void visSequence( const char *sequence ){
        std::string seq(sequence);
        std::thread t_vis(thread_visualization, seq);
        t_vis.detach();
    }
    void updateAngles( THDoubleTensor *angles ){
        if(angles->nDimension != 2){
            std::cerr<<"Not implemented"<<std::endl;
        }
        double *th_angles = THDoubleTensor_data(angles);
        uint total_angles_length = angles->size[0]*angles->size[1];
        try {
            TCPSocket sock(servAddress, echoServPort);
            for(int i=0;i<total_angles_length;i+=RCVBUFSIZE){
                if( (i+RCVBUFSIZE)< total_angles_length)
                    sock.send(th_angles+i, RCVBUFSIZE*sizeof(double));
                else
                    sock.send(th_angles+i, (total_angles_length-i)*sizeof(double));
            }

        } catch(SocketException &e) {
            std::cerr << e.what() << std::endl;
        }
    }
}