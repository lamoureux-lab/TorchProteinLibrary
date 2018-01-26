#include <cUtil.h>

void RigidGroupVis::display(){
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glPointSize(8.0);
    glBegin(GL_POINTS);
        
        for(int i=0; i<rg->atoms_global.size(); i++){
            float x = rg->atoms_global[i].v[0];
            float y = rg->atoms_global[i].v[1];
            float z = rg->atoms_global[i].v[2];
            if(rg->atomNames[i][0]=='C' && rg->atomNames[i]!="CA"){
                glColor3f(0.0,0.5,0.0);
                glVertex3f(x,y,z);
            }
            if(rg->atomNames[i][0]=='O'){
                glColor3f(0.8,0.,0.);
                glVertex3f(x,y,z);
            }
            if(rg->atomNames[i]=="CA"){
                glColor3f(0.0,0.8,0.0);
                glVertex3f(x,y,z);
            }
            if(rg->atomNames[i][0]=='N'){
                glColor3f(0,0,0.8);
                glVertex3f(x,y,z);
            }
            if(rg->atomNames[i][0]=='S'){
                glColor3f(0.9,0.9,0.0);
                glVertex3f(x,y,z);
            }
            
        }
    glEnd();
    if(rg->atoms_global.size()>2 && rg->atoms_global.size()<8){
        glBegin(GL_POLYGON);
            glColor3f(0.5,0.5,0.5);
            for(int i=0; i<rg->atoms_global.size(); i++){
                float x = rg->atoms_global[i].v[0];
                float y = rg->atoms_global[i].v[1];
                float z = rg->atoms_global[i].v[2];
                glVertex3f(x,y,z);
            }
        glEnd();
    }else if(rg->atoms_global.size()>1){
        
        glBegin(GL_LINES);
        for(int i=0; i<rg->atoms_global.size()-1; i++){
            for(int j=i+1; j<rg->atoms_global.size(); j++){
                glColor3f(0.5,0.5,0.5);
                float x = rg->atoms_global[i].v[0];
                float y = rg->atoms_global[i].v[1];
                float z = rg->atoms_global[i].v[2];
                glVertex3f(x,y,z);
                x = rg->atoms_global[j].v[0];
                y = rg->atoms_global[j].v[1];
                z = rg->atoms_global[j].v[2];
                glVertex3f(x,y,z);
            }
        }
        glEnd();
        
    }
    
    glPopAttrib();
};

void ConformationVis::walk(cNode *node){
    if(node->left!=NULL){
        cVector3 x0 = node->group->atoms_global[0];
        cVector3 x1 = node->left->group->atoms_global[0];
        glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
        glVertex3f(x1.v[0],x1.v[1],x1.v[2]);
        walk(node->left);
    }
    if(node->right!=NULL){
        cVector3 x0 = node->group->atoms_global[0];
        cVector3 x1 = node->right->group->atoms_global[0];
        glVertex3f(x0.v[0],x0.v[1],x0.v[2]);
        glVertex3f(x1.v[0],x1.v[1],x1.v[2]);
        walk(node->right);
    }
}
void ConformationVis::display(){
    
    glPushAttrib(GL_LIGHTING_BIT);
    glDisable(GL_LIGHTING);
    glLineWidth(2);
    glBegin(GL_LINES);
        glColor3f(1.0,1.0,1.0);
        walk(c->root);
    glEnd();
    glPopAttrib();

    for(int i=0; i<groupsVis.size(); i++){
        groupsVis[i].display();
    }
};