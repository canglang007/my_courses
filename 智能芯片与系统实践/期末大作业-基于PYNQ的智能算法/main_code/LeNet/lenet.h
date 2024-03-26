#ifndef _LENET_H_
#define _LENET_H_

void conv1(float in[28][28],float Kw[6][5][5],float Kb[6],float out[24][24][6]);
void conv2(float in[12][12][6],float Kw[16][6][5][5],float Kb[16],float out[8][8][16]);
void pool1(float in[24][24][6],float out[12][12][6]);
void pool2(float in[8][8][16],float out[4][4][16]);
void reshape(float in[4][4][16],float out[256]);
void fc_1(float in[256],float fc1_w[120][256],float fc1_b[120],float out[120]);
void fc_2(float in[120],float fc2_w[84][120],float fc2_b[84],float out[84]);
void fc_3(float in[84],float fc3_w[10][84],float fc3_b[10],float out[10]);
void LeNet_PYNQ(float img[28][28],float OUT[10]);


#endif 
