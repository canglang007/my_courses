#include"lenet.h"

//输入为28*28，输出为10，该函数实现的是前向推理过程
 
void LeNet_PYNQ(float img[28][28],float OUT[10]){
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE m_axi depth=784 port=img offset=slave bundle=IMG
#pragma HLS INTERFACE m_axi depth=10 port=OUT offset=slave bundle=OUT
//下面是导入各层的权重 
float c1_w[6][5][5] ={
#include"c1_w.h"
};
float c1_b[6] ={
#include"c1_b.h"
};
float c2_w[16][6][5][5] ={
#include"c2_w.h"
};
float c2_b[16] ={
#include"c2_b.h"
};
float fc1_b[120] ={
#include"fc1_b.h"
};
float fc2_b[84] ={
#include"fc2_b.h"
};
float fc3_w[10][84] ={
#include"fc3_w.h"
};
float fc3_b[10] ={
#include"fc3_b.h"
};
float fc2_w[84][120] ={
#include"fc2_w.h"
};
float fc1_w[120][256] ={
#include"fc1_w.h"
};

	float c1_out[24][24][6];
	float p1_out[12][12][6];
	float c2_out[8][8][16];
	float p2_out[4][4][16];
	float r_out[256];
	float f1_out[120];
	float f2_out[84];


	conv1(img,c1_w,c1_b,c1_out);
	pool1(c1_out,p1_out);
	conv2(p1_out,c2_w,c2_b,c2_out);
	pool2(c2_out,p2_out);
	reshape(p2_out,r_out);
	fc_1(r_out,fc1_w,fc1_b,f1_out);
	fc_2(f1_out,fc2_w,fc2_b,f2_out);
	fc_3(f2_out,fc3_w,fc3_b,OUT);

}
