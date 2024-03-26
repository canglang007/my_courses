#include"lenet.h"
//该代码实现了Lenet 的各个层
//全连接层1 
void fc_1(float in[256],float fc1_w[120][256],float fc1_b[120],float out[120]){
#pragma HLS ARRAY_PARTITION variable=fc1_w cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=16 dim=1
	for (int i=0;i<120;i++){
		for(int set=0;set<16;set++){
#pragma HLS PIPELINE
			for(int j=0;j<16;j++){
				out[i]+=in[j+set*16]*fc1_w[i][j+set*16];
			}
		}
	}
	for(int i=0;i<120;i++){
		out[i]+=fc1_b[i];
		if(out[i]<0)
			out[i]=0;
	}
}
//全连接2 
void fc_2(float in[120],float fc2_w[84][120],float fc2_b[84],float out[84]){
#pragma HLS ARRAY_PARTITION variable=fc2_w cyclic factor=15 dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=15 dim=1

	for (int i=0;i<84;i++){
		for(int set=0;set<15;set++){
#pragma HLS PIPELINE
			for(int j=0;j<8;j++){
				out[i]+=in[j+set*8]*fc2_w[i][j+set*8];
			}
		}
	}
	for(int i=0;i<84;i++){
		out[i]+=fc2_b[i];
		if(out[i]<0)
			out[i]=0;
	}
}
//全连接3 
void fc_3(float in[84],float fc3_w[10][84],float fc3_b[10],float out[10]){
#pragma HLS ARRAY_PARTITION variable=fc3_w cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=4 dim=1
	for (int i=0;i<10;i++){
		for(int set=0;set<21;set++){
#pragma HLS PIPELINE
			for(int j=0;j<4;j++){
				out[i]+=in[j+set*4]*fc3_w[i][j+set*4];
			}

		}
	}
	for(int i=0;i<10;i++){
		out[i]+=fc3_b[i];
		if(out[i]<0)
			out[i]=0;
	}
}
 
//卷积块1 
void conv1(float in[28][28],float Kw[6][5][5],float Kb[6],float out[24][24][6]){
#pragma HLS ARRAY_PARTITION variable=Kb complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=3
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=1
		for(int i=0;i<24;i++){
			for(int j=0;j<24;j++){
				for(int y=0;y<5;y++){
					for(int x=0;x<5;x++){
#pragma HLS PIPELINE
						for(int k=0;k<6;k++){
							out[i][j][k]+= in[i+y][j+x]*Kw[k][y][x];
						}
					}
				}
			}
		}

		for(int i=0;i<24;i++){
			for(int j=0;j<24;j++){
#pragma HLS PIPELINE
				for(int k=0;k<6;k++){
					out[i][j][k]+= Kb[k];
					if(out[i][j][k]<0){
						out[i][j][k]=0;
					}

				}
			}
		}
}
//池化层1 
void pool1(float in[24][24][6],float out[12][12][6]){
#pragma HLS ARRAY_PARTITION variable=out complete dim=3
#pragma HLS ARRAY_PARTITION variable=in complete dim=3
	float max=0;
	for(int i=0;i<12;i++){
		for(int j=0;j<12;j++){
#pragma HLS PIPELINE
			for(int k=0;k<6;k++){
				max=0;
				for(int y=0;y<2;y++){
					for(int x=0;x<2;x++){
						if(in[2*i+y][2*j+x][k]>max){
							max=in[2*i+y][2*j+x][k];
						}
					}
				}
				out[i][j][k]=max;
			}
		}
	}

}
//卷积块2 
void conv2(float in[12][12][6],float Kw[16][6][5][5],float Kb[16],float out[8][8][16]){
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=Kb cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=Kw cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=Kw complete dim=2
#pragma HLS ARRAY_PARTITION variable=in complete dim=3
		for(int i=0;i<8;i++){
			for(int j=0;j<8;j++){
				for(int y=0;y<5;y++){
					for(int x=0;x<5;x++){
						for(int set=0;set<2;set++){
#pragma HLS PIPELINE
							for(int k=0;k<8;k++){
								for(int c=0;c<6;c++){
								out[i][j][k+set*8]+= in[i+y][j+x][c]*Kw[k+set*8][c][y][x];
								}
							}
						}
					}
				}
			}
		}
		for(int i=0;i<8;i++){
			for(int j=0;j<8;j++){
				for(int set=0;set<2;set++){
#pragma HLS PIPELINE
					for(int k=0;k<8;k++){
					out[i][j][k+set*8]+=Kb[k+set*8];
					if(out[i][j][k+set*8]<0){
						out[i][j][k+set*8]=0;
						}
					}
				}
			}
		}
}

//池化层1 
void pool1(float in[24][24][6],float out[12][12][6]){
#pragma HLS ARRAY_PARTITION variable=out complete dim=3
#pragma HLS ARRAY_PARTITION variable=in complete dim=3
	float max=0;
	for(int i=0;i<12;i++){
		for(int j=0;j<12;j++){
#pragma HLS PIPELINE
			for(int k=0;k<6;k++){
				max=0;
				for(int y=0;y<2;y++){
					for(int x=0;x<2;x++){
						if(in[2*i+y][2*j+x][k]>max){
							max=in[2*i+y][2*j+x][k];
						}
					}
				}
				out[i][j][k]=max;
			}
		}
	}

}


//池化层2 
void pool2(float in[8][8][16],float out[4][4][16]){
#pragma HLS ARRAY_PARTITION variable=out cyclic factor=8 dim=3
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=8 dim=3
	float max=0;
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
#pragma HLS PIPELINE
			for(int k=0;k<16;k++){
				max=0;
				for(int y=0;y<2;y++){
					for(int x=0;x<2;x++){
						if(in[2*i+y][2*j+x][k]>max){
							max=in[2*i+y][2*j+x][k];
						}
					}
				}
				out[i][j][k]=max;
			}
		}
	}

}

//展平层 
void reshape(float in[4][4][16],float out[256]){
	int o=0;
	for(int k=0;k<16;k++){
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				out[o]=in[i][j][k];
				o++;
			}
		}
	}
}
//全连接层1 
void fc_1(float in[256],float fc1_w[120][256],float fc1_b[120],float out[120]){
#pragma HLS ARRAY_PARTITION variable=fc1_w cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=16 dim=1
	for (int i=0;i<120;i++){
		for(int set=0;set<16;set++){
#pragma HLS PIPELINE
			for(int j=0;j<16;j++){
				out[i]+=in[j+set*16]*fc1_w[i][j+set*16];
			}
		}
	}
	for(int i=0;i<120;i++){
		out[i]+=fc1_b[i];
		if(out[i]<0)
			out[i]=0;
	}
}
//全连接2 
void fc_2(float in[120],float fc2_w[84][120],float fc2_b[84],float out[84]){
#pragma HLS ARRAY_PARTITION variable=fc2_w cyclic factor=15 dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=15 dim=1

	for (int i=0;i<84;i++){
		for(int set=0;set<15;set++){
#pragma HLS PIPELINE
			for(int j=0;j<8;j++){
				out[i]+=in[j+set*8]*fc2_w[i][j+set*8];
			}
		}
	}
	for(int i=0;i<84;i++){
		out[i]+=fc2_b[i];
		if(out[i]<0)
			out[i]=0;
	}
}
//全连接3 
void fc_3(float in[84],float fc3_w[10][84],float fc3_b[10],float out[10]){
#pragma HLS ARRAY_PARTITION variable=fc3_w cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=in cyclic factor=4 dim=1
	for (int i=0;i<10;i++){
		for(int set=0;set<21;set++){
#pragma HLS PIPELINE
			for(int j=0;j<4;j++){
				out[i]+=in[j+set*4]*fc3_w[i][j+set*4];
			}

		}
	}
	for(int i=0;i<10;i++){
		out[i]+=fc3_b[i];
		if(out[i]<0)
			out[i]=0;
	}
}



