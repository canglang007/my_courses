#include"lenet.h"
#include<iostream>
using namespace std;
//ʵ��һ���򵥵Ĳ��Թ��� 
int main(){
float in[28][28] ={
#include"image3.h"
};
float OUT[10] ={};

	LeNet_PYNQ(in,OUT);
	for(int i=0;i<10;i++){
		cout<<OUT[i]<<endl;
	}
}
