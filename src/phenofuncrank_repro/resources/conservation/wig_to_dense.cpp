#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cstdlib>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=4)return 2;
 long n=std::stol(argv[2]),pos=-1,step=1,span=1;std::vector<float>a(n,std::numeric_limits<float>::quiet_NaN());
 std::ifstream in(argv[1]);if(!in)return 3;std::string line;bool variable=false;long covered=0;
 while(std::getline(in,line)){
  if(line.empty()||line[0]=='#'||line.rfind("track",0)==0||line.rfind("browser",0)==0)continue;
  if(line.rfind("fixedStep",0)==0||line.rfind("variableStep",0)==0){
   variable=line.rfind("variableStep",0)==0;span=1;step=1;std::istringstream ss(line);std::string x;
   while(ss>>x){auto j=x.find('=');if(j==std::string::npos)continue;auto k=x.substr(0,j);auto v=x.substr(j+1);if(k=="start")pos=std::stol(v)-1;if(k=="step")step=std::stol(v);if(k=="span")span=std::stol(v);}continue;
  }
  float value;
  if(variable){std::istringstream ss(line);long p;ss>>p>>value;pos=p-1;}else{char*end;value=std::strtof(line.c_str(),&end);if(end==line.c_str())return 4;}
  if(pos<0||pos+span>n||step<1||span<1){std::cerr<<"bounds "<<pos<<" "<<n<<std::endl;return 5;}
  for(long j=0;j<span;j++)a[pos+j]=value;covered+=span;pos+=step;
 }
 std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<char*>(a.data()),n*sizeof(float));if(!out)return 6;
 std::cout<<argv[1]<<" positions="<<n<<" assigned="<<covered<<std::endl;
}
