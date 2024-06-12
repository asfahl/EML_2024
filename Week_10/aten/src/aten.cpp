#include <cstdlib>
#include <ATen/ATen.h>
#include <iostream>

int main() {
  std::cout << "running the ATen examples" << std::endl;

  float l_data[4*2*3] = {  0.0f,  1.0f,  2.0f, 
                           3.0f,  4.0f,  5.0f,

                           6.0f,  7.0f,  8.0f, 
                           9.0f, 10.0f, 11.0f,
                           
                          12.0f, 13.0f, 14.0f,
                          15.0f, 16.0f, 17.0f,
                          
                          18.0f, 19.0f, 20.0f,
                          21.0f, 22.0f, 23.0f };

  std::cout << "l_data (ptr): " << l_data << std::endl;

  // TODO: Add ATen code

  std::cout << "finished running ATen examples" << std::endl;

  return EXIT_SUCCESS;
}
