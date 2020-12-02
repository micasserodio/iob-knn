#include "stdlib.h"
#include "system.h"
#include "periphs.h"
#include "iob-uart.h"
#include "iob_unum4.h"
#include "iob_timer.h"
#include <stdint.h>

#ifdef DEBUG
#define N 10
#else
#define N 100
#endif


int main() {
  int32_t op = 3, overflow = 0, div_by_zero = 0, underflow = 0, passed=0, sucess_rate=0;
  
  unum4 input1[N] = {0};
  unum4 input2[N] = {0};
  
  //generate a random SEED
  srand(SEED);
   
  uart_init(UART_BASE, FREQ / BAUD);
  unum4_init(UNUM4_BASE);
 
  for(int i=0; i<N; i++) {
    input1[i] = (unum4) rand();
    input2[i] = (unum4) rand();
  
  //
  // embedded
  //

    unum4_setA(input1[i]);
    unum4_setB(input2[i]);
    unum4_setOP(op);
    unum4 x = unum4_getO();

  //  
  // pc
  //   

    Unum4Unpacked o;
    Unum4Unpacked a = unum4_unpack(input1[i]);
    Unum4Unpacked b = unum4_unpack(input2[i]);
    unum4 y;

    if (op == 0 || op == 1)
      o = unum4_add_sub(a, b, & overflow, op);
    else if (op == 2)
      o = unum4_div(a, b, & overflow, & underflow, & div_by_zero);
    else
      o = unum4_mul(a, b, & overflow, & underflow);
    
    y = unum4_pack(o, & overflow);
#ifdef DEBUG
     uart_printf("Sw result:%d \t\t Hw result:%d\t", y, x);
    if (x == y)
      uart_printf("Test passed\n");
    else
      uart_printf("Test failed\n");
    
   
    uart_txwait();
#endif    
    if(x == y)
      passed++;
   
  }  
    sucess_rate=passed/N*100;
    uart_printf("Sucess Rate[%%]: %d\n", sucess_rate);
  

}
