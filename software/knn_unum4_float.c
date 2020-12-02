#include "system.h"
#include "periphs.h"
#include <iob-uart.h>
#include "iob_timer.h"
#include "iob_knn.h"
#include "math.h"
#include "stdlib.h"
#include "iob_unum4.h"
#include "stdint.h"




#ifdef DEBUG //type make DEBUG=1 to print debug info
#define S 12  //random seed
#define N 10  //data set size
#define K 4   //number of neighbours (K)
#define C 4   //number data classes
#define M 4   //number samples to be classified
#else
#define S 12   
#define N 100000
#define K 10  
#define C 4  
#define M 100

#endif


//
//Data structures
//

//labeled dataset (float)
struct datum_float{
  float x;
  float y;
  unsigned char label;
} data_float[N], x_float[M];

//neighbor info (float)
struct neighbor_float {
  unsigned int idx; //index in dataset array
  float dist; //distance to test point
} neighbor_float[K];


//labeled dataset (unum4)
struct datum_unum4 {
  unum4 x;
  unum4 y;
  unsigned char label;
} data_unum4[N], x_unum4[M];

//neighbor info (unum4)
struct neighbor_unum4 {
  unsigned int idx; //index in dataset array
  unum4 dist; //distance to test point
} neighbor_unum4[K];

//
//Functions

//double random generator
double random_real(double min, double max) {
  double d = (double) rand()/RAND_MAX;
  return (min+ d*(max-min));
}


//square distance between 2 points a and b (float)
float sq_dist_float( struct datum_float a, struct datum_float b) {
  float X = a.x-b.x;
  float X2=X*X;
  float Y = a.y-b.y;
  float Y2=Y*Y;
  return (X2 + Y2);
}

//insert element in ordered array of neighbours (float)
void insert_float (struct neighbor_float element, unsigned int position) {
  for (int j=K-1; j>position; j--)
    neighbor_float[j] = neighbor_float[j-1];

  neighbor_float[position] = element;

}


//square distance between 2 points a and b (unum4)
unum4 sq_dist_unum4( struct datum_unum4 a, struct datum_unum4 b) {
  int32_t overflow=0, underflow =0;
  Unum4Unpacked X = unum4_add_sub(unum4_unpack(a.x), unum4_unpack(b.x), &overflow, 1);
  unum4 x = unum4_pack(X,&overflow); //a.x-b.x
  Unum4Unpacked X2= unum4_mul(unum4_unpack(x),unum4_unpack(x), &overflow, &underflow);
  unum4 x2 = unum4_pack(X2,&overflow); //x²
  Unum4Unpacked Y = unum4_add_sub(unum4_unpack(a.y), unum4_unpack(b.y), &overflow, 1);
  unum4 y = unum4_pack(Y,&overflow); //a.y-b.y
  Unum4Unpacked Y2=unum4_mul(unum4_unpack(y),unum4_unpack(y), &overflow, &underflow);
  unum4 y2 = unum4_pack(Y2,&overflow); //y²
  return unum4_pack(unum4_add_sub(unum4_unpack(x2),unum4_unpack(y2), &overflow, 0), &overflow);
}

//insert element in ordered array of neighbours (unum4)
void insert_unum4 (struct neighbor_unum4 element, unsigned int position) {
  for (int j=K-1; j>position; j--)
    neighbor_unum4[j] = neighbor_unum4[j-1];

  neighbor_unum4[position] = element;

}

///////////////////////////////////////////////////////////////////
void knn_float(double random[], double test_points[], unsigned char label_rand[], int votes_acc[]) {


  //init dataset
  for (int i=0; i<N; i++) {

   
    //init coordinates
    data_float[i].x = (float) random[i];
    data_float[i].y = (float) random[N+i];

    //init label
    data_float[i].label = label_rand[i];
  }

#ifdef DEBUG
  uart_printf("\n\n\nDATASET\n");
  uart_printf("Idx \tX \tY \tLabel\n");
  for (int i=0; i<N; i++)
    uart_printf("%d \t%f \t%f \t%d\n", i, data_float[i].x,  data_float[i].y, data_float[i].label);
#endif
  
  //init test points
  for (int k=0; k<M; k++) {
    x_float[k].x  = (float) test_points[k];
    x_float[k].y  = (float) test_points[M+k];
    //x[k].label will be calculated by the algorithm
  }

#ifdef DEBUG
  uart_printf("\n\nTEST POINTS\n");
  uart_printf("Idx \tX \tY\n");
  for (int k=0; k<M; k++)
    uart_printf("%d \t%f \t%f\n", k, x_float[k].x, x_float[k].y);
#endif
  
  //
  // PROCESS DATA
  //

  //start knn here
  
  for (int k=0; k<M; k++) { //for all test points
  //compute distances to dataset points

#ifdef DEBUG
    uart_printf("\n\nProcessing x[%d]:\n", k);
#endif

    //init all k neighbors infinite distance
    for (int j=0; j<K; j++)
      neighbor_float[j].dist = INFINITY;

#ifdef DEBUG
    uart_printf("Datum \tX \tY \tLabel \tDistance\n");
#endif
    for (int i=0; i<N; i++) { //for all dataset points
      //compute distance to x[k]
      float d = sq_dist_float(x_float[k], data_float[i]);

      //insert in ordered list
      for (int j=0; j<K; j++) 
        if ( d < neighbor_float[j].dist ) {
          insert_float( (struct neighbor_float){i,d}, j);
          break;
        }

#ifdef DEBUG
      //dataset
      uart_printf("%d \t%f \t%f \t%d \t%f\n", i, data_float[i].x, data_float[i].y, data_float[i].label, d);
#endif

    }

    
    //classify test point

    //clear all votes
    int votes[C] = {0};
    int best_votation = 0;
    int best_voted = 0;

    //make neighbours vote
    for (int j=0; j<K; j++) { //for all neighbors
      if ( (++votes[data_float[neighbor_float[j].idx].label]) > best_votation ) {
        best_voted = data_float[neighbor_float[j].idx].label;
        best_votation = votes[best_voted];
      }
    }

    x_float[k].label = best_voted;

    votes_acc[best_voted]++;
    
#ifdef DEBUG
    uart_printf("\n\nNEIGHBORS of x[%d]=(%f, %f):\n", k, x_float[k].x, x_float[k].y);
    uart_printf("K \tIdx \tX \tY \tDist \t\tLabel\n");
    for (int j=0; j<K; j++)
      uart_printf("%d \t%d \t%f \t%f \t%f \t%d\n", j+1, neighbor_float[j].idx, data_float[neighbor_float[j].idx].x,  data_float[neighbor_float[j].idx].y, neighbor_float[j].dist,  data_float[neighbor_float[j].idx].label);
    
    uart_printf("\n\nCLASSIFICATION of x[%d]:\n", k);
    uart_printf("X \tY \tLabel\n");
    uart_printf("%f \t%f \t%d\n\n\n", x_float[k].x, x_float[k].y, x_float[k].label);

#endif

  } //all test points classified

  
  //print classification distribution to check for statistical bias
  for (int l=0; l<C; l++)
    uart_printf("%d ", votes_acc[l]);
  uart_printf("\n");
 
  
}


///////////////////////////////////////////////////////////////////
void knn_unum4(double random[], double test_points[], unsigned char label_rand[], int votes_acc[]) {

  int32_t failed,overflow=0;
  Unum4Unpacked dist;

  
 
  //init dataset
  for (int i=0; i<N; i++) {

    //init coordinates
   
    data_unum4[i].x = double2unum4(*(int64_t*)&random[i],&failed);
    data_unum4[i].y = double2unum4(*(int64_t*)&random[N+i],&failed);

    //init label
    data_unum4[i].label = label_rand[i];
  }

#ifdef DEBUG
  uart_printf("\n\n\nDATASET\n");
  uart_printf("Idx \tX \tY \tLabel\n");
  for (int i=0; i<N; i++)
    uart_printf("%d \t%d \t%d \t%d\n", i, data_unum4[i].x,  data_unum4[i].y, data_unum4[i].label);
#endif
  
  //init test points
  for (int k=0; k<M; k++) {

    x_unum4[k].x  = double2unum4(*(int64_t*)&test_points[k],&failed);
    x_unum4[k].y  = double2unum4(*(int64_t*)&test_points[M+k],&failed);
    //x[k].label will be calculated by the algorithm
  }

#ifdef DEBUG
  uart_printf("\n\nTEST POINTS\n");
  uart_printf("Idx \tX \tY\n");
  for (int k=0; k<M; k++)
    uart_printf("%d \t%d \t%d\n", k, x_unum4[k].x, x_unum4[k].y);
#endif
  
  //
  // PROCESS DATA
  //

  //start knn here
  
  for (int k=0; k<M; k++) { //for all test points
  //compute distances to dataset points

#ifdef DEBUG
    uart_printf("\n\nProcessing x[%d]:\n", k);
#endif

    //init all k neighbors infinite distance
    for (int j=0; j<K; j++)
      neighbor_unum4[j].dist = -1; //the higher unum4 number

#ifdef DEBUG
    uart_printf("Datum \tX \tY \tLabel \tDistance\n");
#endif
    for (int i=0; i<N; i++) { //for all dataset points
      //compute distance to x[k]
      unum4 d = sq_dist_unum4(x_unum4[k], data_unum4[i]);

      //insert in ordered list
      for (int j=0; j<K; j++) {
        dist = unum4_add_sub(unum4_unpack(d),unum4_unpack(neighbor_unum4[j].dist), &overflow,1);
        if (~overflow && dist.mantissa <0){
          insert_unum4( (struct neighbor_unum4){i,d}, j);
          break;
        }
      }
#ifdef DEBUG
      //dataset
      uart_printf("%d \t%d \t%d \t%d \t%d\n", i, data_unum4[i].x, data_unum4[i].y, data_unum4[i].label, d);
#endif

    }

    
    //classify test point

    //clear all votes
    int votes[C] = {0};
    int best_votation = 0;
    int best_voted = 0;

    //make neighbours vote
    for (int j=0; j<K; j++) { //for all neighbors
      if ( (++votes[data_unum4[neighbor_unum4[j].idx].label]) > best_votation ) {
        best_voted = data_unum4[neighbor_unum4[j].idx].label;
        best_votation = votes[best_voted];
      }
    }

    x_unum4[k].label = best_voted;

    votes_acc[best_voted]++;
    
#ifdef DEBUG
    uart_printf("\n\nNEIGHBORS of x[%d]=(%d, %d):\n", k, x_unum4[k].x, x_unum4[k].y);
    uart_printf("K \tIdx \tX \tY \tDist \t\tLabel\n");
    for (int j=0; j<K; j++)
      uart_printf("%d \t%d \t%d \t%d \t%d \t%d\n", j+1, neighbor_unum4[j].idx, data_unum4[neighbor_unum4[j].idx].x,  data_unum4[neighbor_unum4[j].idx].y, neighbor_unum4[j].dist,  data_unum4[neighbor_unum4[j].idx].label);
    
    uart_printf("\n\nCLASSIFICATION of x[%d]:\n", k);
    uart_printf("X \tY \tLabel\n");
    uart_printf("%d \t%d \t%d\n\n\n", x_unum4[k].x, x_unum4[k].y, x_unum4[k].label);

#endif

  } //all test points classified


  
  //print classification distribution to check for statistical bias
  for (int l=0; l<C; l++)
    uart_printf("%d ", votes_acc[l]);
  uart_printf("\n");
 
  
}

int knn_float_unum4 () {

  unsigned long long elapsed;
  unsigned int elapsedu;
  unsigned char label_rand[N];
  double random[2*N];
  double test_points[2*M];
  int votes_acc_float[C] = {0};
  int votes_acc_unum4[C] = {0};
  
  
  //init uart and timer
  uart_init(UART_BASE, FREQ/BAUD);
  uart_printf("\nInit timer\n");
  uart_txwait();

  timer_init(TIMER_BASE);
  //read current timer count, compute elapsed time
  //elapsed  = timer_get_count();
  //elapsedu = timer_time_us();
  
  
  //generate random seed 
   srand(SEED);
   
   for (int i=0; i<2*N; i++) 
     random[i] = random_real(-100.0,100.0); 
     
   for (int t=0; t<N; t++) 
     label_rand[t] = (unsigned char) (rand()%C);
     
   for (int k=0; k<2*M; k++)
     test_points[k] = random_real(-100.0,100.0);
   
   
   knn_float(random,test_points,label_rand,votes_acc_float);
   knn_unum4(random,test_points,label_rand,votes_acc_unum4);
   
   for (int n=0; n<C; n++){
     if(votes_acc_float[n]!=votes_acc_unum4[n])
       break;
     uart_printf("Passed: %d\n",n);
   }
   
  //stop knn here
  //read current timer count, compute elapsed time
  elapsedu = timer_time_us(TIMER_BASE);
  uart_printf("\nExecution time: %dus @%dMHz\n\n", elapsedu, FREQ/1000000);


}
