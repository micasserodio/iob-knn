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

//labeled dataset
struct datum_double{
  double x;
  double y;
  unsigned char label;
} data_double[N], x_double[M];

//neighbor info
struct neighbor_double {
  unsigned int idx; //index in dataset array
  double dist; //distance to test point
} neighbor_double[K];

//
//Functions
//

//double random generator
double random_double(double min, double max) {
  double d = (double) rand()/RAND_MAX;
  return (min+ d*(max-min));
}

//square distance between 2 points a and b
double sq_dist_double( struct datum_double a, struct datum_double b) {
  double X = a.x-b.x;
  double X2=X*X;
  double Y = a.y-b.y;
  double Y2=Y*Y;
  return (X2 + Y2);
}

//insert element in ordered array of neighbours
void insert_double (struct neighbor_double element, unsigned int position) {
  for (int j=K-1; j>position; j--)
    neighbor_double[j] = neighbor_double[j-1];

  neighbor_double[position] = element;

}


///////////////////////////////////////////////////////////////////
int knn_double() {

  unsigned long long elapsed;
  unsigned int elapsedu;

  //init uart and timer
  uart_init(UART_BASE, FREQ/BAUD);
  uart_printf("\nInit timer\n");
  uart_txwait();

  timer_init(TIMER_BASE);
  //read current timer count, compute elapsed time
  //elapsed  = timer_get_count();
  //elapsedu = timer_time_us();


  //int vote accumulator
  int votes_acc[C] = {0};

  //generate random seed 
  srand(SEED);

  //init dataset
  for (int i=0; i<N; i++) {

    //init coordinates
    data_double[i].x = random_double(-100.0,100.0); 
    data_double[i].y = random_double(-100.0,100.0);

    //init label
    data_double[i].label = (unsigned char) (rand()%C);
  }

#ifdef DEBUG
  uart_printf("\n\n\nDATASET\n");
  uart_printf("Idx \tX \tY \tLabel\n");
  for (int i=0; i<N; i++)
    uart_printf("%d \t%lld \t%lld \t%d\n", i, data_double[i].x,  data_double[i].y, data_double[i].label);
#endif
  
  //init test points
  for (int k=0; k<M; k++) {
    x_double[k].x  = random_double(-100.0,100.0); 
    x_double[k].y  = random_double(-100.0,100.0); 
    //x[k].label will be calculated by the algorithm
  }

#ifdef DEBUG
  uart_printf("\n\nTEST POINTS\n");
  uart_printf("Idx \tX \tY\n");
  for (int k=0; k<M; k++)
    uart_printf("%d \t%lld \t%lld\n", k, x_double[k].x, x_double[k].y);
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
      neighbor_double[j].dist = INFINITY;

#ifdef DEBUG
    uart_printf("Datum \tX \tY \tLabel \tDistance\n");
#endif
    for (int i=0; i<N; i++) { //for all dataset points
      //compute distance to x[k]
      double d = sq_dist_double(x_double[k], data_double[i]);

      //insert in ordered list
      for (int j=0; j<K; j++)
        if ( d < neighbor_double[j].dist ) {
          insert_double( (struct neighbor_double){i,d}, j);
          break;
        }

#ifdef DEBUG
      //dataset
      uart_printf("%d \t%lld \t%lld \t%d \t%lld\n", i, data_double[i].x, data_double[i].y, data_double[i].label, d);
#endif

    }

    
    //classify test point

    //clear all votes
    int votes[C] = {0};
    int best_votation = 0;
    int best_voted = 0;

    //make neighbours vote
    for (int j=0; j<K; j++) { //for all neighbors
      if ( (++votes[data_double[neighbor_double[j].idx].label]) > best_votation ) {
        best_voted = data_double[neighbor_double[j].idx].label;
        best_votation = votes[best_voted];
      }
    }

    x_double[k].label = best_voted;

    votes_acc[best_voted]++;
    
#ifdef DEBUG
    uart_printf("\n\nNEIGHBORS of x[%d]=(%lld, %lld):\n", k, x_double[k].x, x_double[k].y);
    uart_printf("K \tIdx \tX \tY \tDist \t\tLabel\n");
    for (int j=0; j<K; j++)
      uart_printf("%d \t%lld \t%lld \t%lld \t%lld \t%d\n", j+1, neighbor_double[j].idx, data_double[neighbor_double[j].idx].x,  data_double[neighbor_double[j].idx].y, neighbor_double[j].dist,  data_double[neighbor_double[j].idx].label);
    
    uart_printf("\n\nCLASSIFICATION of x[%d]:\n", k);
    uart_printf("X \tY \tLabel\n");
    uart_printf("%d \t%d \t%d\n\n\n", x_double[k].x, x_double[k].y, x_double[k].label);

#endif

  } //all test points classified

  //stop knn here
  //read current timer count, compute elapsed time
  elapsedu = timer_time_us(TIMER_BASE);
  uart_printf("\nExecution time: %dus @%dMHz\n\n", elapsedu, FREQ/1000000);

  
  //print classification distribution to check for statistical bias
  for (int l=0; l<C; l++)
    uart_printf("%d ", votes_acc[l]);
  uart_printf("\n");
  
}

