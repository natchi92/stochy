///
///

#include "case_studies.h"

#include <taskExec.h>
#include <SHS.h>

int case_study_lisa()
{
  //Set Variables, such as length of time-step, and the rates determining the exponential-distributed time delay
  double tau = 0.01;
  double lambda1 =0.1;
  double lambda2 = 0.08;

  //Specify transition kernel Tq as a 3x3 stochastic matrix
  arma::mat Tq={{1.0-((lambda1+lambda2)*tau),((lambda1+lambda2)*tau)*(lambda1/(lambda1+lambda2)),((lambda1+lambda2)*tau)*(lambda2/(lambda1+lambda2))},
                            {0.0,1.0,0.0},{0.0,0.0,1.0}};

  //Autonomous system -> x_k+1=A*x+F+G*w, where w is the disturbance. As we want to model systems without disturbance, we set G to 0
  //Specify flow in location q0.
  arma::mat Aq0 = arma::colvec({1.0});
  arma::mat Fq0 =arma::colvec({2.0*tau});
  arma::mat Gq0 = arma::colvec({0.0});
  ssmodels_t modelq0(Aq0,Fq0,Gq0);

  //Specify flow in location q1.
  arma::mat Aq1 = arma::colvec({1.0});
  arma::mat Fq1 = arma::colvec({0.0});
  arma::mat Gq1 = arma::colvec({0.0});
  ssmodels_t modelq1(Aq1,Fq1,Gq1);

  //Specify flow in location q2.
  arma::mat Aq2 = arma::colvec({1.0});
  arma::mat Fq2 = arma::colvec({-3.0 * tau});
  arma::mat Gq2 = arma::colvec({0.0});
  ssmodels_t modelq2(Aq2,Fq2,Gq2);

  //Build model
  std::vector<ssmodels_t> models = {modelq0,modelq1,modelq2};
  int x_dim = 1;
  shs_t<arma::mat, int> csSHS(x_dim, models);

  // Define Time horizon
  int T = -1;

  // Define grid boundary
  // [x-coordinates; y coordinates]
  arma::mat boundary = {{-1.5, 1.5}, {-1.5, 1.5}};

  // Define grid size for each dimension
  arma::mat grid = {{0.12, 0.12}};

  // Define tolerance
  arma::mat rtol = {{0.06, 0.06}};

  // Specify task to be performed
  taskSpec_t csSpec(3, T, 3, boundary, grid, rtol);

  // Combine specfication and associated task
  inputSpec_t<arma::mat, int> csInput(csSHS, csSpec);
  performTask(csInput);

  
  return 0;
}
