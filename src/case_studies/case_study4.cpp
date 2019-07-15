///
///

#include "case_studies.h"

#include <taskExec.h>
#include <SHS.h>

int case_study4(
    const std::string &model_path,
    const std::string &input_sig_path)
{
  // Define time horizon for simulation
  int T1 = 32;

  // Number of monte carlo simulations
  int monte = 5000;

  // Define initial distributions
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());

  arma::mat Xinit = join_vert(arma::ones<arma::mat>(1, monte),
                              15 * arma::ones<arma::mat>(1, monte));
  std::normal_distribution<double> d1{450, 25};
  std::normal_distribution<double> d2{17, 2};
  for (size_t i = 0; i < monte; ++i) 
  {
    Xinit(0, i) = d1(generator);
    Xinit(1, i) = d2(generator);
  }

  std::vector<arma::mat> X = {Xinit};
  std::vector<arma::mat> q_init = {arma::zeros<arma::mat>(T1 + 1, monte)};

  // Read input signal from file
  arma::mat U = readInputSignal(input_sig_path.c_str());
  exdata_t data(X, U, q_init);

  // Get model from file
  shs_t<arma::mat, int> cs4SHS(model_path.c_str(), data);

  // Specify task to be performed
  taskSpec_t cs4Spec(1, T1, monte);

  // Combine model and associated task
  inputSpec_t<arma::mat, int> cs4Input(cs4SHS, cs4Spec);

  // Perform Task
  performTask(cs4Input);

  return 0;
}
