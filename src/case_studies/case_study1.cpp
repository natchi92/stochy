///
///

#include "case_studies.h"

#include <taskExec.h>
#include <SHS.h>

int case_study1(const std::string &model_path)
{
  // Get model from file
  shs_t<arma::mat, int> cs1SHS(model_path.c_str());

  // Specify verification task
  std::vector<int> grid_size;
  int num = -1;
  do 
  {
    std::cout << "Please input grid sizes of interest [value, value, ..], "
                 "enter -1 to generate results for all grids in Table 1 of "
                 "paper or 0 to stop: ";
    std::cin >> num;
    if (num != 0) {
      grid_size.push_back(num);
    }
  } while (num > 0);

  if (grid_size.empty()) 
  {
    std::cout << "No dimension provided." << std::endl;
    exit(0);
  }

  // Get number of grid to generate
  // For FAUST set the error value
  // For IMDP set the gridsize
  std::vector<double> E_FAUST;
  std::vector<double> GS_IMDP;
  if (num == -1) {
    std::vector<double> F = {2.5, 1.8, 1.25, 1.02, 0.90};
    E_FAUST = F;
    std::vector<double> G = {0.25, 0.1818, 0.125, 0.1017, 0.09231};
    GS_IMDP = G;
    std::vector<int> states = {576, 1089, 2304, 3481, 4225};
    grid_size = states;
  } else {
    for (unsigned i = 0; i < grid_size.size(); i++) {
      switch (grid_size[i]) {
      case 576: {
        E_FAUST.push_back(2.5);
        GS_IMDP.push_back(0.25);
        break;
      }
      case 1089: {
        E_FAUST.push_back(1.8);
        GS_IMDP.push_back(0.1818);
        break;
      }
      case 2304: {
        E_FAUST.push_back(1.25);
        GS_IMDP.push_back(0.125);
        break;
      }
      case 3481: {
        E_FAUST.push_back(1.02);
        GS_IMDP.push_back(0.1017);
        break;
      }
      case 4225: {
        E_FAUST.push_back(.90);
        GS_IMDP.push_back(0.09231);
        break;
      }
      default: {
        std::cout << "Invalid grid size" << std::endl;
        int E_F = 0, G_I = 0;
        std::cout << " Input FAUST max error required: ";
        std::cin >> E_F;
        E_FAUST.push_back(E_F);
        std::cout << std::endl;
        std::cout << " Input IMDP grid size : ";
        std::cin >> G_I;
        if (G_I <= 0) {
          std::cout << "Invalid grid size. Grid size cannot be equal to 0 or "
                       "negative"
                    << std::endl;
          exit(0);
        }
        GS_IMDP.push_back(G_I);
      }
      }
    }
  }

  for (unsigned g = 0; g < grid_size.size(); g++) {
    // get number of states
    int num_states = grid_size[g];
    std::cout << " Formal verification via FAUST^2 with grid of "
              << num_states << " states" << std::endl;
    // Define max error
    double eps = E_FAUST[g];

    // Define safe set
    arma::mat Safe = {{405, 540}, {18, 24}};

    // Define grid type
    int gridType = 2; // Grid: 1 = uniform, 2 = adaptive

    // Time horizon
    int K = 3;
    faust_t inputFAUST;
    taskSpec_t cs1SpecFAUST(2, K, 1, Safe, eps, gridType);

    // Combine model and associated task
    inputSpec_t<arma::mat, int> cs1InputFAUST(cs1SHS, cs1SpecFAUST);

    // Perform  Task
    performTask(cs1InputFAUST);

    // Perform  Task
    std::cout << " Done with with FAUST^2 with" << num_states << " states "
              << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << " Formal verification via IMDP with grid of " << num_states
              << " states" << std::endl;
    //  Define safe set
    arma::mat bound = {{18, 24}, {18, 24}};
    // Define grid size for 576 states
    arma::mat grid1 = {{GS_IMDP[g], GS_IMDP[g]}};

    // Relative tolerance
    arma::mat rtol1 = {{1, 1}};

    // Specify task to be performed
    taskSpec_t cs1Spec(3, K, 1, bound, grid1, rtol1);

    // Combine model and associated task
    cs1SHS.x_mod[0].F = arma::eye(2, 2);
    inputSpec_t<arma::mat, int> cs1Input(cs1SHS, cs1Spec);

    // Perform Task
    performTask(cs1Input);

    std::cout << "------------Completed Case Study 1 : Results in results "
                 "folder -----------"
              << std::endl;
  }
  return 0;
}
