///
///

#include "case_studies.h"

#include <taskExec.h>
#include <SHS.h>

int case_study3()
{
  // Specify number of dimensions of interest
  std::vector<int> dim;
  int num = -1;
  do {
    std::cout << "Please input dimension of models of interest  [value, "
                 "value , ..], enter -1 to generate results for all "
                 "dimensions in Table 2 of paper or 0 to stop: ";
    std::cin >> num;
    if (num != 0) {
      dim.push_back(num);
    }
  } while (num > 0);

  if (dim.empty()) {
    std::cout << "No dimension provided." << std::endl;
    exit(0);
  }
  // Get dimensions to run
  std::vector<int> dimensions_2Run;
  if (num == -1) {
    std::vector<int> F = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    dimensions_2Run = F;
  } else {
    dimensions_2Run = dim;
  }

  // ------------------------- Case study 3 - Scaling in dimensions
  // -----------------------
  for (unsigned d = 0; d < dimensions_2Run.size(); d++) {
    std::cout
        << "-----------------------------------------------------------------"
        << std::endl;
    std::cout << std::endl;
    // Get current dimension
    int crnt_dim = dimensions_2Run[d];
    std::cout << "Running verification with " << crnt_dim
              << " continuous variables." << std::endl;

    // Define the boundary for each dimesnion
    arma::mat bound(1, 2);
    bound(0, 0) = -1;
    bound(0, 1) = 1;

    // Define grid size for each dimension
    arma::mat grid = arma::ones<arma::mat>(1, 1);

    // Define relative tolerance for each dimension
    arma::mat reft = arma::ones<arma::mat>(1, 1);

    // Concatenate matrices to correct dimension
    for (unsigned con = 0; con < crnt_dim - 1; con++) {
      arma::mat tempBound(1, 2);
      tempBound(0, 0) = -1;
      tempBound(0, 1) = 1;
      bound = join_vert(bound, tempBound);
      grid = join_horiz(grid, arma::ones<arma::mat>(1, 1));
      reft = join_horiz(reft, arma::ones<arma::mat>(1, 1));
    }
    // Define time horizon
    int T = -1;

    // Select task to perform
    // 1 = verification, 2 = synthe:sis for propertySpec
    taskSpec_t cs3Spec(3, T, 1, bound, grid, reft);

    // set appropriate dynamics
    arma::mat Am3 = 0.8 * arma::eye(crnt_dim, crnt_dim);
    arma::mat sigmam3 = 0.2 * arma::eye(crnt_dim, crnt_dim);

    ssmodels_t model03(Am3, sigmam3);

    std::vector<ssmodels_t> models4 = {model03};
    shs_t<arma::mat, int> cs3SHS(crnt_dim, models4);

    inputSpec_t<arma::mat, int> cs3Input(cs3SHS, cs3Spec);

    performTask(cs3Input);
    std::cout
        << "-----------------------------------------------------------------"
        << std::endl;
    std::cout << "Completed verfication task." << std::endl;
    std::cout
        << "-----------------------------------------------------------------"
        << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
