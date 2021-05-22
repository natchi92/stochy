//

#include "SSModels.h"
//#include "matio.h" /*Reading mat files*/
#include <armadillo>
#include <string>

ssmodels_t::ssmodels_t(std::optional<arma::mat> _A, std::optional<arma::mat> _B,
                       std::optional<arma::mat> _C, std::optional<arma::mat> _F,
                       std::optional<arma::mat> _N, std::optional<arma::mat> _G,
                       std::optional<arma::mat> _Sigma, double _delta_t)
    : A(_A), B(_B), C(_C), F(_F), N(_N), G(_G), Sigma(_Sigma),
      delta_t(_delta_t) {
  // Initialising
  std::cout << "Here" << *A << std::endl;
};

ssmodels_t::ssmodels_t(arma::mat _A, arma::mat _G)
    : ssmodels_t(std::make_optional(_A), std::nullopt, std::nullopt,
                 std::nullopt, std::nullopt, std::make_optional(_G),
                 std::nullopt, 1.0){};

ssmodels_t::ssmodels_t(arma::mat _A, arma::mat _B, arma::mat _Sigma)
    : ssmodels_t(std::make_optional(_A), std::make_optional(_B), std::nullopt,
                 std::nullopt, std::nullopt, std::nullopt,
                 std::make_optional(_Sigma), 1.0){};

ssmodels_t::ssmodels_t(arma::mat _A, arma::mat _B, arma::mat _N,
                       arma::mat _Sigma)
    : ssmodels_t(std::make_optional(_A), std::make_optional(_B), std::nullopt,
                 std::nullopt, std::make_optional(_N), std::nullopt,
                 std::make_optional(_Sigma), 1.0){};

ssmodels_t::ssmodels_t(arma::mat _A, arma::mat _B, arma::mat _G, arma::mat _F,
                       arma::mat _Sigma)
    : ssmodels_t(std::make_optional(_A), std::make_optional(_B), std::nullopt,
                 std::make_optional(_F), std::nullopt, std::make_optional(_G),
                 std::make_optional(_Sigma), 1.0){};

void ssmodels_t::checkModelStructure() {
  // most basic model consists of having only the A Matrix
  try {
    if (A.has_value()) {

      if (A.value().is_square()) {
        int x_dim = A.value().n_rows;

        // Is the model stochastic ?
        if (G.has_value()) {
          if (!(G.value().is_square()) || (G.value().n_rows != x_dim)) {
            throw " Incorrectly specified sigma matrix";
          }
        }

        // do we have a control input?
        if (B.has_value()) {
          if (!(B.value().is_vec()) || (B.value().n_rows != x_dim)) {
            throw " Incorrectly specified B matrix";
          }
        }

        // do we have a control input?
        if (F.has_value()) {
          if (!(F.value().is_vec()) || (F.value().n_rows != x_dim)) {
            throw " Incorrectly specified F matrix";
          }
        }

        // TODO(ncauchi) check-dimensions and shapes of rest of matrices (C, N,
        // sigma)

        printf(" Correctly compiled state-space model\n");
        printModelStructure();
      }
    } else {
      throw " The simplest model is of form x[k+1] = Ax[k], A matrix needs to "
            "be specified";
    }
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

void ssmodels_t::printModelStructure() {
  std::cout << " ------------------------------------ " << std::endl;
  std::cout << " Dimensions:   " << std::endl;
  std::cout << " ------------------------------------ " << std::endl;
  int x_dim = A.has_value() ? A.value().n_rows : 0;
  int u_dim = B.has_value() ? B.value().n_rows : 0;
  int g_dim = G.has_value() ? G.value().n_rows : 0;

  std::cout << " # Continuous variables: " << x_dim << std::endl;
  std::cout << " # Inputs: " << u_dim << std::endl;
  std::cout << " # Disturbances: " << g_dim << std::endl;
  std::cout << " ------------------------------------ " << std::endl;
  std::cout << " ------------------------------------ " << std::endl;
  std::cout << " State space model: " << std::endl;
  std::cout << " Sampling time: " << delta_t << std::endl;

  if (delta_t) {
    // TODO(ncauchi) check definition for IMDP version
    std::cout << " x[k+1] = Ax[k] + Bu[k] + N(u[k] (x) x[k]) + G g[k] + F + "
                 "Sigma W[k] \n"
              << std::endl;

  } else {
    std::cout << " dot(x(t)) = (Ax(t)+Bu(t)+ Gg(t) + F) dt + Sigma dW \n "
              << std::endl;
  }

  if (x_dim) {
    std::cout << " A : \n" << A.value() << std::endl;
  }

  if (u_dim) {
    std::cout << " B : \n" << B.value() << std::endl;
  }

  if (N.has_value()) {
    std::cout << " N : \n" << N.value() << std::endl;
  }

  if (g_dim) {
    std::cout << " G : \n" << G.value() << std::endl;
  }

  if (F.has_value()) {
    std::cout << " F : \n" << F.value() << std::endl;
  }

  if (Sigma.has_value()) {
    std::cout << " Sigma : \n" << Sigma.value() << std::endl;
  }
  std::cout << " ------------------------------------ " << std::endl;
}

// Initialise state space model based on data
void ssmodels_t::obtainSSfromMat(const char *fn,
                                 std::optional<int> currentMode) {
  // Reading model file input in MATLAB mat format and storing into ssmodel
  // class
  mat_t *matf;
  matvar_t *matvar, *contents;
  try {
    // Read mat file
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      matvar = Mat_VarReadNextInfo(matf);
      while (matvar != NULL) {
        contents = Mat_VarRead(matf, matvar->name);
        if (contents != NULL) {

          populateStateSpaceModelFromMatFile(
              *contents, currentMode.has_value() ? currentMode.value() : 0);
        }
        Mat_VarFree(matvar);
        matvar = NULL;
        contents = NULL;
        matvar = Mat_VarReadNextInfo(matf);
      }
    } else // unsuccessful in opening file
    {
      throw " ERROR: Could not open MAT file";
    }
    Mat_Close(matf);
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

// Update ssmodel based on input data read from arma::mat file
// 0. Variable name = "Variable + mode"
// 1. populate state space based on variable name
// 2. Check if dimensions match
// Note: This is generic for both abstraction types
void ssmodels_t::populateStateSpaceModelFromMatFile(matvar_t &content,
                                                    int currentMode) {
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity
  // TODO(ncauchi) change Q to G in mat files
  std::string As("A"), Bs("B"), Cs("C"), Fs("F"), Gs("Q"), Sigmas("Sigma"),
      Ns("N"), dt("dt");
  std::string mode = std::to_string(currentMode);

  if (currentMode > 0) {
    std::string mode = std::to_string(currentMode);
    As += mode;
    Bs += mode;
    Cs += mode;
    Fs += mode;
    Gs += mode;
    Sigmas += mode;
    Ns += mode;
  }

  std::string data = (char *)content.data;

  if (As.compare(content.name) == 0) {
    A = std::make_optional(fillMatrix(content));
  } else if (Bs.compare(content.name) == 0) {
    B = std::make_optional(fillMatrix(content));
  } else if (Cs.compare(content.name) == 0) {
    C = std::make_optional(fillMatrix(content));
  } else if (Fs.compare(content.name) == 0) {
    F = std::make_optional(fillMatrix(content));
  } else if (Gs.compare(content.name) == 0) {
    G = std::make_optional(fillMatrix(content));
  } else if (Sigmas.compare(content.name) == 0) {
    Sigma = std::make_optional(fillMatrix(content));
  } else if (Ns.compare(content.name) == 0) {
    N = std::make_optional(fillMatrix(content));
  } else if (dt.compare(content.name) == 0) {
    delta_t = std::stod(data);
  }

  // Note
  std::cout << " Populated state space model for abstraction using mat file"
            << std::endl;
}

// Fill state space matrices from corresponding variable
// in mat file
// Note: matio reads matrices in the form of 2D arrays
arma::mat ssmodels_t::fillMatrix(matvar_t &content) {
  size_t stride = Mat_SizeOf(content.data_type);
  char *data = (char *)content.data;
  arma::mat mt(content.dims[0], content.dims[1]);
  unsigned i, j = 0;
  for (i = 0; i < content.dims[0] && i < 15; i++) {
    for (j = 0; j < content.dims[1] && j < 15; j++) {
      size_t idx = content.dims[0] * j + i;
      char str[64];
      void *t = data + idx * stride;
      sprintf(str, "%g", *(double *)t); // Assumes values are of type double
      double val = strtod(str, NULL);
      mt(i, j) = val;
    }
  }

  return mt;
}

/// Read contents from mat file of type cell
// This is used for Tq when in hybrid mode
std::string ssmodels_t::readCells(matvar_t &content) {
  std::string str;

  try {
    matvar_t *cell;
    unsigned int i, j, k = 0;
    // Get all the cells
    for (i = 0; i < content.dims[0] * content.dims[1]; i++) {
      cell = Mat_VarGetCell(&content, i);
      char *data = (char *)cell->data;
      if (NULL == cell) {
        throw " Error reading 'Tq' variable information";
      }
      for (k = 0; k < cell->dims[0]; k++) {
        for (j = 0; j < cell->dims[1]; j++) {
          char substr[100];
          sprintf(substr, "%c", data[j * cell->dims[0] + k]);
          str.append(substr);
        }
        str.append(" ");
      }
    }
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }

  return str;
}

arma::mat ssmodels_t::getNextStateFromCurrent(arma::mat &x_k,
                                              std::optional<arma::mat> u_k,
                                              std::optional<arma::mat> g_k) {
  arma::mat X = x_k; // If we have no model, return original state-space

  if (delta_t <= 0) {
    // We are in CT so we need to simulate the stochasticity
    // Number of simulations to run
    int monte = x_k.n_cols;

    // Simulate Weiner process
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<double> norm(0, 1);
    arma::mat dW = arma::zeros<arma::mat>(x_k.n_rows, monte);

    for (size_t j = 0; j < monte; j++) {
      for (size_t i = 0; i < x_k.n_rows; i++) {
        dW(i, j) = sqrt(1) * norm(generator); // Weiner increment
      }
    }

    arma::mat sigma = arma::eye(x_k.n_rows, x_k.n_rows);
    if (Sigma.has_value()) {
      sigma = Sigma.value();
    } else {
      std::cout << "Sigma matrix not specified setting to identity matrix "
                << std::endl;
    }

    X += sigma * dW;
  }

  // determine model type
  if (A.has_value()) {
    X = A.value() * x_k;

    if (F.has_value()) {
      X += F.value();
    }
  }

  if (!(u_k.has_value()) && !(g_k.has_value())) {
    return X;
  } else if (u_k.has_value() && !(g_k.has_value())) {
    if (B.has_value()) {
      X += B.value() * u_k.value();

      if (N.has_value()) {
        X += N.value() * kron(u_k.value().col(0), x_k);
      }

      return X;
    } else {
      std::cout
          << " Given an input signal, but have no B matrix, ignoring input "
          << std::endl;
    }
  } else {
    if (G.has_value()) {
      X += G.value() * g_k.value();
      return X;
    } else {
      std::cout << " Given an disturbance signal, but have no G matrix, "
                   "ignoring input"
                << std::endl;
    }
  }

  return X;
}
