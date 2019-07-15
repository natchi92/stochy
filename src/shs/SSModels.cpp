//

#include "SSModels.h"
#include "matio.h" /*Reading .arma::mat files*/
#include <armadillo>

// Initialise an empty state space model
ssmodels_t::ssmodels_t() {
  x_dim = -1;
  u_dim = -1;
  d_dim = -1;
  delta_t = -1;
  A = arma::zeros<arma::mat>(1, 1);
  B = arma::zeros<arma::mat>(1, 1);
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::zeros<arma::mat>(1, 1);
  Q = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  sigma = arma::zeros<arma::mat>(1, 1);
  N.reset();F.reset(); B.reset(); C.reset(); Q.reset();
  A.reset(); sigma.reset();

}

// Initialise an empty state space model
// with correct dimensions
ssmodels_t::ssmodels_t(int x_d, int u_d, int d_d) {
  x_dim = x_d;
  u_dim = u_d;
  d_dim = d_d;
  delta_t = -1;
  A = arma::zeros<arma::mat>(1, 1);
  B = arma::zeros<arma::mat>(1, 1);
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::zeros<arma::mat>(1, 1);
  Q = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  N.reset();F.reset(); B.reset(); C.reset(); Q.reset();
  sigma = arma::zeros<arma::mat>(1, 1);
}

// Initialise state space model based on user
// defined inputs
ssmodels_t::ssmodels_t(int dt, arma::mat Am, arma::mat Sm) {
  x_dim = Am.n_rows;
  u_dim = 0;
  d_dim = 0;
  delta_t = dt;
  A = Am;
  B = arma::zeros<arma::mat>(1, 1);
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::zeros<arma::mat>(1, 1);
  Q = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  N.reset();F.reset(); B.reset(); C.reset(); Q.reset();
  sigma = Sm;
}

ssmodels_t::ssmodels_t(arma::mat Am, arma::mat Qm, arma::mat Sm) {
  x_dim = Am.n_rows;
  u_dim = 0;
  d_dim = 0;
  delta_t = 1;
  A = Am;
  B = arma::zeros<arma::mat>(1, 1);
  B.reset();
  C = arma::zeros<arma::mat>(1, 1);
  C.reset();
  F = arma::zeros<arma::mat>(1, 1);
  F.reset();
  Q = Qm;
  N = arma::zeros<arma::mat>(1, 1);
  N.reset();
  sigma = Sm;
}


// Initialise state space model based on user
// defined inputs
ssmodels_t::ssmodels_t(arma::mat Am, arma::mat Sm) {
  x_dim = Am.n_rows;
  u_dim = 0;
  d_dim = 0;
  delta_t = 1;
  A = Am;
  B = arma::zeros<arma::mat>(1, 1);
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::eye<arma::mat>(x_dim, x_dim);
  Q = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  N.reset(); B.reset(); C.reset(); Q.reset();
  sigma = Sm;
}
// Initialise state space model based on user
// defined inputs (for BMDP model checking)
ssmodels_t::ssmodels_t(int dt, arma::mat Am, arma::mat Fm, arma::mat Sm,
                       int sig) {
  x_dim = Am.n_rows;
  u_dim = 0;
  d_dim = 0;
  delta_t = dt;
  A = Am;
  if (sig) {
    F = Fm;
  } else {
    B = Fm;
  }
  C = arma::zeros<arma::mat>(1, 1);
  Q = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  sigma = Sm;
  N.reset(); C.reset(); Q.reset();
}

ssmodels_t::ssmodels_t( double d_t, arma::mat Am, arma::mat Bm,  arma::mat Fm,
                      arma::mat Qm, arma::mat Sm){
  x_dim = Am.n_rows;
  u_dim = Bm.n_rows;
  d_dim = Fm.n_rows;
  delta_t = d_t;
  A = Am;
  B = Bm;
  F = Fm;
  Q = Qm;
  C = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  sigma = Sm;
  N.reset(); C.reset();
}

// Initialise state space model based on user
// defined inputs (for BMDP model checking)
ssmodels_t::ssmodels_t(arma::mat Am, arma::mat Bm, arma::mat Qm, arma::mat Sm) {
  x_dim = Am.n_rows;
  u_dim = Bm.n_rows;
  d_dim = 0;
  delta_t = 15;
  A = Am;
  B = Bm;
  Q = Qm;
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  sigma = Sm;
  N.reset(); C.reset(); F.reset();
}
// Initialise state space model based on user
// defined inputs
ssmodels_t::ssmodels_t(int dt, arma::mat Am, arma::mat Bm, arma::mat Sm) {
  x_dim = Am.n_rows;
  u_dim = Bm.n_cols;
  d_dim = 0;
  delta_t = dt;
  A = Am;
  B = Bm;
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::zeros<arma::mat>(1, 1);
  Q = arma::zeros<arma::mat>(1, 1);
  N = arma::zeros<arma::mat>(1, 1);
  N.reset(); C.reset(); F.reset(); Q.reset();
  sigma = Sm;
}
// Initialise state space model based on user
// defined inputs
ssmodels_t::ssmodels_t(int dt, arma::mat Am, arma::mat Bm, arma::mat Nm,
                       arma::mat Sm) {
  x_dim = Am.n_rows;
  u_dim = Bm.n_cols;
  d_dim = 0;
  delta_t = dt;
  A = Am;
  B = Bm;
  C = arma::zeros<arma::mat>(1, 1);
  F = arma::zeros<arma::mat>(1, 1);
  Q = arma::zeros<arma::mat>(1, 1);
  N = Nm; C.reset(); F.reset(); Q.reset();
  sigma = Sm;
}

// Initialise state space model based on user
// defined inputs
ssmodels_t::ssmodels_t(int x_d, int u_d, int d_d, double d_t, arma::mat Am,
                       arma::mat Bm, arma::mat Cm, arma::mat Fm, arma::mat Qm,
                       arma::mat Sm) {
  x_dim = x_d;
  u_dim = u_d;
  d_dim = d_d;
  delta_t = d_t;
  A = Am;
  B = Bm;
  C = Cm;
  F = Fm;
  Q = Qm;
  N = arma::zeros<arma::mat>(1, 1);
  N.reset();
  sigma = Sm;
}

ssmodels_t::ssmodels_t(arma::mat Am, arma::mat Bm, arma::mat Nm, arma::mat Qm, arma::mat Sm) {

  x_dim = Am.n_cols;
  u_dim = Bm.n_cols;
  d_dim = 0;
  A = Am;
  B = Bm;
  C = arma::zeros<arma::mat>(x_dim,1);
  F = arma::zeros<arma::mat>(x_dim,d_dim);
  Q = Qm;
  N = Nm;
  sigma = Sm;
  C.reset(); F.reset();
}

ssmodels_t::ssmodels_t(double d_t, arma::mat Am, arma::mat Bm, arma::mat Cm,
                       arma::mat Fm, arma::mat Qm, arma::mat Sm) {
  x_dim = Am.n_cols;
  u_dim = Bm.n_cols;
  d_dim = Fm.n_cols;
  delta_t = d_t;
  A = Am;
  B = Bm;
  C = Cm;
  F = Fm;
  Q = Qm;
  N = arma::zeros<arma::mat>(1, 1);
  N.reset();
  sigma = Sm;
}

// Initialise state space model based on data
// from input .arma::mat file
//
void ssmodels_t::obtainSSfromMat(const char *fn, ssmodels_t &init) {
  // Reading model file input in .arma::mat format
  // and storing into ssmodel class
  try {
    mat_t *matf;
    matvar_t *matvar, *contents;
    // Read .arma::mat file
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      while ((matvar = Mat_VarReadNextInfo(matf)) != NULL) {
        contents = Mat_VarRead(matf, matvar->name);
        init.populate(init, *contents);
        Mat_VarFree(matvar);
        matvar = NULL;
        contents = NULL;
      }
    } else // unsuccesful in opening file
    {
      throw "Error opening mat file";
    }
    Mat_Close(matf);
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

// Initialise state space model based on data
// from input .arma::mat file and current mode
void ssmodels_t::obtainSSfromMat(const char *fn, ssmodels_t &init,
                                 int curMode) {
  // Reading model file input in .arma::mat format
  // and storing into ssmodel class
  mat_t *matf;
  matvar_t *matvar, *contents;
  try {
    // Read .arma::mat file
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      matvar = Mat_VarReadNextInfo(matf);
      while (matvar != NULL) {
        contents = Mat_VarRead(matf, matvar->name);
        if (contents != NULL) {
          init.populate(init, *contents, curMode);
        }
        Mat_VarFree(matvar);
        matvar = NULL;
        contents = NULL;
        matvar = Mat_VarReadNextInfo(matf);
      }
    } else // unsuccessful in opening file
    {
      throw "Error opening arma::mat file";
    }
    Mat_Close(matf);
    init.x_dim = init.A.n_cols;
    // init.u_dim = init.B.n_n_cols();
    // init.d_dim = init.F.n_n_cols();
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

void ssmodels_t::obtainBMDPfromMat(const char *fn, ssmodels_t &init,
                                   int curMode) {
  // Reading model file input in .arma::mat format
  // and storing into ssmodel class

  mat_t *matf;
  matvar_t *matvar, *contents;
  try {
    // Read .arma::mat file
    matf = Mat_Open(fn, MAT_ACC_RDONLY);
    if (matf) // if successful in reading file
    {
      // read each variable within file and populate
      // state space model based on variable name
      matvar = Mat_VarReadNextInfo(matf);
      while (matvar != NULL) {
        contents = Mat_VarRead(matf, matvar->name);
        if (contents != NULL) {
          init.populateBMDP(init, *contents, curMode);
        }
        Mat_VarFree(matvar);
        matvar = NULL;
        contents = NULL;
        matvar = Mat_VarReadNextInfo(matf);
      }
    } else // unsuccessful in opening file
    {
      throw "Error opening .mat file";
    }
    Mat_Close(matf);
    init.x_dim = init.A.n_cols;
    // init.u_dim = init.B.n_n_cols();
    // init.d_dim = init.F.n_n_cols();
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

/// Update ssmodel based on input data read from arma::mat file
// 1. populate state space based on variable name
//  2. Check if dimensions match
//
void ssmodels_t::populate(ssmodels_t &init, matvar_t &content) {
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity

  if (strcmp(content.name, "A") == 0) {
    init.A = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "B") == 0) {
    init.B = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "C") == 0) {
    init.C = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "F") == 0) {
    init.F = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "Q") == 0) {
    init.Q = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "N") == 0) {
    init.N = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "sigma") == 0) {
    init.sigma = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "dim") == 0) {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double xd = strtod(str, NULL);
    init.x_dim = (int)xd;
  } else if (strcmp(content.name, "ddim") == 0) {
    char str[1];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double dd = strtod(str, NULL);
    init.d_dim = (int)dd;
  } else if (strcmp(content.name, "udim") == 0) {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double ud = strtod(str, NULL);
    init.u_dim = (int)ud;
  } else if (strcmp(content.name, "dt") == 0) {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double dt = strtod(str, NULL);
    init.delta_t = dt;
  }
}

// Update ssmodel based on input data read from arma::mat file
// 0. Variable name = "Variable + mode"
// 1. populate state space based on variable name
// 2. Check if dimensions match
//
void ssmodels_t::populate(ssmodels_t &init, matvar_t &content, int curMode) {
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity
  std::string A("A"), B("B"), C("C"), F("F"), Q("Q"), Sigma("sigma"), N("N");
  std::string mode = std::to_string(curMode);
  A += mode;
  B += mode;
  C += mode;
  F += mode;
  Q += mode;
  Sigma += mode;
  N += mode;
  const char *cA = A.c_str(), *cB = B.c_str(), *cC = C.c_str();
  const char *cF = F.c_str(), *cQ = Q.c_str(), *cSigma = Sigma.c_str();
  const char *cN = N.c_str();

  if (strcmp(content.name, cA) == 0) {
    init.A = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cB) == 0) {
    init.B = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cC) == 0) {
    init.C = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cF) == 0) {
    init.F = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cQ) == 0) {
    init.Q = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cSigma) == 0) {
    init.sigma = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cN) == 0) {
    init.N = init.fillMatrix(init, content);
  } else if (strcmp(content.name, "dim") == 0) {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double xd = strtod(str, NULL);
    init.x_dim = (int)xd;
  } else if (strcmp(content.name, "ddim") == 0) {
    char str[1];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double dd = strtod(str, NULL);
    init.d_dim = (int)dd;
  } else if (strcmp(content.name, "udim") == 0) {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double ud = strtod(str, NULL);
    init.u_dim = (int)ud;
  } else if (strcmp(content.name, "dt") == 0) {
    char str[10];
    char *data = (char *)content.data;
    sprintf(str, "%g", *(double *)data);
    double dt = strtod(str, NULL);
    init.delta_t = dt;
  }
}

void ssmodels_t::populateBMDP(ssmodels_t &init, matvar_t &content,
                              int curMode) {
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity
  std::string A("A"), B("B"), F("G"), Sigma("sigma"), N("N");
  std::string mode = std::to_string(curMode);
  A += mode;
  B += mode;
  F += mode;
  Sigma += mode;
  N += mode;
  const char *cA = A.c_str(), *cB = B.c_str(), *cN = N.c_str();
  const char *cF = F.c_str(), *cSigma = Sigma.c_str();

  if (strcmp(content.name, cA) == 0) {
    init.A = init.fillMatrix(init, content);
    // std::cout << init.A <<std::endl;
  } else if (strcmp(content.name, cB) == 0) {
    init.B = init.fillMatrix(init, content);
    // std::cout << init.B <<std::endl;
  } else if (strcmp(content.name, cF) == 0) {
    init.F = init.fillMatrix(init, content);
    std::cout << init.F << std::endl;
  } else if (strcmp(content.name, cSigma) == 0) {
    init.sigma = init.fillMatrix(init, content);
  } else if (strcmp(content.name, cN) == 0) {
    init.N = init.fillMatrix(init, content);
  }
}

/// Fill state space matrices from corresponding variable
//  in .arma::mat file
//  matio reads matrices in the form of 2D arrays
// /
arma::mat ssmodels_t::fillMatrix(ssmodels_t &init, matvar_t &content) {
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

/// Read contents from .arma::mat file of type cell
// This is used for Tq when in hybrid mode
//
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
        throw "Error reading 'Tq' variable information";
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

int ssmodels_t::checkDimFields(int dim, arma::mat mt) {
  // std::cout << "size" << mt.n_n_cols() << std::endl;
  if (dim == -1 || (unsigned)dim != (unsigned)mt.n_cols) {
    dim = mt.n_cols;
  }

  return dim;
}

void ssmodels_t::checkModel(ssmodels_t &init) {
  try {
    if (init.u_dim != 0 && init.B.size() == 1 && init.u_dim != 1) {
      init.u_dim = 0;
    }
    init.d_dim = checkDimFields(init.d_dim, init.F);
    if (init.d_dim != 0 && init.F.size() == 1 && init.x_dim != 1) {
      init.d_dim = 0;
    }
    if (checkMatrices(init.A.size(), init.x_dim, init.x_dim) == 1) {
      throw "Incorrect dimensions of A-matrix";
    }
    if (checkMatrices(init.B.size(), init.x_dim, init.u_dim) == 1 &&
        init.u_dim != 0) {
      if (init.u_dim == 0 || init.u_dim == -1) {
        init.B.reset();
      } else {
        throw "Incorrect dimensions of B-matrix";
      }
    }
    double nr = init.C.n_cols;
    double nc = init.C.n_rows;
    double a = arma::accu((init.C == arma::zeros<arma::mat>(nr, nc)));
    if ((checkMatrices(nr, nc, 1) == 1) && (a == 0)) {
      throw "Incorrect dimensions of C-matrix";
    }
    if (checkMatrices(init.F.size(), init.x_dim, init.d_dim) == 1) {
      if (init.d_dim == 0 || init.d_dim == -1) {
        init.F.reset();// = arma::zeros<arma::mat>(x_dim, 1);
      } else {
        throw "Incorrect dimensions of F-matrix";
      }
    }
    if (checkMatrices(init.Q.size(), init.x_dim, 1) == 1) {
      if (init.Q.size() == 1 && init.x_dim != 1) {
        init.Q = arma::zeros<arma::mat>(x_dim, 1);
      }
    }
    if (checkMatrices(init.sigma.size(), init.x_dim, init.x_dim) == 1) {
      if (init.sigma.size() == 1 && init.x_dim != 1) {
        init.sigma = arma::zeros<arma::mat>(x_dim, 1);
      } else {
        throw "Incorrect dimensions of Sigma-matrix";
      }
    }
    std::cout << "Correctly compiled state space model" << std::endl;
    //		init.printmodel(init);
  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

int ssmodels_t::checkMatrices(int size, int rows, int n_cols) {
  int error = 0;
  if (size != rows * n_cols) {
    error = 1;
  }
  return error;
}

void ssmodels_t::printmodel(ssmodels_t &init) {
  std::cout << "------------------------------------" << std::endl;
  std::cout << "Dimensions:   " << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "# Continuous variables: " << init.x_dim << std::endl;
  std::cout << "# Inputs: " << init.u_dim << std::endl;
  std::cout << "# Disturbances: " << init.d_dim << std::endl;
  std::cout << "------------------------------------" << std::endl;
  std::cout << "State space model: " << std::endl;
  std::cout << "delta_t: " << init.delta_t << std::endl;
  std::cout << "sigma : " << init.sigma << std::endl;
  if (init.delta_t == 0) {
    double lhs = arma::accu(init.sigma ==
                            arma::zeros<arma::mat>(init.x_dim, init.x_dim));
    if (lhs > 0) {
      std::cout << "dot(x(t)) = Ax(t)+Bu(t)+ Dd(t) + F" << std::endl;
      std::cout << "------------------------------------" << std::endl;
    } else {
      std::cout << "dot(x(t)) = (Ax(t)+Bu(t)+ Qd(t) + F)dt + sigmadW"
                << std::endl;
      std::cout << "------------------------------------" << std::endl;
    }
  } else {
    double lhs = arma::accu(init.sigma ==
                            arma::zeros<arma::mat>(init.x_dim, init.x_dim));
    if (lhs > 0) {
      std::cout << "x[k+1] = Ax[k]+Bu[k]+ Dd[k] + F" << std::endl;
      std::cout << "Sampling time: " << init.delta_t << std::endl;
      std::cout << "------------------------------------" << std::endl;
    } else {
      if (init.N.is_empty()) {
        std::cout << "x[k+1] = Ax[k]+Bu[k]+ Fd[k] + F + sigmaW[k]" << std::endl;
        std::cout << "Sampling time: " << init.delta_t << std::endl;
        std::cout << "------------------------------------" << std::endl;
      } else {
        std::cout << "x[k+1] = Ax[k]+Bu[k]+ N(u[k] (x) x[k])  + F + sigmaW[k]"
                  << std::endl;
        std::cout << "Sampling time: " << init.delta_t << std::endl;
        std::cout << "------------------------------------" << std::endl;
      }
    }
  }

  std::cout << "A: " << std::endl;
  std::cout << init.A << std::endl;
  std::cout << "B: " << std::endl;
  std::cout << init.B << std::endl;
  std::cout << "C: " << std::endl;
  std::cout << init.C << std::endl;
  std::cout << "N :" << std::endl;
  std::cout << init.N << std::endl;
  std::cout << "F: " << std::endl;
  std::cout << init.F << std::endl;
  std::cout << "Q: " << std::endl;
  std::cout << init.Q << std::endl;
  std::cout << "Sigma: " << std::endl;
  std::cout << init.sigma << std::endl;
}

ssmodels_t::~ssmodels_t() {}

arma::mat ssmodels_t::updateLTI(ssmodels_t &old, arma::mat x_k, arma::mat u_k) {
  arma::mat X = arma::zeros<arma::mat>((double)x_k.n_rows, 1);
  if (old.B.is_empty() && old.Q.is_empty()) {
    X = old.A * x_k;
  }
  else if(old.B.is_empty() && !old.Q.is_empty()){
    X = old.A*x_k + old.Q;
  }
  else {
    X = old.A * x_k + old.B * u_k + old.Q;
  }
  // Update x_k
  return X;
}

arma::mat ssmodels_t::updateLTI(arma::mat A, arma::mat B, arma::mat Q,
                                arma::mat x_k, arma::mat u_k) {
  arma::mat X(1, 1);

  if (B.is_empty() && Q.is_empty()) {
    X = A * x_k;
  }
  else if(B.is_empty() && !Q.is_empty()){
    X = A*x_k + Q;
  }
  else {
    X = A * x_k + B * u_k + Q;
  }
  // Update x_k
  return X;
}

arma::mat ssmodels_t::updateLTI(arma::mat A, arma::mat Q, arma::mat x_k) {
  arma::mat X(1, 1);
  X = A * x_k + Q;

  return X;
}

arma::mat ssmodels_t::updateLTIad(ssmodels_t &old, arma::mat x_k, arma::mat u_k,
                                  arma::mat d_k) {
  arma::mat X(old.x_dim, 1);
  if (old.B.is_empty() && old.Q.is_empty()) {
    X = old.A * x_k + old.F * d_k;
  }
  else if(old.B.is_empty() && !old.Q.is_empty()){
    X = old.A*x_k  + old.F * d_k + old.Q;
  }
  else {
    X = old.A * x_k + old.B * u_k + old.F * d_k + old.Q;
  }
  // Update x_k
  return X;
}

arma::mat ssmodels_t::updateLTIad(arma::mat A, arma::mat B, arma::mat F,
                                  arma::mat Q, arma::mat x_k, arma::mat u_k,
                                  arma::mat d_k) {
  arma::mat X(1, 1);
  if (B.is_empty() && Q.is_empty()) {
    X = A * x_k + F * d_k;
  }
  else if(B.is_empty() && !Q.is_empty()){
    X = A*x_k  + F * d_k + Q;
  }
   else {
    X = A * x_k + B * u_k + F * d_k + Q;
  }
  // Update x_k
  return X;
}

arma::mat ssmodels_t::updateLTIst(ssmodels_t &old, arma::mat x_k, arma::mat u_k,
                                  arma::mat d_k) {
  size_t monte = x_k.n_cols;
  arma::mat X(x_k.n_rows, monte);
  arma::mat Q = old.Q;

  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<double> norm(0, 1);
  double nr = old.sigma.n_cols;
  arma::mat dW = arma::zeros<arma::mat>(nr, monte);
  for(size_t j = 0; j < monte; j++){
    for (size_t i= 0; i < old.sigma.n_cols; i++) {
      dW(i,j) = sqrt(1) * norm(generator); // Weiner increment
    }
  }

  if(!old.Q.is_empty()){
    Q = arma::repmat(old.Q, 1,monte);
  }
  if (u_k.is_empty() && d_k.is_empty() && old.Q.is_empty() ) {
    X = old.A * x_k + old.sigma * dW;
  } else if(u_k.is_empty()  && !d_k.is_empty() && old.Q.is_empty()) {
    X = old.A * x_k + old.F * d_k + old.sigma * dW;
  }
  else if(u_k.is_empty()  && d_k.is_empty() && !old.Q.is_empty()) {
    X = old.A * x_k + old.B * u_k + old.F * d_k + old.sigma * dW;
  }
  else if(u_k.is_empty()  && !d_k.is_empty() && !old.Q.is_empty()) {
    X = old.A * x_k + old.Q + old.sigma * dW;
  }
  else if(!u_k.is_empty()  && !d_k.is_empty() && !old.Q.is_empty()) {
    if(old.F.n_rows == d_k.n_cols){
      d_k = d_k.t();
    }
    if(old.B.n_rows == u_k.n_cols){
      u_k = u_k.t();
    }
    X = old.A * x_k + old.B * u_k + old.F * d_k + Q + old.sigma * dW;
  }
  else if(!u_k.is_empty()  && d_k.is_empty() && !old.Q.is_empty()) {
    X = old.A * x_k + old.B * u_k + Q + old.sigma * dW;
  }
  else if(!u_k.is_empty()  && d_k.is_empty() && old.Q.is_empty()) {
    X = old.A * x_k + old.B * u_k + old.sigma * dW;
  }
  else {
    X = old.A * x_k + old.B * u_k + old.F * d_k + old.sigma * dW;
  }
  return X;
}
arma::mat ssmodels_t::updateBi(arma::mat A, arma::mat B, arma::mat N,
                               arma::mat Q, arma::mat x_k, arma::mat u_k) {
  arma::mat X(x_k.n_cols, 1);
  arma::mat temp = arma::ones<arma::mat>(1, x_k.n_cols);

  if(Q.is_empty()) {
    X = A * x_k + B * u_k * temp + N * kron(u_k.col(0), x_k);
  }
  else{
    X = A * x_k + B * u_k * temp + N * kron(u_k.col(0), x_k) + Q * temp;

  }

  return X;
}
