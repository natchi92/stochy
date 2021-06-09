/*
 * Bmdp.cpp
 *
 *  Created on: 14 Jun 2018
 *      Author: natchi
 */

#include "Bmdp.h"
#include <task_parallelisation/task_manager.h>

#include <armadillo>
#include <cstddef>
#include <ostream>
#include <string>

// Initialisation
bmdp_t::bmdp_t() {
  Strmode a0;
  a0.vertices = {arma::zeros<arma::mat>(1, 1)};
  a0.mode_center = arma::zeros<arma::mat>(1, 1);
  a0.states = arma::zeros<arma::mat>(1, 1);
  mode = {a0};
  arma::cube a(1, 1, 1);
  vertices = a;
  states = arma::ones<arma::mat>(1, 1);
  arma::sp_mat Stepsmin(1, 1);
  arma::sp_mat Stepsmax(1, 1);
  L = arma::ones<arma::mat>(1, 1);
  actNum = -1;
  desc.boundary = {{1, 1, 1, 1}, {-1, -1, -1, -1}};
  desc.cov = {arma::eye<arma::mat>(1, 1)};
  desc.mean = {arma::ones<arma::mat>(1, 1)};
  desc.reftol = {0.01};
  desc.gridsize = {0.5};
  ssmodels_t b;
  desc.dyn.dynamics = {b};
  desc.dyn.mode = {-1};
  Intmode c;
  c.transfermatrix = arma::zeros<arma::mat>(1, 1);
  c.boundary = {{1, 1, 1, 1}, {-1, -1, -1, -1}};
  c.reftol = {0.01};
  c.gridsize = {0.5};
  c.postCov = arma::eye<arma::mat>(1, 1);
  desc.mode = {c};
  eps = -1;
  E_max = 0;
  Solution = arma::zeros<arma::mat>(0,0);
  Policy = arma::zeros<arma::mat>(0,0);
}

bmdp_t::bmdp_t(taskSpec_t &myTask) {
  Strmode a0;
  a0.vertices = {arma::zeros<arma::mat>(1, 1)};
  a0.mode_center = arma::zeros<arma::mat>(1, 1);
  a0.states = arma::zeros<arma::mat>(1, 1);
  mode = {a0};
  arma::cube a(1, 1, 1);
  vertices = a;
  states = arma::ones<arma::mat>(1, 1);
  arma::sp_mat Stepsmin(1, 1);
  arma::sp_mat Stepsmax(1, 1);
  L = arma::ones<arma::mat>(1, 1);
  actNum = -1;
  desc.boundary = myTask.boundary;
  desc.cov = {arma::eye<arma::mat>(1, 1)};
  desc.mean = {arma::ones<arma::mat>(1, 1)};
  desc.reftol = myTask.reftol;
  desc.gridsize = myTask.gridsize;
  ssmodels_t b;
  desc.dyn.dynamics = {b};
  desc.dyn.mode = {-1};
  Intmode c;
  c.transfermatrix = arma::zeros<arma::mat>(1, 1);
  c.boundary = {{1, 1, 1, 1}, {-1, -1, -1, -1}};
  c.reftol = {0.01};
  c.gridsize = {0.5};
  c.postCov = arma::eye<arma::mat>(1, 1);
  desc.mode = {c};
  eps = -1;
  E_max = 0;
  Solution = arma::zeros<arma::mat>(0,0);
  Policy = arma::zeros<arma::mat>(0,0);
}

bmdp_t::bmdp_t(taskSpec_t &myTask, shs_t<arma::mat, int> &inModel) {
  Strmode a0;
  a0.vertices = {arma::zeros<arma::mat>(1, 1)};
  a0.mode_center = arma::zeros<arma::mat>(1, 1);
  a0.states = arma::zeros<arma::mat>(1, 1);
  mode = {a0};
  arma::cube a(1, 1, 1);
  vertices = a;
  states = arma::ones<arma::mat>(1, 1);
  arma::sp_mat Stepsmin(1, 1);
  arma::sp_mat Stepsmax(1, 1);
  L = arma::ones<arma::mat>(1, 1);
  actNum = -1;
  desc.boundary = myTask.boundary;
  desc.cov = {arma::eye<arma::mat>(1, 1)};
  desc.mean = {arma::ones<arma::mat>(1, 1)};
  desc.reftol = myTask.reftol;
  desc.gridsize = myTask.gridsize;
  ssmodels_t b;
  desc.dyn.dynamics = inModel.x_mod;
  desc.dyn.mode = {-1};
  Intmode c;
  c.transfermatrix = arma::zeros<arma::mat>(1, 1);
  c.boundary = {{1, 1, 1, 1}, {-1, -1, -1, -1}};
  c.reftol = {0.01};
  c.gridsize = {0.5};
  c.postCov = arma::eye<arma::mat>(1, 1);
  desc.mode = {c};
  eps = -1;
  E_max = 0;
  Solution = arma::zeros<arma::mat>(0,0);
  Policy = arma::zeros<arma::mat>(0,0);
}

bmdp_t::~bmdp_t() {}

// Create python file to plot grid of IMDP
// TODO: FIX 3D plotting
void bmdp_t::createPyGrid(std::vector<Strmode> ver_valid, arma::mat boundary) {
  unsigned x_dim = boundary.n_rows;
  unsigned ver = ver_valid[0].vertices[0].n_cols;
  std::ofstream myfile;
  myfile.open("gridPlot.py");
  if (x_dim < 3) {
    myfile << "import numpy as np \n";
    myfile << "import matplotlib.pyplot as plt \n";
    myfile << "from matplotlib.ticker import FuncFormatter\n \n";
  } else if (x_dim == 3) {
    myfile << "import numpy as np \n";
    myfile << "import matplotlib.pyplot as plt \n";
    myfile << "from mpl_toolkits.mplot3d import Axes3D\n \n";
  } else {
    std::cout << "Plotting dimensions > 3" << std::endl;
    return;
  }
  for (size_t i = 0; i < x_dim; i++) {
    myfile << "boundary" << i << "=( ";
    for (size_t j = 0; j < ver; j++) {
      if (j == ver - 1) {
        myfile << boundary(i, j); //<< "," << boundary(i, 0);
      } else {
        myfile << boundary(i, j) << ",";
      }
    }
    myfile << ") \n";
  }
  for (unsigned m = 0; m < ver_valid.size(); ++m) {
    if (m > 0) {
      myfile << "plt.figure(" << m << ")" << std::endl;
    }
    if (x_dim < 3) {
      myfile << "fig" << m << ",ax" << m << "= plt.subplots()" << std::endl;
    } else {
      myfile << "fig" << m << ",ax" << m << "= plt.subplots()" << std::endl;
      myfile << "ax" << m << " = Axes3D(fig" << m << ")" << std::endl;

      //			ax = fig.add_subplot(111, projection='3d')
    }
    size_t vvsize = ver_valid[m].vertices.size();
    for (size_t i = 0; i < vvsize; i++) {
      for (size_t j = 0; j < x_dim; j++) {
        myfile << "v" << i << j << "=(";
        for (size_t k = 0; k < ver; k++) {
          if (k == ver - 1) {
            myfile << ver_valid[m].vertices[i](j, k) << ")" << std::endl;
          } else {
            myfile << ver_valid[m].vertices[i](j, k) << ",";
          }
        }
      }
      myfile << "ax" << m << ".fill(v";
      for (unsigned r = 0; r < x_dim; ++r) {
        if (r == x_dim - 1) {
          myfile << i << r << ")" << std::endl;
        } else {
          myfile << i << r << ",v";
        }
      }
      arma::vec meanVv = arma::mean(ver_valid[m].vertices[i], 1);
      myfile << "plt.text(";
      for (unsigned r = 0; r < x_dim; ++r) {
        if (r == x_dim - 1) {
          myfile << meanVv(r) << "," << i + 1
                 << ",horizontalalignment='center',fontsize=10)" << std::endl;
        } else {
          myfile << meanVv(r) << ",";
        }
      }
    }
  }
  myfile << "plt.savefig(\"IMDP_LowerSatProbability.svg\",dpi=150)\n";
  myfile << "plt.show()" << std::endl;

  myfile.close();
}
// Create python file to plot grid of IMDP
// with color mapping of probabilities on top of grid
void bmdp_t::createPyGrid(std::vector<Strmode> ver_valid, arma::mat boundary,std::vector<double> minP) {
  // dimensions
  unsigned x_dim = boundary.n_rows;

  unsigned ver = ver_valid[0].vertices[0].n_cols;
  // Check if folde
  std::ofstream myfile;
  if(checkFolderExists("../results") == -1) {
    if(mkdir("../results", 0777) == -1) {
      std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
      exit(0);
     }
  }
  myfile.open("../results/IMDP_gridPlot.py");
  if (x_dim < 3) {
    myfile << "import numpy as np \n";
    myfile << "import matplotlib.pyplot as plt \n";
    myfile << "import matplotlib.colorbar as cbar\n \n";
  } else if (x_dim == 3) {
    myfile << "import numpy as np \n";
    myfile << "import matplotlib.pyplot as plt \n";
    myfile << "from mpl_toolkits.mplot3d import Axes3D\n \n";
  } else {
    std::cout << "Plotting dimensions > 3" << std::endl;
    return;
  }
  for (size_t i = 0; i < x_dim; i++) {
    myfile << "boundary" << i << "=( ";
    for (size_t j = 0; j < ver; j++) {
      if (j == ver - 1) {
        myfile << boundary(i, j); //<< "," << boundary(i, 0);
      } else {
        myfile << boundary(i, j) << ",";
      }
    }
    myfile << ") \n";
  }
  for (unsigned m = 0; m < ver_valid.size(); ++m) {
    if (m > 0) {
      myfile << "plt.figure(" << m << ")" << std::endl;
    }
    if (x_dim < 3) {
      myfile << "fig" << m << ",ax" << m << "= plt.subplots()" << std::endl;

    } else {
      myfile << "fig" << m << ",ax" << m << "= plt.subplots()" << std::endl;
      myfile << "ax" << m << " = Axes3D(fig" << m << ")" << std::endl;
    }
    // Define colour map
    myfile << "z = np.array([";
    for(unsigned mp=0; mp < minP.size(); mp++){
      if(mp < minP.size()-1) {
        myfile << minP[mp] << ", ";
      }
      else {
        myfile << minP[mp] << "])\n";
      }
    }
    myfile << "normal =plt.Normalize(z.min(), z.max())\n";
    myfile << "cmap = plt.cm.binary(normal(z))\n";
    size_t vvsize = ver_valid[m].vertices.size();

    // Offset is used to obtain vertices associated with
    // each mode from list of vertices
    if(this->vertices.n_cols == 4){
      int offset = 0;
      if(m > 0) {
        for(size_t f =0; f <m; f++){
          offset +=ver_valid[m-1].vertices.size()-1;
        }
      }
      for (size_t i = 0; i < vvsize; i++) {
        for (size_t j = 0; j < x_dim; j++) {
          myfile << "v" << i << j << "=(";
          myfile << this->vertices(j, 0, i+ offset) <<"," << this->vertices(j, 1, i+ offset) << ", "<<this->vertices(j,2 , i + offset) <<"," << this->vertices(j, 3, i+ offset) << ")" << std::endl;
        }
        myfile << "ax" << m << ".fill(v";
        for (unsigned r = 0; r < x_dim; ++r) {
          if (r == x_dim - 1) {
            myfile << i << r << ",facecolor=cmap[" << i+offset<<"])" << std::endl;
          } else {
            myfile << i << r << ",v";
          }
        }
      }

    }
    else {
      for (size_t i = 0; i < vvsize; i++) {
        for (size_t j = 0; j < x_dim; j++) {
          myfile << "v" << i << j << "=(";
          for (size_t k = 0; k < ver; k++) {
            if ((k == ver - 1) && (j == 0)) {
              myfile <<ver_valid[m].vertices[i](j, k)/this->desc.mode[m].transfermatrix(j,j) <<"," <<ver_valid[m].vertices[i](j, k-1)/this->desc.mode[m].transfermatrix(j,j)  << ")" << std::endl;
            } else if((k != ver - 1) && (j == 0)) {
              myfile << ver_valid[m].vertices[i](j, k)/this->desc.mode[m].transfermatrix(j,j)   << "," <<ver_valid[m].vertices[i](j, k+1)/this->desc.mode[m].transfermatrix(j,j)  << ", ";
            }
            else if((k == ver - 1) && (j != 0)) {
              myfile <<ver_valid[m].vertices[i](j, k)/this->desc.mode[m].transfermatrix(j,j) <<"," <<ver_valid[m].vertices[i](j, k)/this->desc.mode[m].transfermatrix(j,j)  << ")" << std::endl;
            } else {
              myfile << ver_valid[m].vertices[i](j, k)/this->desc.mode[m].transfermatrix(j,j)   << "," <<ver_valid[m].vertices[i](j, k)/this->desc.mode[m].transfermatrix(j,j) << ", ";
            }
          }
        }
        myfile << "ax" << m << ".fill(v";
        for (unsigned r = 0; r < x_dim; ++r) {
          if (r == x_dim - 1) {
            myfile << i << r << ",facecolor=cmap[" << i<<"])" << std::endl;
          } else {
            myfile << i << r << ",v";
          }
        }
      }
    }

    myfile << "cax"<<m << ", _ = cbar.make_axes(ax" << m<< ")\n";
    myfile << "cb" << m+1 << " = cbar.ColorbarBase(cax" << m << ", cmap=plt.cm.binary, norm = normal)\n";
  }
  myfile << "plt.savefig(\"IMDP_LowerSatProbability.svg\",dpi=150)\n";
  myfile << "plt.show()" << std::endl;
  myfile.close();
}

void bmdp_t::getHybSysModes() {
  // Define necessary matrices
  arma::mat transfermatrix = arma::zeros<arma::mat>(
      this->desc.dyn.dynamics[0].A.n_rows, this->desc.dyn.dynamics[0].A.n_rows);

  arma::mat postCov_diag = arma::zeros<arma::mat>(
      this->desc.dyn.dynamics[0].A.n_rows, this->desc.dyn.dynamics[0].A.n_rows);
  arma::mat gridsize = arma::zeros<arma::mat>(
      this->desc.dyn.dynamics[0].A.n_rows, this->desc.dyn.dynamics[0].A.n_rows);
  arma::mat reftol = arma::zeros<arma::mat>(
      this->desc.dyn.dynamics[0].A.n_rows, this->desc.dyn.dynamics[0].A.n_rows);
  unsigned dim = this->desc.dyn.dynamics.size();
  for (size_t d = 0; d < dim; ++d) {
    // If sigma is ever non diagonal update boundary
    // as need more info
    if (!isDiagonal(this->desc.dyn.dynamics[d].sigma)) {
      arma::mat boundary = arma::zeros<arma::mat>(2, 4);
      boundary(0, 0) = this->desc.boundary(0, 0);
      boundary(0, 1) = this->desc.boundary(0, 0);
      boundary(0, 2) = this->desc.boundary(0, 1);
      boundary(0, 3) = boundary(0, 2);
      boundary(1, 0) = this->desc.boundary(1, 0);
      boundary(1, 1) = this->desc.boundary(1, 1);
      boundary(1, 2) = boundary(1, 1);
      boundary(1, 3) = boundary(1, 0);
      this->desc.boundary = boundary;
    }
  }
  for (size_t i = 0; i < dim; ++i) {
    // System of form x = Ax + Fw
    // for mode i
    arma::mat A = this->desc.dyn.dynamics[i].A;
    arma::mat F = this->desc.dyn.dynamics[i].F;
    arma::mat Sigma =
        this->desc.dyn.dynamics[i].sigma; // covariance of the noise
    // post of state covariance
    arma::mat postCov = F * Sigma * F.t();
    // check if the post of state covariance is = scalar * Identity
    arma::mat check = arma::round(1e4 * postCov / (postCov(0, 0))) * 1e-4;
    arma::mat id = arma::eye<arma::mat>(arma::size(postCov));
    bool equiv = arma::approx_equal(check, id, "reldiff", 0.0001);
    if (equiv) {
      // transfer function is identity
      transfermatrix = arma::eye(arma::size(A));
      if (i == 0) {
        this->desc.mode[i].transfermatrix = transfermatrix;
      } else {
        this->desc.mode.push_back(this->desc.mode[0]);
        this->desc.mode[i].transfermatrix = transfermatrix;
      }
      this->desc.mode[i].transfermatrix = transfermatrix;
      postCov_diag = postCov;
      gridsize = this->desc.gridsize; // Original whole system grid size
      reftol = this->desc.reftol;     // Original whole system tolerance
    } else {
      // if not, then the transfer function is the matrix of eigen vectors
      arma::vec eigVal;
      arma::mat eigVec;
      arma::eig_sym(eigVal, eigVec, postCov);

      arma::mat eV = arma::conv_to<arma::mat>::from(eigVal);
      arma::mat eVc = arma::conv_to<arma::mat>::from(eigVec);

      arma::mat temp = arma::diagmat(arma::pow(eV, -0.5));


      transfermatrix = temp * eVc.t();
      if (i == 0) {
        this->desc.mode[i].transfermatrix = transfermatrix;
      } else {
        this->desc.mode.push_back(this->desc.mode[0]);
        this->desc.mode[i].transfermatrix = transfermatrix;
      }
      postCov_diag = arma::eye<arma::mat>(this->desc.dyn.dynamics[0].x_dim,
                                          this->desc.dyn.dynamics[0].x_dim);
      gridsize = (temp.diag().t()) % this->desc.gridsize; // Original whole system grid size
      reftol =
          temp.diag().t() % this->desc.reftol; // Original whole system rel tol
    }

    // check if such a transfer matrix already exists
    int mode_num = -1;
    bool mode_exists = 0;
    while ((!mode_exists) && mode_num < ((int)this->desc.mode.size())) {
      mode_num++;
      mode_exists = arma::approx_equal(transfermatrix,
                                       this->desc.mode[mode_num].transfermatrix,
                                       "reldiff", 0.0001);
    }
    if (mode_exists) {
      // Update
      if (transfermatrix.n_elem == 1) {
        this->desc.mode[mode_num].transfermatrix =
            arma::repmat(transfermatrix, 2, 2);
        this->desc.mode[mode_num].boundary =
            this->desc.boundary * (arma::repmat(transfermatrix, 2, 2));

      } else {
        this->desc.mode[mode_num].transfermatrix = transfermatrix;
        this->desc.mode[mode_num].boundary =
            transfermatrix * this->desc.boundary;
      }
      this->desc.mode[mode_num].gridsize = gridsize;
      this->desc.mode[mode_num].reftol = reftol;
    } else {
      // create a new mode
      mode_num++;
      Intmode tempMode;
      tempMode.transfermatrix = transfermatrix;
      if (transfermatrix.n_elem == 1) {
        tempMode.boundary =
            transfermatrix * arma::ones<arma::mat>(1, 2) * this->desc.boundary;

      } else {
        tempMode.boundary = transfermatrix * this->desc.boundary;
      }
      tempMode.gridsize = gridsize;
      tempMode.reftol = reftol;
      this->desc.mode[mode_num] = tempMode;
    }
    // assign the correct diag covariance to the mode
    this->desc.mode[mode_num].postCov = postCov_diag;
    // add check if diagonal or not if not stop
    checkDiagonal(postCov_diag);
    // also record the mode number in dynamics
    this->desc.dyn.mode[i] = mode_num;
  }
}
// Generate the abstraction
// and get equivalent IMDP
void bmdp_t::bmdpAbstraction(int T, int RA) {
  clock_t begin, end;
  double time;
  begin = clock();

  // Get modes
  this->getHybSysModes();
  // discretize the space in each mode (valid states only)
  double num_states = 0;
  size_t total_States, diagonalFlag = 0;
  arma::vec states;
  std::vector<std::vector<arma::mat>> v_phi1, v_phi2;
  // For reachability properties that are not safety
  // need to get labels which are read from files phi1 and phi2
  if (RA) {
    v_phi1 = this->getLabelVertices("../phi1.txt", 2);

    v_phi2 = this->getLabelVertices("../phi2.txt", 2);
  }
  // Now let us start constructing the grid
  // Check if have diagonal or not
  for (size_t i = 0; i < this->desc.mode.size(); ++i) {
    if (!isDiagonal(this->desc.dyn.dynamics[i].sigma)) {
      diagonalFlag = 1;
      if(this->desc.dyn.dynamics[i].sigma.n_rows > 2) {
        std::cout << "Non-diagonal covariance matrices for continuous variables >2 is a work in progress" <<std::endl;
        exit(0);
      }
    }

  }
  for (size_t i = 0; i < this->desc.mode.size(); ++i) {
       // Construct grid over rectangular space
    if (diagonalFlag == 0) {
      states = this->getGrid(this->desc.mode[i].boundary,
                             this->desc.mode[i].gridsize,
                             this->desc.mode[i].reftol, i);
    } else {
      if (RA) {
        states = this->getGridNondiagRA(
        this->desc.mode[i].boundary, this->desc.mode[i].gridsize,
        this->desc.mode[i].reftol, i, v_phi1, v_phi2);
      } else {
        states = this->getGridNondiag(this->desc.mode[i].boundary,
                                      this->desc.mode[i].gridsize,
                                      this->desc.mode[i].reftol, i);
      }
    }
    // Fix the number of states in bmdp
    this->mode[i].states =
        arma::regspace(num_states, 1, num_states + states(states.n_rows - 1));
    num_states += states(states.n_rows - 1);

    // resize number of vertices cube
    this->vertices.resize(
        this->mode[i].vertices[0].n_rows, this->mode[i].vertices[0].n_cols,
        this->mode[i].states(mode[i].states.n_rows - 1, 0) + 1);

    // Save the original coordinates of the vertices in the bmdp
    for (size_t j = 0; j < states.n_rows; ++j) {
      arma::mat temp = arma::solve(this->desc.mode[i].transfermatrix,
                                   this->mode[i].vertices[j]);

      int index = (int)(this->mode[i].states(0, 0) + j);
      this->vertices.slice(index) = temp;
    }
  }
  total_States = num_states;
  std::cout << "num_states" << num_states<<std::endl;
  this->states = arma::regspace(1, total_States);

  end = clock();
  time = (double)(end - begin) / CLOCKS_PER_SEC;
  // compute min & max steps matrices (transition probabilities)
  if (diagonalFlag == 0) {
    if (this->eps > 0) {
      std::cout << "Performing abstraction via adaptive sequential refinement"
                << std::endl;
      this->getStepsBasedonMedian(this->eps, T);
    } else {
      std::cout << "Performing abstraction via uniform grid" << std::endl;
      this->getSteps();
    }

  } else {
    this->getStepsNonDiag();
  }
}

// this function creates modes for each dynamics whose covariance matrix is
// non-diagonal
// If prob is too small it means that points are far from
// qprime hence pmin and pmax = 0
// Check if center of qprime is in q
double bmdp_t::checkPmin(double pmin, arma::mat qpost_TF, arma::mat qprime_TF,size_t dim) {
  arma::mat vd;
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  // Set initial values on border
  std::vector<double> x1(dim);
  for (size_t i = 0; i < qprime_TF.n_rows; i++) {
    double Xa = qprime_TF(i, 0); // min
    double Xb = qprime_TF(i, 1); // max

    arma::mat temp = {{Xa, Xb}};
    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);
  // If prob is too small it means that points are far from
  // qprime hence pmin and pmax = 0
  double denom = std::sqrt(2)*sigx;
  double outer = 1 / (std::pow(2, dim)), inner = 1;
  arma::vec prob = -2 * arma::ones<arma::vec>(4);
  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned i = 0; i < dim; ++i) {
      double lower = ((vd(i, 1) - qpost_TF(i, j)) / denom);
      double upper = ((vd(i, 0) - qpost_TF(i, j)) / denom);
      inner *= std::erf(lower) - std::erf(upper);
    }
    prob(j) = std::abs(outer * inner);
    inner = 1;
  }

  inner = 1;

  // Check if center of qprime is in q
  arma::mat qprime_ctr = arma::mean(qprime_TF, 1);
  qprime_ctr = join_horiz(qprime_ctr, qprime_ctr);
  if (pnHyperRect(qprime_ctr, qpost_TF)) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned i = 0; i < dim; ++i) {
        double lower = ((vd(i, 1) - qprime_ctr(i, j)) / denom);
        double upper = ((vd(i, 0) - qprime_ctr(i, j)) / denom);
        inner *= std::erf(lower) - std::erf(upper);
      }
      prob(j + 2) = std::abs(outer * inner);
      inner = 1;
    }
  }
  double new_pmin = arma::min(prob);
  if (((pmin == -1) || (pmin > new_pmin))) {
    pmin = new_pmin;
  }

  return pmin;
}

double bmdp_t::checkPmax(double pmax, arma::mat qpost_TF, arma::mat qprime_TF,size_t dim) {
  arma::mat vd;

  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  // Set initial values on border
  std::vector<double> x1(dim);
  for (size_t i = 0; i < qprime_TF.n_rows; i++) {
    double Xa = qprime_TF(i, 0); // min
    double Xb = qprime_TF(i, 1); // max

    arma::mat temp = {{Xa, Xb}};
    //	std::cout << "temp: " << temp << std::endl;
    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);

  // If prob is too small it means that points are far from
  // qprime hence pmin and pmax = 0
  double denom = std::sqrt(2)*sigx;
  double outer = 1 / (std::pow(2, dim)), inner = 1;
  arma::vec prob = 2 * arma::ones<arma::vec>(4);
  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned i = 0; i < dim; ++i) {
      double lower = ((vd(i, 1) - qpost_TF(i, j)) / denom);
      double upper = ((vd(i, 0) - qpost_TF(i, j)) / denom);
      inner *= std::erf(lower) - std::erf(upper);
    }
    prob(j) = std::abs(outer * inner);
    inner = 1;
  }

  inner = 1;
  // Check if center of qprime is in q
  arma::mat qprime_ctr = arma::mean(qprime_TF, 1);
  qprime_ctr = join_horiz(qprime_ctr, qprime_ctr);
  //	std::cout << "qprime_ctr: " << qprime_ctr << std::endl;
  if (pnHyperRect(qprime_ctr, qpost_TF)) {
    for (unsigned j = 0; j < 2; ++j) {
      for (unsigned i = 0; i < dim; ++i) {
        double lower = ((vd(i, 1) - qprime_ctr(i, j)) / denom);
        double upper = ((vd(i, 0) - qprime_ctr(i, j)) / denom);
        inner *= std::erf(lower) - std::erf(upper);
      }
      prob(j + 2) = std::abs(outer * inner);
      //			std::cout << "prob: " << prob << std::endl;
      inner = 1;
    }
  }
  double new_pmax = arma::max(prob);
  if (((pmax == -1) || (pmax < new_pmax))) {
    pmax = new_pmax;
  }
  return pmax;
}

double bmdp_t::getMinTranProb(arma::mat qpost_TF, arma::mat qprime_TF,size_t dim) {
  // Set initial values on border
  arma::mat vd;
  std::vector<double> x1(dim);
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  for (size_t i = 0; i < qprime_TF.n_rows; i++) {
    double Xa = qprime_TF(i, 0); // min
    double Xb = qprime_TF(i, 1); // max
    arma::mat temp = {{Xa, Xb}};

    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);
  // Define upper and lower bounds
  std::vector<double> lb1;
  std::vector<double> ub1;

  for (size_t i = 0; i < dim; i++) {
    double Xa = arma::min(qpost_TF.row(i)); // min
    double Xb = arma::max(qpost_TF.row(i)); // max

    if (any(find(qpost_TF < 0))) {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa);
    } else {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa); // Xa
    }
    if (i % 2 == 0)
      x1[i] = Xa;
    else
      x1[i] = Xb;
  }
  nlopt::opt opt2(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim);
  nlopt::opt opt(nlopt::LN_COBYLA, dim);
  opt2.set_local_optimizer(opt);
  opt2.set_lower_bounds(lb1);
  opt2.set_upper_bounds(ub1);
  opt2.set_maxeval(200);
  // Setup minimiser
  opt2.set_min_objective(myMinOptfuncBMDP, (void *)&vd);
  //	opt.set_xtol_rel(1e-8);
  //	opt.set_ftol_rel(1e-16);
  // Perform optimisation
  double minf2;

  try {
    nlopt::result result2 = opt2.optimize(x1, minf2);
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    exit(0);
  }
  arma::vec res(1);
  res << minf2;
  if (res(0) > 0) {
    res = arma::exp(-res);
  } else {
    res = arma::exp(res);
  }

  return res(0);
}

double bmdp_t::getMinTranProbNonDiag(arma::mat qpost_TF, arma::mat qprime_TF,size_t dim) {
  // Set initial values on border
  arma::mat vd;
  std::vector<double> x1(dim);
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));


  for (size_t i = 0; i < qprime_TF.n_rows; i++) {
    double Xa = arma::min(qprime_TF.row(i));
    double Xb = arma::max(qprime_TF.row(i));
    arma::mat temp = {{Xa, Xb}};

    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);

  // Define upper and lower bounds
  std::vector<double> lb1;
  std::vector<double> ub1;

  for (size_t i = 0; i < dim; i++) {
    double Xa = arma::min(qpost_TF.row(i)); // for precision
    double Xb = arma::max(qpost_TF.row(i));

    if (any(find(qpost_TF < 0))) {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa);
    } else {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa); // Xa
    }
    if (i % 2 == 0)
      x1[i] = Xa;
    else
      x1[i] = Xb;
  }
  nlopt::opt opt2(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim);
  nlopt::opt opt(nlopt::LN_COBYLA, dim);
  opt2.set_local_optimizer(opt);

  opt2.set_lower_bounds(lb1);
  opt2.set_upper_bounds(ub1);
  opt2.set_maxeval(200);
  // Setup minimiser
  opt2.set_min_objective(myMinOptfuncBMDP, (void *)&vd);
  // opt2.set_xtol_rel(1e-8);
  // opt2.set_ftol_rel(1e-16);
  // Perform optimisation
  double minf2;

  try {
    nlopt::result result2 = opt2.optimize(x1, minf2);
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    exit(0);
  }
  arma::vec res(1);
  res << minf2;
  if (res(0) > 0) {
    res = arma::exp(-res);
  } else {
    res = arma::exp(res);
  }

  return res(0);
}

double bmdp_t::getMaxTranProbNonDiag(arma::mat qpost_TF, arma::mat qprime_TF,size_t dim) {

  arma::mat vd;
  std::vector<double> v;
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  // Set initial values on border
  std::vector<double> x1(dim);
  for (size_t i = 0; i < qprime_TF.n_rows; i++) {
    double Xa = arma::min(qprime_TF.row(i));
    double Xb = arma::max(qprime_TF.row(i));

    arma::mat temp = {{Xa, Xb}};
    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
    v.push_back(Xa);
    v.push_back(Xb);
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);

  // Define upper and lower bounds
  std::vector<double> lb1;
  std::vector<double> ub1;

  for (size_t i = 0; i < dim; i++) {
    double Xa = arma::min(qpost_TF.row(i)); // for precision
    double Xb = arma::max(qpost_TF.row(i));

    if (any(find(qpost_TF < 0))) {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa);
    } else {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa); // Xa
    }
    if (i % 2 == 0)
      x1[i] = Xb;
    else
      x1[i] = Xa;
  }

  nlopt::opt opt2(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim);
  nlopt::opt opt(nlopt::LN_COBYLA, dim);
  opt2.set_local_optimizer(opt);

  opt2.set_lower_bounds(lb1);
  opt2.set_upper_bounds(ub1);
  opt2.set_maxeval(200);
  // Setup minimiser
  opt2.set_max_objective(myMinOptfuncBMDP, (void *)&vd);
  // opt2.set_xtol_rel(1e-8);
  // opt2.set_ftol_rel(1e-16);
  // Perform optimisation
  double minf2;
  try {
    nlopt::result result2 = opt2.optimize(x1, minf2);
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    exit(0);
  }
  //	std::cout << minf2 << std::endl;
  arma::vec res(1);
  res << minf2;
  if (res(0) > 0) {
    res = arma::exp(-res);
  } else {
    res = arma::exp(res);
  }

  return res(0);
}

// this function computes the extreme minimum transition probabilities from a
// polygon q to a set of retangular cells qprime, where the noise ditribution is
// Normal distribution with covariance equal to identity matrix times sigma It
// is an underapproximation
double bmdp_t::getMinTranProb2Rect(arma::mat qpost_TF, arma::mat qprime_TF,arma::mat qprime_ctr, size_t dim) {

  // Set initial values on border
  arma::mat vd;
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  std::vector<double> x1(dim);
  size_t size_qp = std::sqrt(qprime_TF.n_rows - 1);
  for (size_t i = 0; i < dim; i++) {
    arma::vec Xa =
        arma::min(qprime_TF.rows(i * size_qp, (i + 1) * size_qp - 1)).t();
    arma::vec Xb =
        arma::max(qprime_TF.rows(i * size_qp, (i + 1) * size_qp - 1)).t();

    arma::mat temp = join_horiz(Xa, Xb);
    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);
  // add size of qprime
  arma::mat qcols = {{(double)size_qp * size_qp + 1, 0}};
  vd = join_vert(vd, qcols);

  // Define upper and lower bounds
  std::vector<double> lb1;
  std::vector<double> ub1;

  for (size_t i = 0; i < dim; i++) {
    double Xa = arma::min(qpost_TF.row(i)); // min
    double Xb = arma::max(qpost_TF.row(i)); // max

    if (any(find(qpost_TF < 0))) {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa);
    } else {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa); // Xa
    }
    if (i % 2 == 0)
      x1[i] = Xa;
    else
      x1[i] = Xb;
  }
  double denom = std::sqrt(2);
  double outer = 1 / (std::pow(2, dim));
  arma::vec inner = arma::ones<arma::vec>(qprime_TF.n_cols);

  // Check if center of qprime is in q
  arma::vec prob = arma::zeros<arma::vec>(qpost_TF.n_cols + 1);
  qprime_ctr =
      join_vert(arma::mean(qprime_TF), qprime_TF.rows(1, qprime_TF.n_rows - 1));
  if (pnHyperRect(qprime_ctr, qpost_TF)) {
    arma::vec inner = arma::ones<arma::vec>(size_qp);
    for (unsigned i = 0; i < dim; ++i) {

      arma::vec lower =
          (vd(arma::span(i * size_qp, (i + 1) * size_qp - 1), 1) -
           qprime_ctr(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0)) /
          denom;

      arma::vec upper =
          (vd(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0) -
           qprime_ctr(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0)) /
          denom;
      inner %= arma::erf(lower) - arma::erf(upper);
    }
    prob(0) = std::abs(outer * arma::sum(inner));

  } else {
    nlopt::opt opt2(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim);
    nlopt::opt opt(nlopt::LN_COBYLA, dim);
    opt2.set_local_optimizer(opt);
    // Setup minimiser
    opt2.set_lower_bounds(lb1);
    opt2.set_upper_bounds(ub1);
    opt2.set_maxeval(2000);

    opt2.set_min_objective(myMinOptfuncRectBMDP, (void *)&vd);
    // opt2.set_xtol_rel(1e-8);
    // opt2.set_ftol_rel(1e-16);
    // Perform optimisation
    double minf2;

    try {
      nlopt::result result2 = opt2.optimize(x1, minf2);
    } catch (std::exception &e) {
      std::cout << "nlopt failed: " << e.what() << std::endl;
      exit(0);
    }
    arma::vec res(1);
    res << minf2;
    if (res(0) > 0) {
      res = arma::exp(-res);
    } else {
      res = arma::exp(res);
    }
    prob(0) = res(0);
  }
  inner = arma::ones<arma::vec>(qprime_TF.n_cols);

  for (size_t k = 0; k < size_qp; k++) {
    for (unsigned i = 0; i < dim; ++i) {
      arma::vec xnew = arma::ones<arma::vec>(qprime_TF.n_cols);
      double qnew = qpost_TF(i, k);
      xnew = xnew * qnew;
      arma::vec lower =
          ((vd(arma::span(i * qprime_TF.n_cols, (i + 1) * qprime_TF.n_cols - 1),
               1) -
            xnew) /
           denom);
      arma::vec upper =
          ((vd(arma::span(i * qprime_TF.n_cols, (i + 1) * qprime_TF.n_cols - 1),
               0) -
            xnew) /
           denom);

      inner %= arma::erf(lower) - arma::erf(upper);
    }
    prob(k + 1) = std::abs(outer * arma::sum(inner));
    inner = arma::ones<arma::vec>(qprime_TF.n_cols);
  }
  return arma::min(prob);
}
// % this function computes the extreme minimum transition probabilities from a
// polygon q to a set of retangular cells qprime, where the noise ditribution is
// Normal distribution with covariance equal to identity matrix times sigma It
// is an underapproximation
double bmdp_t::getMaxTranProb2Rect(arma::mat qpost_TF, arma::mat qprime_TF,arma::mat qprime_ctr, size_t dim) {

  // Set initial values on border
  arma::mat vd;
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  std::vector<double> x1(dim);
  size_t size_qp = std::sqrt(qprime_TF.n_rows - 1);
  for (size_t i = 0; i < dim; i++) {
    arma::vec Xa =
        arma::min(qprime_TF.rows(i * size_qp, (i + 1) * size_qp - 1)).t();
    arma::vec Xb =
        arma::max(qprime_TF.rows(i * size_qp, (i + 1) * size_qp - 1)).t();
    arma::mat temp = join_horiz(Xa, Xb);

    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
  }


  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);

  // add size of qprime
  arma::mat qcols = {{(double)size_qp * size_qp + 1, 0}};
  vd = join_vert(vd, qcols);

  // Define upper and lower bounds
  std::vector<double> lb1;
  std::vector<double> ub1;

  for (size_t i = 0; i < dim; i++) {
    double Xa = arma::min(qpost_TF.row(i)); // for precision
    double Xb = arma::max(qpost_TF.row(i));

    if (any(find(qpost_TF < 0))) {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa);
    } else {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa); // Xa
    }
    if (i % 2 == 0)
      x1[i] = Xa;
    else
      x1[i] = Xb;
  }
  double denom = std::sqrt(2);
  double outer = 1 / (std::pow(2, dim));
  arma::vec inner = arma::ones<arma::vec>(qprime_TF.n_cols);

  // Check if center of qprime is in q
  // TODO: Generalise
  // std::cout << "qprime_ctr: " << qprime_ctr;

  qprime_ctr = arma::ones<arma::vec>(size_qp * size_qp + 1) * qprime_ctr(0, 0);
  qprime_ctr =
      join_vert(qprime_ctr, arma::ones<arma::vec>(size_qp * size_qp + 1) *
                                qprime_ctr(1, 0));
  // std::cout << "qprime_ctr: " << qprime_ctr(0, 0);

  arma::vec prob = arma::zeros<arma::vec>(qpost_TF.n_cols + 1);

  qprime_ctr =
      join_vert(arma::mean(qprime_TF), qprime_TF.rows(1, qprime_TF.n_rows - 1));
  if (pnHyperRect(qprime_ctr, qpost_TF)) {
    arma::vec inner = arma::ones<arma::vec>(size_qp);
    for (unsigned i = 0; i < dim; ++i) {

      arma::vec lower =
          (vd(arma::span(i * size_qp, (i + 1) * size_qp - 1), 1) -
           qprime_ctr(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0)) /
          denom;

      arma::vec upper =
          (vd(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0) -
           qprime_ctr(arma::span(i * size_qp, (i + 1) * size_qp - 1), 0)) /
          denom;
      inner %= arma::erf(lower) - arma::erf(upper);
    }

    prob(0) = std::abs(outer * arma::sum(inner));
    //		std::cout << "prob: " << prob << std::endl;

  } else {
    nlopt::opt opt2(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim);
    nlopt::opt opt(nlopt::LN_COBYLA, dim);
    opt2.set_local_optimizer(opt);

    opt2.set_lower_bounds(lb1);
    opt2.set_upper_bounds(ub1);
    opt2.set_maxeval(2000);
    // Setup minimiser
    opt2.set_max_objective(myMinOptfuncRectBMDP, (void *)&vd);
    // opt2.set_xtol_rel(1e-8);
    // opt2.set_ftol_rel(1e-16);
    // Perform optimisation
    double minf2;

    try {
      nlopt::result result2 = opt2.optimize(x1, minf2);
    } catch (std::exception &e) {
      std::cout << "nlopt failed: " << e.what() << std::endl;
      exit(0);
    }
    arma::vec res(1);
    res << minf2;
    if (res(0) > 0) {
      res = arma::exp(-res);
    } else {
      res = arma::exp(res);
    }
    prob(0) = res(0);
  }
  // std::cout << qpost_TF << std::endl;
  inner = arma::ones<arma::vec>(qprime_TF.n_cols);

  for (size_t k = 0; k < size_qp; k++) {
    for (unsigned i = 0; i < dim; ++i) {
      arma::vec xnew = arma::ones<arma::vec>(qprime_TF.n_cols);
      double qnew = qpost_TF(i, k);
      xnew = xnew * qnew;
      // std::cout << "xnew: " << xnew << std::endl;
      arma::vec lower =
          ((vd(arma::span(i * qprime_TF.n_cols, (i + 1) * qprime_TF.n_cols - 1),
               1) -
            xnew) /
           denom);
      // std::cout << "lower: " << lower << std::endl;
      arma::vec upper =
          ((vd(arma::span(i * qprime_TF.n_cols, (i + 1) * qprime_TF.n_cols - 1),
               0) -
            xnew) /
           denom);

      inner %= arma::erf(lower) - arma::erf(upper);
    }
    prob(k + 1) = std::abs(outer * arma::sum(inner));
    //		std::cout << "prob(k): " << prob(k + 1) << std::endl;
    inner = arma::ones<arma::vec>(qprime_TF.n_cols);
  }
  //	std::cout << "prob: " << prob << std::endl;
  return arma::max(prob);
}
double bmdp_t::getMaxTranProb(arma::mat qpost_TF, arma::mat qprime_TF,size_t dim) {

  arma::mat vd;
  std::vector<double> v;
  arma::mat Qpost = this->desc.mode[0].transfermatrix*this->desc.dyn.dynamics[0].F*this->desc.dyn.dynamics[0].sigma*this->desc.dyn.dynamics[0].F.t()*this->desc.mode[0].transfermatrix.t();
  double sigx = std::sqrt(Qpost(0,0));

  // Set initial values on border
  std::vector<double> x1(dim);
  for (size_t i = 0; i < qprime_TF.n_rows; i++) {
    double Xa = qprime_TF(i, 0); // min
    double Xb = qprime_TF(i, 1); // max

    arma::mat temp = {{Xa, Xb}};
    //	std::cout << "temp: " << temp << std::endl;
    if (i == 0) {
      vd = temp;
    } else {
      vd = join_vert(vd, temp);
    }
    v.push_back(Xa);
    v.push_back(Xb);
  }
  // add dimension at end of matrix
  arma::mat dimen = {{(double)dim, sigx}};
  vd = join_vert(vd, dimen);

  // Define upper and lower bounds
  std::vector<double> lb1;
  std::vector<double> ub1;

  for (size_t i = 0; i < dim; i++) {
    double Xa = arma::min(qpost_TF.row(i)); // min
    double Xb = arma::max(qpost_TF.row(i)); // max

    if (any(find(qpost_TF < 0))) {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa);
    } else {
      ub1.push_back(Xb); // Xb
      lb1.push_back(Xa); // Xa
    }
    if (i % 2 == 0)
      x1[i] = Xb;
    else
      x1[i] = Xa;
  }

  nlopt::opt opt2(nlopt::GN_DIRECT_L_RAND_NOSCAL, dim);
  nlopt::opt opt(nlopt::LN_COBYLA, dim);
  opt2.set_local_optimizer(opt);

  opt2.set_lower_bounds(lb1);
  opt2.set_upper_bounds(ub1);
  opt2.set_maxeval(200);
  // Setup minimiser
  opt2.set_max_objective(myMinOptfuncBMDP, (void *)&vd);
  // opt2.set_xtol_rel(1e-8);
  // opt2.set_ftol_rel(1e-16);
  // Perform optimisation
  double minf2;
  try {
    nlopt::result result2 = opt2.optimize(x1, minf2);
  } catch (std::exception &e) {
    std::cout << "nlopt failed: " << e.what() << std::endl;
    exit(0);
  }
  //	std::cout << minf2 << std::endl;
  arma::vec res(1);
  res << minf2;
  if (res(0) > 0) {
    res = arma::exp(-res);
  } else {
    res = arma::exp(res);
  }

  return res(0);
}
// this function creates modes for each dynamics whose covariance matrix is
// non-diagonal for general type
void bmdp_t::getSteps() {
  // initialize the matrices
  size_t num_dyn = this->mode.size();
  size_t num_states = this->states.n_elem + 1;
  size_t currentrow = 0;
  size_t index = 0, x_dim = this->desc.dyn.dynamics[0].x_dim;
  arma::sp_mat Smin(num_dyn * num_states, num_states);
  arma::sp_mat Smax(num_dyn * num_states, num_states);

  double pminCheck = -1, pmaxCheck = -1;
  for (size_t q = 0; q < this->states.n_rows; ++q) {
    for (size_t a = 0; a < num_dyn; ++a) {

      currentrow = (q)*num_dyn + a;
      size_t HSmode = this->desc.dyn.mode[a];

      // Get boundary in transformed space
      arma::mat A = this->desc.dyn.dynamics[a].A;
      arma::mat F = this->desc.dyn.dynamics[a].F;
      arma::mat Sigma = this->desc.dyn.dynamics[a].sigma;
      arma::mat TransferMatrix, Qpost, qpost, qpost_TF;
      if (A.n_cols == 1) {
        TransferMatrix = this->desc.mode[HSmode].transfermatrix(0, 0);
        Qpost = TransferMatrix * F * Sigma * F.t() * TransferMatrix.t();
        qpost = A * this->vertices(0, 0, q);
        qpost_TF = TransferMatrix * qpost;
      } else {
        TransferMatrix = this->desc.mode[HSmode].transfermatrix;
        Qpost = TransferMatrix * F * Sigma * F.t() * TransferMatrix.t();
        float sigx = std::sqrt(Qpost(0, 0));
        float sigy = std::sqrt(Qpost(1, 1));


        arma::mat Qcheck = arma::round(Qpost * 1e5) * 1e-5;
        bool isQdiag = true;
        if (Qcheck(0, 1) != 0) {
          isQdiag = false;
        }
        bool sigCheck = (std::round(sigx * 1e5) != std::round(sigy * 1e5));
        if (!isQdiag && sigCheck) {
          throw "Covariance Matrices are NOT scalar * identity";
        }
        // post of mode q
        qpost = A * this->vertices.slice(q);
        qpost_TF = TransferMatrix * qpost;

      }

      arma::mat qprime_TF(qpost_TF.n_rows, qpost_TF.n_cols);
      double pmin = 0, pmax = 0;

      // Go through each cell
      for (size_t qprime = 0; qprime < this->mode[HSmode].vertices.size();
           ++qprime) {
        qprime_TF = this->mode[HSmode].vertices[qprime];

        // get min and max probabilities for currrent transitions
        double pmini = checkPmin(pminCheck, qpost_TF, qprime_TF,
                                 this->desc.dyn.dynamics[HSmode].x_dim);


        if (pmini == -2) {
          pmin = this->getMinTranProb(qpost_TF, qprime_TF,
                                      this->desc.dyn.dynamics[HSmode].x_dim);
        } else {
          if (pminCheck == -1) {
            pminCheck = pmini;
            pmin = pmini;
          } else if (pmini > pminCheck) {
            pmin = pminCheck;
          } else {
            pmin = pmini;
            pminCheck = pmini;
          }
        }
        double pmaxi = checkPmax(pmaxCheck, qpost_TF, qprime_TF, x_dim);
        if (pmaxi == 2) {
          pmax = this->getMaxTranProb(qpost_TF, qprime_TF, x_dim);
        } else {
          if (pmaxi < pmaxCheck) {
            pmax = pmaxCheck;
          } else {
            pmax = pmaxi;
            pmaxCheck = pmaxi;
          }
        }
      /*  if (pmin < 1e-7) {
          pmin = 0;
        }
        if (pmax < 1e-7) {
          pmax = 0;
        }*/

        Smin(currentrow, this->mode[HSmode].states(index, 0)) = pmin;
        Smax(currentrow, this->mode[HSmode].states(index, 0)) = pmax;
        index++;
      }
      index = 0;

      // for the out-of-boundary state pmin = 1-pmax
      // for states out of boundary need to compute underapproximation
      qprime_TF = this->desc.mode[0].transfermatrix * this->desc.boundary;

      // Check if  qprime_TF lies outside qprime_TF
      // find the min/max transition prob to the boundary
      double v = 0;
      if (arma::accu(qpost_TF.col(0) < qprime_TF.col(0)) > 0) {
        v = 1;
      }

      if (arma::accu(qpost_TF.col(1) > qprime_TF.col(1)) > 0) {
        v = 1;
      }
     if ((v == 0) || (q > this->mode[HSmode].vertices.size()-1)) {
        pmin = this->getMinTranProb(qpost_TF, qprime_TF,
                                    this->desc.dyn.dynamics[HSmode].x_dim);
        pmax = this->getMaxTranProb(qpost_TF, qprime_TF,
                                    this->desc.dyn.dynamics[HSmode].x_dim);
      } else {
        // if not rectangle, find tight bounds
        arma::mat qprime_set_TF = this->mode[HSmode].vertices[q];
        arma::mat qprime_set_TF_ctr = this->mode[HSmode].mode_center;
        pmin = this->getMinTranProb2Rect(qpost_TF, qprime_set_TF,
                                         qprime_set_TF_ctr,
                                         this->desc.dyn.dynamics[HSmode].x_dim);
        pmax = this->getMaxTranProb2Rect(qpost_TF, qprime_set_TF,
                                         qprime_set_TF_ctr,
                                         this->desc.dyn.dynamics[HSmode].x_dim);
      }
    /*  if (pmin < 1e-7) {
        pmin = 0;
      }
      if (pmax < 1e-7) {
        pmax = 0;
      }*/
      Smin(currentrow, num_states - 1) = 1 - pmax;
      Smax(currentrow, num_states - 1) = 1 - pmin;
    }
  }

  // self transition for the out-of-boundary state
  size_t q = num_states;
  for (size_t a = 0; a < num_dyn; ++a) {
    currentrow = (q - 1) * num_dyn + a;
    Smin(currentrow, num_states - 1) = 1;
    Smax(currentrow, num_states - 1) = 1;
  }

  this->Stepsmax = Smax;
  this->Stepsmin = Smin;

  // ------------------------------------------------
  // sanity check
  // ------------------------------------------------
  arma::mat stepsmin = arma::conv_to<arma::mat>::from(Smin);
  arma::mat stepsmax = arma::conv_to<arma::mat>::from(Smax);
  arma::mat minSum = arma::sum(stepsmin.t());
  arma::mat maxSum = arma::sum(stepsmax.t());

  arma::mat minmaxS =
      arma::conv_to<arma::mat>::from(this->Stepsmin - this->Stepsmax);

  int stepsminsum = arma::accu(minSum > 1.01);
  if (stepsminsum) {
    for (unsigned i = 0; i < minSum.n_cols-num_dyn; i++) {
     unsigned nc =  i;
     if(i > stepsmin.n_cols-1) {
        nc = i - stepsmin.n_cols;
      }
      if (maxSum(i) < 1) {
        unsigned end = stepsmin.n_cols - 1;
        stepsmin(i, nc) = stepsmin(i, nc) +
                         (1 - arma::accu(stepsmin(i, arma::span(0, end - 1))));
      }
    }
    this->Stepsmin = arma::conv_to<arma::sp_mat>::from(stepsmin);
//  }
  }
  int stepsmaxsum = arma::accu(maxSum < 0.999991);

  if (stepsmaxsum) {
    for (unsigned i = 0; i < maxSum.n_cols-num_dyn; i++) {
     unsigned nc =  i;
     if(i > stepsmax.n_cols-1) {
        nc = i - stepsmax.n_cols;
      }
      if (maxSum(i) < 1) {
        unsigned end = stepsmax.n_cols - 1;
        stepsmax(i, nc) = stepsmax(i, nc) +
                         (1 - arma::accu(stepsmax(i, arma::span(0, end - 1))));
      }
    }
    this->Stepsmax = arma::conv_to<arma::sp_mat>::from(stepsmax);
  }

  int stepsminmaxsum = arma::accu(arma::sum(arma::sum(minmaxS > 0)));

  if (stepsminmaxsum) {
    std::cout << "Error; Lower bound prob > upper bound prob";
    exit(0);
  }
}
// this function creates modes for each dynamics whose covariance matrix is
// non-diagonal for general type
// assuming performing safety verification
void bmdp_t::getStepsBasedonMedian(double epsilon, int T) {
  // initialize the matrices
  size_t x_dim = this->desc.dyn.dynamics[0].x_dim;

  if (x_dim > 20) {
    for (unsigned q = 0; q < this->mode.size(); q++) {

      arma::mat all_v(this->mode[q].vertices.size(), 2 * x_dim);
      unsigned c_idx = 0;
      for (unsigned i = 0; i < this->mode[q].vertices.size(); i++) {
        for (unsigned j = 0; j < x_dim; j++) {
          for (unsigned k = 0; k < 2; k++) {
            all_v(i, c_idx) = this->mode[q].vertices[i](j, k);
            c_idx++;
          }
        }
        c_idx = 0;
      }

      arma::mat new_v = unique_rows(all_v);
      all_v.reset();

      c_idx = 0;
      // Push new vertices
      this->mode[q].vertices.clear();

      arma::mat tempV(x_dim, 2);
      for (unsigned i = 0; i < new_v.n_rows; i++) {
        for (unsigned j = 0; j < x_dim; j++) {
          for (unsigned k = 0; k < 2; k++) {
            tempV(j, k) = new_v(i, c_idx);
            c_idx++;
          }
        }
        this->mode[q].vertices.push_back(tempV);
        c_idx = 0;
      }
      this->states = arma::regspace(0, 1, this->mode[q].vertices.size() - 1);
    }
  }
  this->getSteps();

  arma::uvec phi1 = arma::ones<arma::uvec>(this->Stepsmax.n_cols);
  phi1(phi1.n_rows - 1, 0) = 0;

  arma::uvec labels = arma::zeros<arma::uvec>(this->Stepsmax.n_cols);
  labels(labels.n_rows - 1) = 1;
  this->createSynthFile(phi1, labels);
  arma::vec E_med = this->getESafety(1e-4, T);

  // Individual cell matrix intialiser
  arma::mat ind_cell(x_dim, 2), new_cell(x_dim, 2);

  arma::uvec E_ij_idx = arma::find(E_med >= epsilon);
  arma::uvec En_ij_idx = arma::find(E_med < epsilon);

  // While Error >= epsilon refine according to largest dimension
  while (arma::max(E_med) >= epsilon && !E_ij_idx.is_empty()) {
    // Keep only errors greater then epsilon
    arma::umat E_idx = arma::ind2sub(arma::size(E_med), E_ij_idx);
    arma::umat En_idx = arma::ind2sub(arma::size(E_med), En_ij_idx);
    arma::uvec idx = arma::unique(E_idx.row(0).t());
    arma::uvec n_idx = arma::unique(En_idx.row(0).t());
    std::vector<arma::mat> tv;
    if (!idx.is_empty()) {
      unsigned offset = 0, q = -1;
      unsigned original_nStates = 0;
      std::vector<arma::mat> original_states;
      // get original states
      for (unsigned k = 0; k < this->mode.size(); k++) {
        original_states.push_back(this->mode[k].states);
        for (unsigned j = 0; j < this->mode[k].vertices.size(); j++) {
          tv.push_back(this->mode[k].vertices[k]);
        }
      }
      for (unsigned i = 0; i < idx.n_rows; ++i) {
        // Identify which grid and then which cell index belongs to
        for (unsigned k = 0; k < this->mode.size(); k++) {
          if (arma::accu(arma::any(original_states[k] == idx(i))) > 0) {
            if (k != q) {
              q = k;
              offset = 0;
              // Previous number of states
              original_nStates = this->mode[q].states.n_rows - 1;
            }
            if (idx(i) < q * original_nStates) {
              idx(i) = q * original_nStates - idx(i);
            } else {
              idx(i) = idx(i) - q * original_nStates;
            }
          }
        }
        // Identified grid
        // Now I need to select cell
        ind_cell = this->mode[q].vertices[idx(i)];

        // Refine hyper rectangle
        arma::mat refined = refineHyperRectangleLargestDiam(ind_cell);

        // Get combinations of new cells
        std::vector<arma::mat> ref_cells;
        rec(ref_cells, refined, refined.cols(0, 1), 0, refined.n_cols,
            refined.n_rows, 0);
        for (unsigned c = 0; c < ref_cells.size(); c++) {
          new_cell = ref_cells[c];
          // Check if new cell already exists
          auto iter = tv.begin() + idx(i) + offset;
          arma::mat a = *iter;
          // if new_cell is within current current cell
          if (c == 0) {
            tv[idx(i) + offset] = new_cell;
          } else {
            // keep current cell and append to it
            // unsigned n_idx = c-1;
            if (idx(i) + offset > tv.size()) {
              tv.push_back(new_cell);

            } else {

              iter = tv.insert(iter, new_cell);
            }
            offset++;
          }
        }
      }
      // Push back the cells which where not split
      if (!n_idx.is_empty()) {
        for (unsigned i = 0; i < n_idx.n_rows - 1; i++) {
          arma::mat adj_cell =
              this->mode[q].vertices[n_idx(i)]; // + arma::repmat(addition,1,2);
          tv.push_back(adj_cell);
        }
      }

      // Clean vertices
      arma::mat all_v(tv.size(), 2 * x_dim);
      unsigned c_idx = 0;
      for (unsigned i = 0; i < tv.size(); i++) {
        for (unsigned j = 0; j < x_dim; j++) {
          for (unsigned k = 0; k < 2; k++) {
            all_v(i, c_idx) = tv[i](j, k);
            c_idx++;
          }
        }
        c_idx = 0;
      }

      arma::mat new_v = unique_rows(all_v);
      all_v.reset();

      c_idx = 0;
      // Push new vertices
      this->mode[q].vertices.clear();

      arma::mat tempV(x_dim, 2);
      for (unsigned i = 0; i < new_v.n_rows; i++) {
        for (unsigned j = 0; j < x_dim; j++) {
          for (unsigned k = 0; k < 2; k++) {
            tempV(j, k) = new_v(i, c_idx);
            c_idx++;
          }
        }
        this->mode[q].vertices.push_back(tempV);
        c_idx = 0;
      }
      if (this->mode[0].vertices[0].n_rows == 2) {
        createPyGrid(this->mode, this->desc.mode[0].boundary);
      }
      // Recompute Steps by only updating corresponding column
      // Need to update number of states
      // and vertices of boundary
      int num_states = 0;

      for (unsigned k = 0; k < this->mode.size(); k++) {
        this->mode[k].states = arma::regspace(
            num_states, 1, this->mode[k].vertices.size() + num_states - 1);
        num_states += this->mode[k].states.n_rows - 1;
      }
      this->states = arma::regspace(0, 1, num_states - 1);
      int index = 0;
      // resize number of vertices cube
      this->vertices.resize(this->mode[0].vertices[0].n_rows,
                            this->mode[0].vertices[0].n_cols, num_states + 1);
      for (unsigned k = 0; k < this->mode.size(); k++) {
        // Save the original coordinates of the vertices in the bmdp
        for (unsigned j = 0; j < mode[k].vertices.size(); ++j) {
          arma::mat temp = arma::solve(this->desc.mode[k].transfermatrix,
                                       this->mode[k].vertices[j]);
          this->vertices.slice(index) = temp;
          index++;
        }
      }
      this->getSteps();
      offset = 0;
      q = -1;
      original_nStates = 0;

      // Get new error
      phi1 = arma::ones<arma::uvec>(this->Stepsmax.n_cols);
      phi1(phi1.n_rows - 1, 0) = 0;
      arma::uvec labels = arma::zeros<arma::uvec>(this->Stepsmax.n_cols);
      labels(labels.n_rows - 1) = 1;
      this->createSynthFile(phi1, labels);
      E_med = this->getESafety(1e-4, T);
      E_ij_idx = arma::find(E_med >= epsilon);
      En_ij_idx = arma::find(E_med < epsilon);
    }
  }

  // ------------------------------------------------
  // sanity check
  // ------------------------------------------------
  arma::mat stepsmin = arma::conv_to<arma::mat>::from(this->Stepsmin);
  arma::mat stepsmax = arma::conv_to<arma::mat>::from(this->Stepsmax);
  arma::mat minSum = arma::sum(stepsmin.t());
  arma::mat maxSum = arma::sum(stepsmax.t());
  arma::mat minmaxS =
      arma::conv_to<arma::mat>::from(this->Stepsmin - this->Stepsmax);
  int stepsminsum = arma::accu(minSum > 1);
  if (stepsminsum) {
    for (unsigned i = 0; i < minSum.n_cols; i++) {
      if (minSum(i) > 1) {
        double delta = (minSum(i) - 1) / (stepsmin.n_cols);
        for (unsigned t = 0; t < stepsmin.n_rows; t++) {
          double new_v = stepsmin(t, i) - delta;
          if (new_v < 0) {
            new_v = 0;
          }
          stepsmax(t, i) = new_v;
        }
      }
    }
    this->Stepsmin = arma::conv_to<arma::sp_mat>::from(stepsmin);
  }

  int stepsmaxsum = arma::accu(maxSum < 1);
  if (stepsmaxsum) {
    for (unsigned i = 0; i < minSum.n_cols; i++) {
      if (maxSum(i) < 1) {
        unsigned end = stepsmax.n_cols - 1;
        stepsmax(i, i) = stepsmax(i, i) +
                         (1 - arma::accu(stepsmax(i, arma::span(0, end - 1))));
      }
    }
    this->Stepsmax = arma::conv_to<arma::sp_mat>::from(stepsmax);
  }
  int stepsminmaxsum = arma::accu(arma::sum(arma::sum(minmaxS > 0)));
  if (stepsminmaxsum) {
    throw "Error; Lower bound prob > upper bound prob";
  }
}
// this function creates modes for each dynamics whose covariance matrix is
// non-diagonal for general type

/* 
 * Computes the transition probabilities starting from a given grid cell.
 *
 * @param q the cell for which to compute the transition probabilities
 * 
 * @note it is threadsafe - uses steps_mutex
 */
void bmdp_t::getStepsNonDiag(int q) {
  size_t num_dyn = this->mode.size();
  size_t num_states = this->states.n_elem + 1;
  size_t ogg_index = 0;

  for (size_t a = 0; a < num_dyn; ++a) {

      int currentrow = (q)*num_dyn + a;
      size_t HSmode = a;

      arma::mat A = this->desc.dyn.dynamics[a].A;
      arma::mat F = this->desc.dyn.dynamics[a].F;
      arma::mat Sigma = this->desc.dyn.dynamics[a].sigma;
      arma::mat TransferMatrix = this->desc.mode[HSmode].transfermatrix;
      arma::mat Qpost = TransferMatrix * F * Sigma * F.t() * TransferMatrix.t();

      float sigx = std::sqrt(Qpost(0, 0));
      float sigy = std::sqrt(Qpost(1, 1));

      arma::mat Qcheck = arma::round(Qpost * 1e5) * 1e-5;
      bool isQdiag = true;
      if (Qcheck(0, 1) != 0) {
        isQdiag = false;
      }
      bool sigCheck = (std::round(sigx * 1e5) != std::round(sigy * 1e5));
      if (!isQdiag && sigCheck) {
        throw "Covariance Matrices are NOT scalar * identity";
      }

      // post of mode q
      arma::mat qpost = A * this->vertices.slice(q);
      arma::mat qpost_TF = TransferMatrix * qpost;
      arma::mat qprime_TF(qpost_TF.n_rows, qpost_TF.n_cols);
      double pmin = 0, pmax = 0;

      int index = 0;
      for (size_t qprime = 0; qprime < this->mode[HSmode].states.n_elem - 1;
           ++qprime) {

        qprime_TF = TransferMatrix *
                    this->vertices.slice(this->mode[HSmode].states(index, 0));
        // get min and max probabilities for currrent transitions
        pmin = this->getMinTranProbNonDiag(
            qpost_TF, qprime_TF, this->desc.dyn.dynamics[HSmode].x_dim);
        pmax = this->getMaxTranProbNonDiag(
            qpost_TF, qprime_TF, this->desc.dyn.dynamics[HSmode].x_dim);

        steps_mutex.lock();
        this -> Stepsmin(currentrow, this->mode[HSmode].states(index, 0)) = pmin;
        this -> Stepsmax(currentrow, this->mode[HSmode].states(index, 0)) = pmax;
        steps_mutex.unlock();

        index++;
      }
      // for the out-of-boundary state pmin = 1-pmax
      qprime_TF = TransferMatrix * this->desc.boundary;

      // find the min/max transition prob to the boundary
      // first, check if boundary is rectangle
      arma::vec v21 = qprime_TF.col(0) - qprime_TF.col(1);
      arma::vec v23 = qprime_TF.col(2) - qprime_TF.col(1);
      arma::vec v43 = qprime_TF.col(2) - qprime_TF.col(3);
      arma::vec v41 = qprime_TF.col(0) - qprime_TF.col(3);

      double v21v23 = arma::dot(v21, v23) * 1e6;
      double v43v41 = arma::dot(v43, v41) * 1e6;
      if ((std::round(v21v23) == 0) && (std::round(v43v41) == 0)) {
        pmin = this->getMinTranProbNonDiag(
            qpost_TF, qprime_TF, this->desc.dyn.dynamics[HSmode].x_dim);
        pmax = this->getMaxTranProbNonDiag(
            qpost_TF, qprime_TF, this->desc.dyn.dynamics[HSmode].x_dim);
      } else {
        // if not rectangle, find tight bounds
        unsigned v_mode = this->mode[HSmode].vertices.size();
        if (q > v_mode - 1) {
          HSmode = 1;
          ogg_index++;
          if (ogg_index >= num_states - 1) {
            ogg_index = 0;
          }
        }
        arma::mat qprime_set_TF = this->mode[HSmode].vertices[ogg_index];
        arma::mat qprime_set_TF_ctr = this->mode[HSmode].mode_center;

        pmin = this->getMinTranProb2Rect(qpost_TF, qprime_set_TF,
                                         qprime_set_TF_ctr,
                                         this->desc.dyn.dynamics[HSmode].x_dim);
        pmax = this->getMaxTranProb2Rect(qpost_TF, qprime_set_TF,
                                         qprime_set_TF_ctr,
                                         this->desc.dyn.dynamics[HSmode].x_dim);
      }

      steps_mutex.lock();
      this -> Stepsmin(currentrow, num_states - 1) = 1 - pmax;
      this -> Stepsmax(currentrow, num_states - 1) = 1 - pmin;
      steps_mutex.unlock();
    }
}

void bmdp_t::getStepsNonDiag() {
  // initialize the matrices
  clock_t begin, end;
  double time;
  begin = clock();
  size_t num_dyn = this->mode.size();
  size_t num_states = this->states.n_elem + 1;
  
  this -> Stepsmin = arma::sp_mat(num_dyn * num_states, num_states);
  this -> Stepsmax = arma::sp_mat(num_dyn * num_states, num_states);

  std::vector<int> tasks;
  for (size_t q = 0; q < num_states - 1; ++q) 
    tasks.push_back(q);

  task_manager<int, void> manager(tasks, 
          [&](int q) { getStepsNonDiag(q); });
  manager.run();

  // self transition for the out-of-boundary state
  size_t q = num_states;
  for (size_t a = 0; a < num_dyn; ++a) {
    int currentrow = (q - 1) * num_dyn + a;
    Stepsmin(currentrow, num_states - 1) = 1;
    Stepsmax(currentrow, num_states - 1) = 1;
  }

  end = clock();
  time = (double)(end - begin) / CLOCKS_PER_SEC;
  // ------------------------------------------------
  // sanity check
  // ------------------------------------------------
  arma::mat stepsmin = arma::conv_to<arma::mat>::from(this -> Stepsmin);
  arma::mat stepsmax = arma::conv_to<arma::mat>::from(this -> Stepsmax);

  arma::mat minSum = arma::sum(stepsmin.t());
  arma::mat maxSum = arma::sum(stepsmax.t());
  arma::mat minmaxS =
      arma::conv_to<arma::mat>::from(this->Stepsmin - this->Stepsmax);
  int stepsminsum = arma::accu(minSum > 1);
  if (stepsminsum) {
    throw "Error; The sum of probabilities in stepsmin exceed 1";
  }
  int stepsmaxsum = arma::accu(maxSum < 1);
  if (stepsmaxsum) {
    for (unsigned i = 0; i < maxSum.n_cols - 1; i++) {
      if (maxSum(i) < 1) {
        if (i > stepsmax.n_cols) {
          stepsmax(i, stepsmax.n_rows - 1 - i) =
              stepsmax(i, stepsmax.n_rows - 1 - i) +
              (1 - arma::accu(stepsmax(i, arma::span(0, stepsmax.n_cols - 2))));
        } else {
          stepsmax(i, i) =
              stepsmax(i, i) +
              (1 - arma::accu(stepsmax(i, arma::span(0, stepsmax.n_cols - 2))));
        }
      }
    }
    this->Stepsmax = arma::conv_to<arma::sp_mat>::from(stepsmax);
  }

  int stepsminmaxsum = arma::accu(arma::sum(arma::sum(minmaxS > 0)));
  if (stepsminmaxsum) {
    throw "Error; Lower bound prob > upper bound prob";
  }
}
/* constructs a grid over a rectangular space
// input:
//          - boundary: vertices of the boundary
//          - gridsize: rectangle lengths in x and y direction
//           - tol: tolerance for the length of reCtangle in x & y directions
//                in refinement
// output:
//           - states_valid: cell numbers of the rectangles in the boundary
//           (last state is out of boundary state)
//           - ver_valid: a cell structure containing the vertices of the valid
//           states
//           - ver_valid_vec: vertices of the valid states in a vector form:
//              column i corresponds to the vertices of state i:
//              x-coordinates are 1:4 and y-coordinations are 5:8
//        boundary, gridsize, tol   - discretization_ctr: the center of mass for
//        the discretize
//              domain
//  		- states_all: cell numbers of all the rectangles (last state is out
//  of boundary state)
//          - ver_all: a cell structure containing the vertices of all the
//          states
*/
arma::vec bmdp_t::getGridNondiag(arma::mat boundary, arma::mat gridsize,arma::mat tol, int i) {

  // get x and y- coordinated of the grid cells in each direction
  arma::vec xcells = arma::regspace(arma::min(boundary.row(0)), gridsize(0),
                                    arma::max(boundary.row(0)));

  // check if last cell is outside grid
  int end = xcells.n_elem - 1;
  double xmax = arma::max(boundary.row(0));
  bool needMore = (xcells(end) != xmax);
  if (needMore) {
    float newpoint = xcells(end) + gridsize(0);
    xcells.resize(
        end + 2); // resize vector such that new point is added in last position
    xcells(end + 1) = newpoint;
  }
  if(xcells.n_elem < 4) {
    std::cout << "Too large of a grid size " << std::endl;
    exit(0);
  }

  // Do same for y coordinates
  arma::vec ycells = arma::regspace(arma::min(boundary.row(1)), gridsize(1),
                                    arma::max(boundary.row(1)));
  // check if last cell is outside grid
  int endy = ycells.n_elem - 1;
  double ymax = arma::max(boundary.row(1));
  bool needMorey = (ycells(endy) != ymax);
  if (needMorey) {
    float newpoint = ycells(endy) + gridsize(1);
    ycells.resize(
        endy +
        2); // resize vector such that new point is added in last position
    ycells(endy + 1) = newpoint;
  }
  if(ycells.n_elem < 4) {
    std::cout << "Too large of a grid size " << std::endl;
    exit(0);
  }

  // compute the vertices of each cell grid
  arma::cube ver_all(boundary.n_rows, boundary.n_cols,
                     (xcells.n_elem * ycells.n_elem + 1));
  unsigned counter = 0;
  std::vector<int> validcellnum;

  for (unsigned vy = 0; vy < ycells.n_elem - 1; ++vy) {
    for (unsigned vx = 0; vx < xcells.n_elem - 1; ++vx) {
      arma::vec x = {xcells(vx, 0), xcells(vx, 0),
                     (xcells(vx, 0) + gridsize(0)),
                     (xcells(vx, 0) + gridsize(0))};
      arma::vec y = {ycells(vy, 0), ycells(vy, 0) + gridsize(1),
                     ycells(vy, 0) + gridsize(1), ycells(vy, 0)};
      arma::mat v(x.n_elem, 2);
      v = arma::join_vert(x.t(), y.t());

      // check if the cell is within the boundary
      int inpoly = pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                          boundary.row(0), boundary.row(1), v.row(0), v.row(1));
      if (inpoly) {
        counter++;
        if ((counter) > ver_all.n_slices) {
          ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
        }
        ver_all.slice(counter - 1) = v;
        validcellnum.push_back(counter - 1);
      } else {
        // refine until u rach the required xtol and ytol
        arma::mat v1 = v.row(0);
        arma::mat v2 = v.row(1); // TODO: Check if really need this step
        arma::mat toberef = arma::join_horiz(v1, v2);

        while (!(toberef.is_empty())) {
          // if not in the boundary refine once
          arma::mat ver = refineRectangle(toberef.row(0));
          if (toberef.n_rows > 1) {
            toberef = toberef.rows(1, toberef.n_rows - 1);
          } else {
            toberef.reset();
          }
          for (int i = 0; i < std::pow(2, this->desc.dyn.dynamics[0].x_dim);
               ++i) {
            int inboundary =
                pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                       boundary.row(0), boundary.row(1),
                       ver(i, arma::span(0, 3)), ver(i, arma::span(4, 7)));

            if (inboundary) {
              counter++;
              arma::mat vtx = ver(i, arma::span(0, 3));
              arma::mat vty = ver(i, arma::span(4, 7));
              arma::mat vt = arma::join_vert(vtx, vty);

              if ((counter) > ver_all.n_slices) {
                ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
              }
              ver_all.slice(counter - 1) = vt;
              // ver_all_temp.push_back(vt);
              validcellnum.push_back(counter - 1);
            } else if (((arma::sum(inboundary) == 0) ||
                        ((arma::max(ver(i, arma::span(0, 3))) -
                          arma::min(ver(i, arma::span(0, 3)))) < tol(0, 0))) &&
                       (((arma::max(ver(i, arma::span(4, 7))) -
                          arma::min(ver(i, arma::span(4, 7)))) < tol(0, 1)))) {
              counter++;
              arma::mat vtx = ver(i, arma::span(0, 3));
              arma::mat vty = ver(i, arma::span(4, 7));
              arma::mat vt = arma::join_vert(vtx, vty);

              if ((counter) > ver_all.n_slices) {
                ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
              }
              ver_all.slice(counter - 1) = vt;
            } else {
              toberef = arma::join_vert(toberef, ver.row(i));
            }
          }
        }
      }
    }
  }
  arma::vec stalettes_all = arma::regspace(1, counter);

  // assign valid vertices and compute center of mass
  int validcellnum_L = validcellnum.size();
  std::vector<arma::mat> ver_valid;
  arma::mat ver_valid_vec = arma::zeros<arma::mat>(8, validcellnum_L);
  arma::mat ver_valid_ctr = arma::zeros<arma::mat>(2, validcellnum_L);
  arma::mat ver_valid_area = arma::zeros<arma::mat>(1, validcellnum_L);
  for (int i = 0; i < validcellnum_L; ++i) {
    arma::mat a = ver_all.slice(validcellnum[i]);
    ver_valid.push_back(a);
    arma::mat tempv = join_horiz(a.row(0), a.row(1));
    ver_valid_vec.col(i) = tempv.t();
    ver_valid_ctr(0, i) = arma::mean(ver_valid[i].row(0));
    ver_valid_ctr(1, i) = arma::mean(ver_valid[i].row(1));
    ver_valid_area(i) =
        polygonArea(ver_valid[i].row(0).t(), ver_valid[i].row(1).t());
  }
  arma::vec states_valid = arma::regspace(1, validcellnum.size());

  // center of the discretuzation (center of mass)
  arma::vec discretization_ctr(2);
  discretization_ctr(0) = arma::accu(ver_valid_ctr.row(0) % ver_valid_area) /
                          arma::accu(ver_valid_area);
  discretization_ctr(1) = arma::accu(ver_valid_ctr.row(1) % ver_valid_area) /
                          arma::accu(ver_valid_area);
  Strmode newMode;
  if (this->mode.size() > (unsigned)i ||
      (this->mode.size() == 1 && (unsigned)i == 0)) {
    this->mode[i].vertices = ver_valid;
    // this->mode[i].vertices_vec = ver_valid_vec;
    this->mode[i].mode_center = discretization_ctr;

  } else {
    newMode.vertices = ver_valid;
    //	newMode.vertices_vec = ver_valid_vec;
    newMode.mode_center = discretization_ctr;
    newMode.states = states_valid;
    this->mode.push_back(newMode);
  }

  return states_valid;
}
/*
// constructs a grid over a rectangular space
// input:
//          - boundary: vertices of the boundary
//          - gridsize: rectangle lengths in x and y direction
//           - tol: tolerance for the length of reCtangle in x & y directions
//                in refinement
// output:
//           - states_valid: cell numbers of the rectangles in the boundary
//           (last state is out of boundary state)
//           - ver_valid: a cell structure containing the vertices of the valid
//           states
//           - ver_valid_vec: vertices of the valid states in a vector form:
//              column i corresponds to the vertices of state i:
//              x-coordinates are 1:4 and y-coordinations are 5:8
//        boundary, gridsize, tol   - discretization_ctr: the center of mass for
//        the discretize
//              domain
//  		- states_all: cell numbers of all the rectangles (last state is out
//  of boundary state)
//          - ver_all: a cell structure containing the vertices of all the
//          states*/
arma::vec bmdp_t::getGridNondiagRA(arma::mat boundary, arma::mat gridsize,arma::mat tol, int c_mode,std::vector<std::vector<arma::mat>> v_phi1,std::vector<std::vector<arma::mat>> v_phi2) {

  // get x and y- coordinated of the grid cells in each direction
  arma::vec xcells = arma::regspace(arma::min(boundary.row(0)), gridsize(0),
                                    arma::max(boundary.row(0)));
  if(xcells.n_elem < 4) {
    std::cout << "Too large of a grid size " << std::endl;
    exit(0);
  }
  // check if last cell is outside grid
  int end = xcells.n_elem - 1;
  double xmax = arma::max(boundary.row(0));
  bool needMore = (xcells(end) != xmax);
  if (needMore) {
    float newpoint = xcells(end) + gridsize(0);
    xcells.resize(
        end + 2); // resize vector such that new point is added in last position
    xcells(end + 1) = newpoint;
  }

  // Do same for y coordinates
  arma::vec ycells = arma::regspace(arma::min(boundary.row(1)), gridsize(1),
                                    arma::max(boundary.row(1)));
  if(ycells.n_elem < 4) {
    std::cout << "Too large of a grid size " << std::endl;
    exit(0);
  }
  // check if last cell is outside grid
  int endy = ycells.n_elem - 1;
  double ymax = arma::max(boundary.row(1));
  bool needMorey = (ycells(endy) != ymax);
  if (needMorey) {
    float newpoint = ycells(endy) + gridsize(1);
    ycells.resize(
        endy +
        2); // resize vector such that new point is added in last position
    ycells(endy + 1) = newpoint;
  }

  // compute the vertices of each cell grid
  arma::cube ver_all(boundary.n_rows, boundary.n_cols,
                     (xcells.n_elem * ycells.n_elem + 1));
  unsigned counter = 0;
  std::vector<int> validcellnum;
  unsigned in_phi = 0;
  for (unsigned vy = 0; vy < ycells.n_elem - 1; ++vy) {
    for (unsigned vx = 0; vx < xcells.n_elem - 1; ++vx) {
      arma::vec x = {xcells(vx, 0), xcells(vx, 0),
                     (xcells(vx, 0) + gridsize(0)),
                     (xcells(vx, 0) + gridsize(0))};
      arma::vec y = {ycells(vy, 0), ycells(vy, 0) + gridsize(1),
                     ycells(vy, 0) + gridsize(1), ycells(vy, 0)};
      arma::mat v(x.n_elem, 2);
      v = arma::join_vert(x.t(), y.t());

      // Get coarse grid
      // check if the cell is within the boundary
      int inpoly = pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                          boundary.row(0), boundary.row(1), v.row(0), v.row(1));
      in_phi = 0;
      if (inpoly) {
        arma::mat or_v = v;
        // Check if within phi1
        if (pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                   v_phi1[1][c_mode].row(0), v_phi1[1][c_mode].row(1), v.row(0),
                   v.row(1))) {
          in_phi = 1;
          // refine until u rach the required xtol and ytol
          arma::mat v1 = v.row(0);
          arma::mat v2 = v.row(1);
          arma::mat toberef = arma::join_horiz(v1, v2);
          while (!(toberef.is_empty())) {
            // if not in the boundary refine once
            arma::mat ver = refineRectangle(toberef.row(0));
            if (toberef.n_rows > 1) {
              toberef = toberef.rows(1, toberef.n_rows - 1);
            } else {
              toberef.reset();
            }
            for (int xi = 0; xi < std::pow(2, this->desc.dyn.dynamics[0].x_dim);
                 ++xi) {
              int inboundary =
                  pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                         v_phi1[1][c_mode].row(0), v_phi1[1][c_mode].row(1),
                         ver(xi, arma::span(0, 3)), ver(xi, arma::span(4, 7)));

              if (inboundary) {
                counter++;
                arma::mat vtx = ver(xi, arma::span(0, 3));
                arma::mat vty = ver(xi, arma::span(4, 7));
                arma::mat vt = arma::join_vert(vtx, vty);

                if ((counter) > ver_all.n_slices) {
                  ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
                }
                ver_all.slice(counter - 1) = vt;
                // ver_all_temp.push_back(vt);
                validcellnum.push_back(counter - 1);
              } else if (((arma::sum(inboundary) == 0) ||
                          ((arma::max(ver(xi, arma::span(0, 3))) -
                            arma::min(ver(xi, arma::span(0, 3)))) <
                           tol(0, 0))) &&
                         (((arma::max(ver(xi, arma::span(4, 7))) -
                            arma::min(ver(xi, arma::span(4, 7)))) <
                           tol(0, 1)))) {
                counter++;
                arma::mat vtx = ver(xi, arma::span(0, 3));
                arma::mat vty = ver(xi, arma::span(4, 7));
                arma::mat vt = arma::join_vert(vtx, vty);

                if ((counter) > ver_all.n_slices) {
                  ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
                }
                ver_all.slice(counter - 1) = vt;
              } else {
                toberef = arma::join_vert(toberef, ver.row(xi));
              }
            }
          }
        }
        // Check if within phi2
        if (pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                   v_phi2[1][c_mode].row(0), v_phi2[1][c_mode].row(1), v.row(0),
                   v.row(1))) {
          in_phi = 1;
          // refine until u rach the required xtol and ytol
          arma::mat v1 = v.row(0);
          arma::mat v2 = v.row(1);
          arma::mat toberef = arma::join_horiz(v1, v2);
          while (!(toberef.is_empty())) {
            // if not in the boundary refine once
            arma::mat ver = refineRectangle(toberef.row(0));
            if (toberef.n_rows > 1) {
              toberef = toberef.rows(1, toberef.n_rows - 1);
            } else {
              toberef.reset();
            }
            for (int xi = 0; xi < std::pow(2, this->desc.dyn.dynamics[0].x_dim);
                 ++xi) {
              int inboundary =
                  pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                         v_phi2[1][c_mode].row(0), v_phi2[1][c_mode].row(1),
                         ver(xi, arma::span(0, 3)), ver(xi, arma::span(4, 7)));

              if (inboundary) {
                counter++;
                arma::mat vtx = ver(xi, arma::span(0, 3));
                arma::mat vty = ver(xi, arma::span(4, 7));
                arma::mat vt = arma::join_vert(vtx, vty);

                if ((counter) > ver_all.n_slices) {
                  ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
                }
                ver_all.slice(counter - 1) = vt;
                // ver_all_temp.push_back(vt);
                validcellnum.push_back(counter - 1);
              } else if (((arma::sum(inboundary) == 0) ||
                          ((arma::max(ver(xi, arma::span(0, 3))) -
                            arma::min(ver(xi, arma::span(0, 3)))) <
                           tol(0, 0))) &&
                         (((arma::max(ver(xi, arma::span(4, 7))) -
                            arma::min(ver(xi, arma::span(4, 7)))) <
                           tol(0, 1)))) {
                counter++;
                arma::mat vtx = ver(xi, arma::span(0, 3));
                arma::mat vty = ver(xi, arma::span(4, 7));
                arma::mat vt = arma::join_vert(vtx, vty);

                if ((counter) > ver_all.n_slices) {
                  ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
                }
                ver_all.slice(counter - 1) = vt;
              } else {
                toberef = arma::join_vert(toberef, ver.row(xi));
              }
            }
          }
        }
        if (in_phi == 0) {
          counter++;
          if ((counter) > ver_all.n_slices) {
            ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
          }
          ver_all.slice(counter - 1) = v;
          validcellnum.push_back(counter - 1);
        }

      } else {
        // refine until u rach the required xtol and ytol
        arma::mat v1 = v.row(0);
        arma::mat v2 = v.row(1); // TODO: Check if really need this step
        arma::mat toberef = arma::join_horiz(v1, v2);

        while (!(toberef.is_empty())) {
          // if not in the boundary refine once
          arma::mat ver = refineRectangle(toberef.row(0));
          if (toberef.n_rows > 1) {
            toberef = toberef.rows(1, toberef.n_rows - 1);
          } else {
            toberef.reset();
          }
          for (int ki = 0; ki < std::pow(2, this->desc.dyn.dynamics[0].x_dim);
               ++ki) {
            int inboundary = pnpoly(
                std::pow(2, this->desc.dyn.dynamics[0].x_dim), boundary.row(0),
                boundary.row(1), ver(c_mode, arma::span(0, 3)),
                ver(c_mode, arma::span(4, 7)));

            if (inboundary) {
              counter++;
              arma::mat vtx = ver(ki, arma::span(0, 3));
              arma::mat vty = ver(ki, arma::span(4, 7));
              arma::mat vt = arma::join_vert(vtx, vty);

              if ((counter) > ver_all.n_slices) {
                ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
              }
              ver_all.slice(counter - 1) = vt;
              // ver_all_temp.push_back(vt);
              validcellnum.push_back(counter - 1);
            } else if (((arma::sum(inboundary) == 0) ||
                        ((arma::max(ver(ki, arma::span(0, 3))) -
                          arma::min(ver(ki, arma::span(0, 3)))) < tol(0, 0))) &&
                       (((arma::max(ver(ki, arma::span(4, 7))) -
                          arma::min(ver(ki, arma::span(4, 7)))) < tol(0, 1)))) {
              counter++;
              arma::mat vtx = ver(ki, arma::span(0, 3));
              arma::mat vty = ver(ki, arma::span(4, 7));
              arma::mat vt = arma::join_vert(vtx, vty);

              if ((counter) > ver_all.n_slices) {
                ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter);
              }
              ver_all.slice(counter - 1) = vt;
            } else {
              toberef = arma::join_vert(toberef, ver.row(ki));
            }
          }
        }
      }
    }
  }

  arma::vec stalettes_all = arma::regspace(1, counter);
  // assign valid vertices and compute center of mass
  int validcellnum_L = validcellnum.size();
  std::vector<arma::mat> ver_valid;
  arma::mat ver_valid_vec = arma::zeros<arma::mat>(8, validcellnum_L);
  arma::mat ver_valid_ctr = arma::zeros<arma::mat>(2, validcellnum_L);
  arma::mat ver_valid_area = arma::zeros<arma::mat>(1, validcellnum_L);
  for (int i = 0; i < validcellnum_L; ++i) {
    arma::mat a = ver_all.slice(validcellnum[i]);
    ver_valid.push_back(a);
    arma::mat tempv = join_horiz(a.row(0), a.row(1));
    ver_valid_vec.col(i) = tempv.t();
    ver_valid_ctr(0, i) = arma::mean(ver_valid[i].row(0));
    ver_valid_ctr(1, i) = arma::mean(ver_valid[i].row(1));
    ver_valid_area(i) =
        polygonArea(ver_valid[i].row(0).t(), ver_valid[i].row(1).t());
  }
  arma::vec states_valid = arma::regspace(1, validcellnum.size());

  // center of the discretuzation (center of mass)
  arma::vec discretization_ctr(2);
  discretization_ctr(0) = arma::accu(ver_valid_ctr.row(0) % ver_valid_area) /
                          arma::accu(ver_valid_area);
  discretization_ctr(1) = arma::accu(ver_valid_ctr.row(1) % ver_valid_area) /
                          arma::accu(ver_valid_area);
  Strmode newMode;
  if (this->mode.size() > (unsigned)c_mode ||
      (this->mode.size() == 1 && (unsigned)c_mode == 0)) {
    this->mode[c_mode].vertices = ver_valid;
    // this->mode[i].vertices_vec = ver_valid_vec;
    this->mode[c_mode].mode_center = discretization_ctr;

  } else {
    newMode.vertices = ver_valid;
    newMode.mode_center = discretization_ctr;
    newMode.states = states_valid;
    this->mode.push_back(newMode);
  }
  return states_valid;
}
/*
// constructs a grid over a rectangular space
// input:
//          - boundary: vertices of the boundary
//          - gridsize: rectangle lengths in x and y direction
//           - tol: tolerance for the length of reCtangle in x & y directions
//                in refinement
// output:
//           - states_valid: cell numbers of the rectangles in the boundary
//           (last state is out of boundary state)
//           - ver_valid: a cell structure containing the vertices of the valid
//           states
//           - ver_valid_vec: vertices of the valid states in a vector form:
//              column i corresponds to the vertices of state i:
//              x-coordinates are 1:4 and y-coordinations are 5:8
//        boundary, gridsize, tol   - discretization_ctr: the center of mass for
//        the discretize
//              domain
//  		- states_all: cell numbers of all the rectangles (last state is out
//  of boundary state)
//          - ver_all: a cell structure containing the vertices of all the
//          states
*/
arma::vec bmdp_t::getGrid(arma::mat boundary, arma::mat gridsize, arma::mat tol,int m) {

  // Number of continuous variables
  int x_dim = this->desc.dyn.dynamics[0].x_dim;

  // get coordinates of the grid cells in each direction
  // Coordinates for x-direction
  // Split into a hyper rectangle for each cell
  // cols are in pairs with min and max of each cell
  arma::mat xcells = arma::regspace(arma::min(boundary(0, 0)), gridsize(0, 0),
                                    arma::max(boundary(0, 1)));
  // obtain number of basic cells needed
  // basic i.e. not with refinement
  arma::vec basicNum(x_dim);
  basicNum(0) = xcells.n_rows - 1;
  double totalBN = basicNum(0);

  double xmax = boundary(0, 1);
  bool needMore = 0;
  if((xcells(xcells.n_elem - 1) -  xmax ) > 0.005 ||(xcells(xcells.n_elem - 1) -  xmax ) < -0.005  ){ // To account for imprecisions
    needMore = 1;
  }
  if (needMore) {
    float newpoint = xcells(xcells.n_elem - 1) + gridsize(0);
    xcells.resize(
        xcells.n_elem - 1 +
        2); // resize vector such that new point is added in last position
    xcells(xcells.n_elem - 1) = newpoint;
  }
  unsigned or_size = xcells.n_elem;
  arma::vec x_all_cells(xcells.n_elem + xcells.n_elem - 2);
  arma::mat cells = xcells;

  int count = 0;
  if (x_dim > 1) {
    unsigned count = 1;
    arma::vec x_new(x_all_cells.n_elem - or_size + 1);
    for (unsigned i = 0; i < x_new.n_elem; ++i) {
      x_new(i) = xcells(count);
      count++;
    }
    count = 0;
    for (unsigned i = 0; i < x_all_cells.n_elem - 1; i = i + 2) {
      x_all_cells(i) = xcells(count);
      x_all_cells(i + 1) = x_new(count);
      count++;
    }
    count = 1;
    xcells.clear();
    x_new.clear();
    // Repeat process for remainder of cells in all directions
    cells.resize(x_dim, x_all_cells.n_rows);
    cells.row(0) = x_all_cells.t();

    for (int i = 1; i < x_dim; i++) {
      xcells = arma::regspace(arma::min(boundary(i, 0)), gridsize(0, i),
                              arma::max(boundary(i, 1)));
      basicNum(1) = xcells.n_rows - 1;
      totalBN *= basicNum(1);
      double xmax = boundary(i, 1);
      bool needMore = (xcells(xcells.n_elem - 1) != xmax);

      if (needMore) {
        float newpoint = xcells(xcells.n_elem - 1) + gridsize(0,i);
        xcells.resize(xcells.n_elem - 1 +2); // resize vector such that new point is added in last position
        xcells(xcells.n_elem - 1) = newpoint;
      }
      x_new.resize(xcells.n_elem);

      for (unsigned i_in = 0; i_in < x_new.n_elem; ++i_in) {
        if(count > xcells.n_elem -1){
          x_new(i_in) = arma::datum::nan;
        }
        else{
          x_new(i_in) = xcells(count);
        }
        count++;
      }
      count = 0;
      for (unsigned i_in2 = 0; i_in2 < x_all_cells.n_elem - 1; i_in2 = i_in2 + 2) {
        if(count > xcells.n_elem -1){
          x_all_cells(i_in2) = arma::datum::nan;
          x_all_cells(i_in2 + 1) =arma::datum::nan;
        }
        else{
          x_all_cells(i_in2) = xcells(count);
          x_all_cells(i_in2 + 1) = x_new(count);
        }
        count++;
      }
      count = 1;

      cells.row(i) = x_all_cells.t();
    }
    xcells.clear();
    x_all_cells.clear();
    x_new.clear();
  }
  // Obtained all grid cells defined as hyper rectangles
  // stored x_all_cells
  count = 0;

  // Compute the total possible combinations of cells
  // to obtain grid
  double counter = 0;

  // Define where all vertices are being stored
  // in terms of hyper rectangles
  std::vector<double> validcellnum;

  int slices = std::pow((cells.n_cols / 2), x_dim);
  arma::cube ver_all(boundary.n_rows, boundary.n_cols, slices);
  arma::mat boundary_tol = boundary;
  boundary.col(1) = boundary.col(1) + 0.99995 * tol.t();

  // Get pairs of hyper rectangles and for pair
  // permute along each dimension to get an individual cell
  std::vector<arma::mat> inC;
  arma::mat current_idx(1, 2), ind_cell(x_dim, 2);
  std::vector<arma::mat> all_cells;

  if (x_dim > 1) {
    rec(all_cells, cells, cells.cols(0, 1), 0, cells.n_cols, cells.n_rows, 0);
  } else {
    if (cells.n_elem == 1) {
      std::cout << "Too coarse a grid for dimension size of 1." << std::endl;
      exit(0);
    }
    for (unsigned vx = 0; vx < cells.n_elem - 1; ++vx) {
      all_cells.push_back(cells.row(vx));
    }
  }

  for (unsigned r = 0; r < all_cells.size(); r++) {
    ind_cell = all_cells[r];

    // Check if point lies within boundary rectangle
    // defined also using a HyperRect
    int inpoly = pnHyperRect(ind_cell, boundary_tol);
    if(!ind_cell.has_nan()){
      if (inpoly) {
        if ((counter) > (int)ver_all.n_slices - 1) {
          ver_all.resize(ver_all.n_rows, ver_all.n_cols, counter + 1);
        }
        ver_all.slice(counter) = ind_cell;
        validcellnum.push_back(counter);
        counter++;
      }
      else {
        // If inside boundary store cells
        // else split cell into smaller cells (i.e. refine it)

        // refine hyper rectangles until they lay within
        // tolerance requirement

        // 1. Check if cell lies within tolerance
        // 2. Not then refine
        int jp = 0;
        while (pnHyperRect(ind_cell, boundary) && jp <((2*x_dim) - 1) ) {
          // Refine only along dimensions which are outside tolerance
          // Refining by splitting into 2 for each dimension, resulting in
          // 2^(x_dim) cells
          arma::mat refined =
              refineHyperRectangle(ind_cell);

          // Reroder to get all possible combinations following split
          arma::mat new_cell(x_dim, 2);
          count = 0;
          double toKeep = 0, prev = -1;
          // Get combinations of new cells depending on
          // the number of dimensions to be refined
          std::vector<arma::mat> ref_cells;

          rec(ref_cells, refined, refined.cols(0, 1), 0, refined.n_cols,refined.n_rows, 0);

          for (unsigned c = 0; c < ref_cells.size(); c++) {
            new_cell = ref_cells[c];

            // Check if new point lies within boundary
            int inboundary = pnHyperRect(new_cell, boundary_tol);
            // Store Cell as part of grid if it lies within
            // boundary
            // else need to resplit
            if (inboundary) {
              jp++;
              if ((counter) > (int)ver_all.n_slices - 1) {
                ver_all.resize(ver_all.n_rows, ver_all.n_cols,
                               ver_all.n_slices + 1);
              }
              ver_all.slice(counter) = new_cell;
              validcellnum.push_back(counter);
              counter++;
              ind_cell = new_cell;
            }
            else if(arma::accu((new_cell.col(1) - new_cell.col(0)) < tol.t()) == x_dim){
              if ((counter) > (int)ver_all.n_slices - 1) {
                ver_all.resize(ver_all.n_rows, ver_all.n_cols,
                               ver_all.n_slices + 1);
              }
              ver_all.slice(counter) = new_cell;
              counter++;
              ind_cell = new_cell;
              jp++;
            }
            else {
              // Store only the cell which is closer to the boundary
              toKeep = arma::accu(new_cell.col(1) - boundary.col(1));
              if (prev == -1) {
                prev = toKeep;
                ind_cell = new_cell;

              } else {
                if (toKeep < prev) {
                  ind_cell = new_cell;
                  prev = toKeep;
                }
              }
            }
          }
        }
      }
    }

  }

  // Number of grid cells
  arma::vec stalettes_all = arma::regspace(1, counter);
  // assign valid vertices and compute center of mass
  unsigned validcellnum_L = validcellnum.size();
  std::vector<arma::mat> ver_valid;
  Strmode newMode;
  arma::vec states_valid = arma::regspace(1, validcellnum.size());
  arma::mat ver_valid_ctr = arma::zeros<arma::mat>(x_dim, validcellnum_L);
  arma::mat ver_valid_vec = arma::zeros<arma::mat>(x_dim, validcellnum_L);
  for (unsigned i = 0; i < validcellnum_L; ++i) {
    ver_valid.push_back(ver_all.slice(validcellnum[i]));
    ver_valid_ctr.col(i) = arma::mean(ver_valid[i], 1);
  }

  if (this->mode.size() > (unsigned)m ||
      (this->mode.size() == 1 && (unsigned)m == 0)) {
    this->mode[m].vertices = ver_valid;
    this->mode[m].mode_center = ver_valid_ctr;

  } else {
    newMode.vertices = ver_valid;
    newMode.mode_center = ver_valid_ctr;
    newMode.states = states_valid;
    this->mode.push_back(newMode);
  }
  return states_valid;
}
// this function defines the modes of system so that in each mode the
// 1-step covariance is diagonal
// input:    - sys: system with all its dynamics and noise
// output:   - sys.mode:
void bmdp_t::getModes(Sys sys) {
  int num_dyn = sys.dyn.dynamics[0].A.n_elem;

  for (int a = 0; a < num_dyn; ++a) {
    arma::mat A = sys.dyn.dynamics[a].A;
    arma::mat F = sys.dyn.dynamics[a].F;
    arma::mat Sigma = sys.dyn.dynamics[a].sigma;

    arma::mat Qpost = F * Sigma * F.t();

    // diagonalise Qpost
    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    arma::eig_gen(eigval, eigvec, Qpost);
    arma::mat V = arma::conv_to<arma::mat>::from(eigval);
    arma::mat D = arma::conv_to<arma::mat>::from(eigvec);

    // transformation function
    arma::mat Tf = arma::pow(D, -0.5) * V.t();

    // get the boundary of the mode
    sys.mode[a].boundary = Tf * sys.boundary;
  }
  this->desc.mode = sys.mode;
}

arma::uvec bmdp_t::getLabels(std::string phiFile, int x_dim,bool under_approx) {
  // Get labels for phi1, if it is not present then "true"
  std::string line;
  std::ifstream myfile(phiFile);

  arma::uvec phi = arma::zeros<arma::uvec>(this->Stepsmax.n_cols);
  std::vector<arma::mat> v_labels;
  if (myfile.is_open()) {
    arma::mat temp_v(x_dim, std::pow(2, x_dim));
    auto index = 0;
    while (std::getline(myfile, line)) {
      std::istringstream iss(line);
      std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>());

      if (index == x_dim) {
        v_labels.push_back(temp_v);
        index = 0;
      }

      for (size_t i = 0; i < results.size(); i++) {
        temp_v(index, i) = std::stod(results[i]);
      }
      index++;
    }
    v_labels.push_back(temp_v);

    // Obtained vertices associated with labels
    // Now need to connect these vertices with the states
    size_t num_dyn = this->mode.size();

    // Transform vertices to post and adjust depending on over and under
    // approximation
    std::vector<std::vector<arma::mat>> v_labels_mode;
    std::vector<arma::mat> v_mode;
    for (size_t i = 0; i < num_dyn; ++i) {
      for (size_t j = 0; j < v_labels.size(); ++j) {
        arma::mat temp = this->desc.mode[i].transfermatrix * v_labels[j];
        v_mode.push_back(temp);
      }
      v_labels_mode.push_back(v_mode);
    }

    index = 0;
    for (size_t q = 0; q < num_dyn; ++q) {
      for (size_t a = 0; a < this->mode[q].vertices.size(); ++a) {
        arma::mat crnt_vertex = this->mode[q].vertices[a];
        if (crnt_vertex.n_cols != std::pow(2, this->desc.dyn.dynamics[0].x_dim )) {
          arma::mat crnt_vertex_or = crnt_vertex;
          crnt_vertex.resize(2,4);
          crnt_vertex(1, 0) = crnt_vertex_or(0, 0);
          crnt_vertex(1, 1) = crnt_vertex_or(0, 1);
          crnt_vertex(1, 2) = crnt_vertex_or(0, 1);
          crnt_vertex(1, 3) = crnt_vertex_or(0, 0);
          crnt_vertex(0, 0) = crnt_vertex_or(1, 0);
          crnt_vertex(0, 1) = crnt_vertex_or(1, 1);
          crnt_vertex(0, 2) = crnt_vertex_or(1, 0);
          crnt_vertex(0, 3) = crnt_vertex_or(1, 1);

        }
        // For current vertex check if it is within
        // vertex - tol for under
        // vertex + told for over
        for (size_t i = 0; i < v_labels_mode[q].size(); i++) {
          int inpoly =
              pnpoly(std::pow(2, this->desc.dyn.dynamics[0].x_dim),
                     v_labels_mode[q][i].row(0), v_labels_mode[q][i].row(1),
                     crnt_vertex.row(1), crnt_vertex.row(0));
          if (inpoly) {
            phi(index) = 1;
          }
        }
        index++;
      }
    }
  } else {
    std::cout << "File not present, setting label as true for all states "
                 "except last one"
              << std::endl;
    phi = arma::ones<arma::uvec>(this->Stepsmax.n_cols);
    phi(phi.n_rows - 1, 0) = 0;
  }
  return phi;
}

std::vector<std::vector<arma::mat>>
bmdp_t::getLabelVertices(std::string phiFile, int x_dim) {
  // Get labels for phi1, if it is not present then "true"
  std::string line;
  std::ifstream myfile(phiFile);
  std::vector<arma::mat> v_labels;
  std::vector<std::vector<arma::mat>> v_labels_mode;
  if (myfile.is_open()) {
    arma::mat temp_v(x_dim, std::pow(2, x_dim));
    auto index = 0;
    while (std::getline(myfile, line)) {
      std::istringstream iss(line);
      std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>());
      if (index == x_dim) {
        v_labels.push_back(temp_v);
        index = 0;
      }

      for (size_t i = 0; i < results.size(); i++) {
        temp_v(index, i) = std::stod(results[i]);
      }
      index++;
    }
    v_labels.push_back(temp_v);

    // Obtained vertices associated with labels
    // Now need to connect these vertices with the states
    size_t num_dyn = this->desc.mode.size();

    // Transform vertices to post and adjust depending on over and under
    // approximation

    std::vector<arma::mat> v_mode;
    for (size_t i = 0; i < num_dyn; ++i) {
      for (size_t j = 0; j < v_labels.size(); ++j) {
        arma::mat temp = this->desc.mode[i].transfermatrix * v_labels[j];
        v_mode.push_back(temp);
      }
      v_labels_mode.push_back(v_mode);
    }
  } else {
    throw "no phi1 labels";
  }

  return v_labels_mode;
}

// Create file for synthesiser
// Safety property
void bmdp_t::createSynthFile(arma::uvec phi1, arma::uvec Labels) {
  // True
  arma::uvec phi1_indV = phi1;
  arma::uvec phi2_indV = Labels;

  arma::uvec Qyes = arma::find(phi2_indV);
  arma::uvec states = arma::ones<arma::uvec>(this->states.n_rows);
  states.resize(states.n_rows + 1);
  states(states.n_rows - 1) = 1;
  arma::uvec Qno = arma::find(states != (phi1_indV || phi2_indV));
  int stateNum = this->states.n_rows + 1;
  int actNum = this->actNum;

  // Write to file for performing synthesis
  std::ofstream myfile;
  if(checkFolderExists("../results") == -1) {
    if(mkdir("../results", 0777) == -1) {
       std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
       exit(0);
    }
  }
  myfile.open("../results/bmdp.txt");
  myfile << stateNum << std::endl;    // number of states
  myfile << actNum << std::endl;      // number of actions
  myfile << Qyes.n_rows << std::endl; // number of terminal states
  for (size_t i = 0; i < Qyes.n_rows; i++) {
    myfile << Qyes(i) << std::endl;
  }
  // myfile << std::endl;
  arma::mat Stepsmax = arma::conv_to<arma::mat>::from(this->Stepsmax);

  arma::mat Stepsmin = arma::conv_to<arma::mat>::from(this->Stepsmin);

  for (int i = 0; i < stateNum; i++) {
    arma::uvec Qt = arma::find(Qno == i);
    if (Qno.is_empty() || Qt.is_empty()) {
      for (int a = 0; a < actNum; a++) {
        arma::vec ij = Stepsmax.row((i)*actNum + a).t();
        double ij_Sum = arma::sum(ij);
        if (ij_Sum < 1) {
          double remain = 1 - ij_Sum;
          for (unsigned j = 0; j < ij.n_rows; j++) {
            this->Stepsmin((i)*actNum + a, j) =
                this->Stepsmin((i)*actNum + a, j) + remain;
          }
        }
        for (unsigned j = 0; j < ij.n_rows; j++) {
          myfile << i << " " << a << " " << j << " "
                 << this->Stepsmin((i)*actNum + a, j) << " "
                 << this->Stepsmax((i)*actNum + a, j) << " " << std::endl;
          if (i < stateNum || j < ij(ij.n_cols - 1) || a < actNum) {
            myfile << std::endl;
          }
        }
      }
    } else {
      myfile << i << " " << 0 << " " << i << " " << 1.0 << " "
             << 1.0; // number of actions
      if (i < stateNum) {
        myfile << std::endl;
      }
    }
  }
  myfile.close();
}

void bmdp_t::populateBMDPSpec(matvar_t &content) 
{
  // For Tq check type for hybrid systems guards are sorted in cell type
  // For stochastic systems unless Tq is function of state, Tq is
  // given in form of numeric matrix and one has to check for
  // stochasticity

  std::string boundary{"boundary"}, gridsize{"gridsize"}, reftol{"reftol"};
  std::string::size_type sz;

  size_t stride = Mat_SizeOf(content.data_type);
  std::string data  = (char *) content.data;
  
  arma::mat mt(content.dims[0], content.dims[1]);
  unsigned i, j = 0;
  
  for (i = 0; i < content.dims[0] && i < 15; i++) 
  {
    for (j = 0; j < content.dims[1] && j < 15; j++) 
    {
      size_t idx = content.dims[0] * j + i;
      data += std::to_string(idx*stride);
      double val = std::stod(data.c_str(), &sz);
      mt(i, j) = val;
    }
  }

  
  if (boundary.compare(content.name) == 0)
  {
    this->desc.boundary = mt;
  } 
  else if (gridsize.compare(content.name) == 0)
  {
    this->desc.gridsize = mt;
  } 
  else if (reftol.compare(content.name) == 0)
  {
    this->desc.reftol = mt;
  }
}

void bmdp_t::obtainBMDPdetailsfromMat(const char *fn) 
{
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
          this->populateBMDPSpec(*contents);
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

  } catch (const char *msg) {
    std::cerr << msg << std::endl;
    exit(0);
  }
}

// Use constructed BMDP and call synthetiser
// obtain optimal policy
void bmdp_t::runSynthesis(double eps, double iterationNum) 
{
  const char* policyType = "pessimistic";
  bool maxPolicy = true;

  //___________________________________________________________________________________________________
  // SETTING UP BMDP MODEL
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP bmdp;

  // Store results
  // If folder does not already exist then
  // create it else store in already existant
  // folder
  if(checkFolderExists("../results") == -1) {
    if(mkdir("../results", 0777) == -1) {
       std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
       exit(0);
    }
  }
  readFile("../results/bmdp.txt", bmdp);
  // check if bmpd is valid
  bmdp.isValid();
  //---------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // BMDP INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi(bmdp);

  MDP::BMDP::Policy policy;
  std::vector<double> minVals;
  std::vector<double> maxVals;

  // set the dicount rate
  ivi.setDiscount(1.);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transitin probabilties
  ivi.computePessimisticPolicy(policy, maxPolicy, minVals, minVals,
                               iterationNum, eps);

  //--------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // SETTING UP IMC
  // build an IMC (new BMDP) with the computed policy to find the lower bounds
  // for optimistic or upper bounds for pessimistic
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP imc;

  unsigned int stateNum = bmdp.getNumStates();
  std::vector<MDP::State> vecState;
  std::cout << "stateNum " << stateNum <<std::endl;
  // add states to IMC with cost 0 every where and cost 1 at the terminal states
  for (unsigned int i = 0; i < stateNum; i++) {
    vecState.push_back(MDP::State(i));
    imc.addState(&vecState[i], bmdp.getCost(i));
  }

  // add actions
  MDP::Action imcAction = 0;
  imc.addAction(&imcAction);

  // add the tranistion probabilities to imc
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) {
    std::vector<MDP::State::StateID> dest =
        bmdp.getDestinations(it->first, it->second);
    for (unsigned int i = 0; i < dest.size(); i++) {
      std::pair<double, double> probs =
          bmdp.getProbabilityInterval(it->first, it->second, dest[i]);
      imc.addTransition(it->first, 0, dest[i], probs.first, probs.second);
    }
  }

  //___________________________________________________________________________________________________
  // IMC INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi_imc(imc);

  MDP::BMDP::Policy imcPolicy;

  // set the discount rate
  ivi_imc.setDiscount(1.);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transition probabilties
  ivi_imc.computeOptimisticPolicy(imcPolicy, maxPolicy, maxVals, maxVals,
                                  iterationNum, eps);

  //--------------------------------------------------------------------------------------------------
  // store policy & bounds
  int numStates = 0;
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) {
    numStates += 1;
  }
  std::string Ns = std::to_string(numStates);
  std::vector<double> e;
  double E_q = 0;
  this->Solution = arma::mat(minVals.size(),2);
  this->Policy = arma::mat(minVals.size(),1);
  size_t count = 0;
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) {
    int mu = it->second;
    if (mu < 0) {
      mu = -mu;
    }
    this->Solution(count,0) = minVals[it->first];
    this->Solution(count,1) = maxVals[it->first];
    this->Policy(count,0) = mu;
    E_q = (maxVals[it->first] - minVals[it->first]);
    e.push_back(E_q);
    count++;
  }
  auto biggest = std::max_element(std::begin(e), std::end(e));
  this->E_max = *biggest;
}

// Use constructed BMDP and call model checker
// to obtain the "maximal optimistic" and "minimal pessimistic" policies on the
// IMDP.
void bmdp_t::runSafety(double eps, double iterationNum) 
{
  const char* policyType = "pessimistic";
  bool maxPolicy = true;

  //___________________________________________________________________________________________________
  // SETTING UP BMDP MODEL
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP bmdp;

  // Store results
  // If folder does not already exist then
  // create it else store in already existant
  // folder
  if(checkFolderExists("../results") == -1) 
  {
    if(mkdir("../results", 0777) == -1) 
    {
       std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
       exit(0);
    }
  }

  readFile("../results/bmdp.txt", bmdp);
  // check if bmpd is valid
  bmdp.isValid();
  //---------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // BMDP INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi(bmdp);

  MDP::BMDP::Policy policy;
  std::vector<double> minVals;
  std::vector<double> maxVals;

  // set the dicount rate
  ivi.setDiscount(1.);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transitin probabilties
  ivi.computePessimisticPolicy(policy, false, minVals, minVals, iterationNum,
                               eps);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transition probabilties
  ivi.computeOptimisticPolicy(policy, true, maxVals, maxVals, iterationNum,
                              eps);

  //--------------------------------------------------------------------------------------------------
  // store solution & bounds

  int numStates = 0;
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) 
  {
    numStates += 1;
  }
  std::string Ns = std::to_string(numStates);

  std::vector<double>  e;
  size_t count = 0;
  this->Solution = arma::zeros<arma::mat>(minVals.size(),2);
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) 
  {
    double sol = minVals[it->first];
    double E_q = (maxVals[it->first] - minVals[it->first]);
    if (E_q < 0) 
    {
      E_q = -E_q;
    }
    
    this->Solution(count,1) = 1 - sol;
    this->Solution(count,0) = 1 - maxVals[it->first];
    
    e.push_back(E_q);
    count++;
  }

  auto biggest = std::max_element(std::begin(e), std::end(e));
  this->E_max = *biggest;

}

arma::vec bmdp_t::getESafety(double eps, double iterationNum) 
{

  //___________________________________________________________________________________________________
  // SETTING UP BMDP MODEL
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP bmdp;

  // Store results
  // If folder does not already exist then
  // create it else store in already existant
  // folder
  if(checkFolderExists("../results") == -1) 
  {
    if(mkdir("../results", 0777) == -1) 
    {
       std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
       exit(0);
    }
  }

  readFile("../results/bmdp.txt", bmdp);

  // check if bmpd is valid
  bmdp.isValid();
  //---------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // BMDP INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi(bmdp);

  MDP::BMDP::Policy policy;
  std::vector<double> minVals;
  std::vector<double> maxVals;

  // set the dicount rate
  ivi.setDiscount(1.);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transitin probabilties
  ivi.computePessimisticPolicy(policy, false, minVals, minVals, iterationNum,
                               eps);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transition probabilties
  ivi.computeOptimisticPolicy(policy, true, maxVals, maxVals, iterationNum,
                              eps);

  //--------------------------------------------------------------------------------------------------

  std::vector<double> e_med;
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) 
  {
    e_med.push_back(maxVals[it->first] - minVals[it->first]);
  }

  arma::vec e(e_med.size());
  for (unsigned i = 0; i < e_med.size(); ++i) 
  {
    e(i) = e_med[i];
  }
  return e;
}

// Use constructed BMDP and call synthetiser
// obtain optimal policy
arma::vec bmdp_t::getESynthesis(double eps, double iterationNum) 
{
  const char* policyType = "pessimistic";
  bool maxPolicy = true;

  //___________________________________________________________________________________________________
  // SETTING UP BMDP MODEL
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP bmdp;
  // Store results
  // If folder does not already exist then
  // create it else store in already existant
  // folder
  if(checkFolderExists("../results") == -1) 
  {
    if(mkdir("../results", 0777) == -1) 
    {
       std::cerr << "Error cannot create results directory: " << std::strerror(errno) << std::endl;
       exit(0);
    }
  }

  readFile("../results/bmdp.txt", bmdp);

  // check if bmpd is valid
  bmdp.isValid();

  //---------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // BMDP INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi(bmdp);

  MDP::BMDP::Policy policy;
  std::vector<double> minVals;
  std::vector<double> maxVals;

  // set the dicount rate
  ivi.setDiscount(1.);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transitin probabilties
  ivi.computePessimisticPolicy(policy, maxPolicy, minVals, minVals,
                               iterationNum, eps);

  //--------------------------------------------------------------------------------------------------

  //___________________________________________________________________________________________________
  // SETTING UP IMC
  // build an IMC (new BMDP) with the computed policy to find the lower bounds
  // for optimistic or upper bounds for pessimistic
  //---------------------------------------------------------------------------------------------------
  MDP::BMDP imc;

  unsigned int stateNum = bmdp.getNumStates();
  std::vector<MDP::State> vecState;

  // add states to IMC with cost 0 every where and cost 1 at the terminal states
  for (unsigned int i = 0; i < stateNum; i++) 
  {
    vecState.push_back(MDP::State(i));
    imc.addState(&vecState[i], bmdp.getCost(i));
  }

  // add actions
  MDP::Action imcAction = 0;
  imc.addAction(&imcAction);

  // add the tranistion probabilities to imc
  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) 
  {
    std::vector<MDP::State::StateID> dest = bmdp.getDestinations(it->first, it->second);
    for (unsigned int i = 0; i < dest.size(); i++) 
    {
      std::pair<double, double> probs =
          bmdp.getProbabilityInterval(it->first, it->second, dest[i]);
      imc.addTransition(it->first, 0, dest[i], probs.first, probs.second);
    }
  }

  //___________________________________________________________________________________________________
  // IMC INTERVAL VALUE ITERATION
  //---------------------------------------------------------------------------------------------------
  MDP::IntervalValueIteration ivi_imc(imc);

  MDP::BMDP::Policy imcPolicy;

  // set the discount rate
  ivi_imc.setDiscount(1.);

  // compute policy.  maxVals and minVals are the upper and lower bounds of
  // transition probabilties
  ivi_imc.computeOptimisticPolicy(imcPolicy, maxPolicy, maxVals, maxVals,
                                  iterationNum, eps);

  //--------------------------------------------------------------------------------------------------

  // print policy & bounds
  std::vector<double> e_med;

  for (MDP::BMDP::Policy::iterator it = policy.begin(); it != policy.end();
       it++) 
  {
    e_med.push_back(maxVals[it->first] - minVals[it->first]);
  }

  arma::vec e(e_med.size());
  for (unsigned i = 0; i < e_med.size(); ++i) 
  {
    e(i) = e_med[i];
  }
  
  return e;
}

void bmdp_t::readSpec(const char *t_path) 
{
  std::ifstream fin(t_path);
  int actNum = 0;
  int specTH = 0;
  if (!fin)
  {
    std::cout << "Could NOT open the file" << std::endl;
    return;
  }

  // read the number of switching modes and specification time horizon

  fin >> actNum >> specTH;

  this->actNum = actNum;

  // Add synthesis specifcations

  fin.close();
}

void bmdp_t::formatOutput(double time, std::string cT) 
{

  // Get initial values
  // 1. Max abstraction errors
  double error = this->E_max;

  // 2. Number of states
  int nstates = 0;
  for (unsigned i = 0; i < this->mode.size(); ++i) 
  {
    nstates += this->mode[i].vertices.size();
  }

  std::string Ns = std::to_string(nstates);


  // Output results formatting
  std::cout << std::endl;
  std::cout << "---------------------------------------" <<std::endl;
  std::cout << "Method |  |Q| states | Time(s) | Error " << std::endl;
  std::cout << "---------------------------------------" <<std::endl;
  std::cout << "IMDP   | "<< nstates  << "     | " << time << "     | " << error << std::endl;
  std::cout << "---------------------------------------" <<std::endl;
  std::cout << std::endl;

  // Option to export to file results
  std::ofstream myfile;
  std::string exportOpt;
  std::string str("y");
  std::string str1("yes");
  std::cout << "Would you like to store the results genereated by IMDP [y- yes, n - no] " << std::endl;
  std::cin >> exportOpt;
  if ((exportOpt.compare(str) == 0) || (exportOpt.compare(str1) == 0)) 
  {
    if(checkFolderExists("../results") == -1) 
    {
      if(mkdir("../results", 0777) == -1) 
      {
        std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
        exit(0);
      }
    }

    // Store run times
    std::string f_name = "../results/IMDP_Runtime_" + Ns + "_" + cT + ".txt";
    myfile.open(f_name);
    myfile << time;
    myfile.close();

    // Store transition probability matrices
    arma::mat stepsmin = arma::conv_to<arma::mat>::from(this->Stepsmin);
    arma::mat stepsmax = arma::conv_to<arma::mat>::from(this->Stepsmax);
    std::string stepsmin_name =
      "../results/IMDP_Stepsmin_" + Ns + "_" + cT + ".txt";
      std::string stepsmax_name =
      "../results/IMDP_Stepsmax_" + Ns + "_" + cT + ".txt";

    stepsmin.save(stepsmin_name, arma::raw_ascii);
    stepsmax.save(stepsmax_name, arma::raw_ascii);

    // Storing of verification solution
    std::string sol_name = "../results/IMDP_Solution_" + Ns + "_" + cT + ".txt";
    this->Solution.save(sol_name, arma::raw_ascii);

    // Storing of policy
    if(this->Policy.n_rows > 1){
      std::string pol_name = "../results/IMDP_Policy_" + Ns + "_" + cT + ".txt";
      this->Policy.save(pol_name, arma::raw_ascii);
    }
  }
  
  // Remove bmdp.txt file used to generate policy / perform verification
  // by imdp engine
  remove( "../results/bmdp.txt" );

  // Plotting of grid
  std::string str2("Y");
  std::string str3("YES");
  std::cout << "Would you like to generate abstraction figure [y "
               "- yes, n - no] "
            << std::endl;
  std::cin >> exportOpt;
  if ((exportOpt.compare(str) == 0) || (exportOpt.compare(str1) == 0) || (exportOpt.compare(str2) == 0)|| (exportOpt.compare(str3) == 0)) {
    if(checkFolderExists("../results") == -1) {
      if(mkdir("../results", 0777) == -1) {
        std::cerr << "Error cannot create results directory: " <<std::strerror(errno) <<std::endl;
        exit(0);
       }
    }

    arma::vec res = this->Solution.col(0);
    std::vector<std::vector<double>> Sp = mat_to_std_vec(res);
    std::vector<double> minP;
    for(unsigned m =0; m < res.n_rows-1; m++) {
      minP.push_back(res(m));
    }
    createPyGrid(this->mode, this->desc.boundary, minP);
  }
}
