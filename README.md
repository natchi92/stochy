# Stochastic Hybrid Systems tool : StocHy

Stochastic hybrid systems (shs) are a rich mathematical framework which capture the probabilistic relationship between discrete and continuous variables and their evolution in time.

StocHy is a C++ tool aimed at the modelling of stochastic hrybid systems  together with their analysis, which takes the form of formal verification and control synthesis.
It targets the wider adoption of shs both in academia and in industry.

The tools allows to described discrete time shs by parsing well known state space models and allows for performing any of the following three tasks:

    (1) simulation of stochastic processes (serving as a visualisation tool of shs)
    (2) formal verification via abstractions taking the form of Markov processes and their variants of interval Markov decision processes.
    (3) strategy synthesis for e.g. selection of control actions to maximise satisfiability of a certain property.

# Installation, documentation, and examples

# Development version 
  (1)  Run: git submodule update --init --recursive
  (2)  Obtain required dependencies by running get_dep.dev.sh 
  (3)  Use build_debug.sh to run StocHy in development mode
  

# Release version
  (1)  Obtain required dependencies by running get_dep.dist.sh 
  (2)  Use build_release.sh to run StocHy 
  
 Unless opting to extend or connect StocHy with your own tools, it is advised to opt to install the release version. 

# Compiled versions
All the built and compiled binary files are then found within /build/bin. One can then run the corresponding binary files by using ./stochy_XX where XX is the corresponding model name.

# Docker system

To facilitate sharing of StocHy between different operating environments we provide a docker container containing StocHy. Details are found within the Docker folder.

NOTE: this is currently no longer maintained
# Wiki

We maintain a wiki with all the installation details and running examples on how to use StocHy. This can be found at: [Welcome](https://gitlab.com/natchi92/StocHy/wikis/home)

# Examples

We provide four examples as part of StocHy. For each example we have a dedicated wiki page illustrating the examples and how to run them. The details are also found within the tool paper which can be found within this repository.

[Example 1](https://gitlab.com/natchi92/StocHy/wikis/Example-1:-Formal-Verification) Formal verification via IMDP or MDP method

[Example 2](https://gitlab.com/natchi92/StocHy/wikis/Example-2:-Strategy-Synthesis) Strategy synthesis for obstacle avoidance

[Example 3](https://gitlab.com/natchi92/StocHy/wikis/Example-3-:-Scaling-in-dimensions) Scaling in continuous dimensions

[Example 4](https://gitlab.com/natchi92/StocHy/wikis/example-4:-simulation) Simulation of shs

# Running your own models

Simply create your own Case_study.cpp file within /src/case_studies/. This follows the same structure as described within the TACAS paper, however we don't need to modify the main file each time. We now create an individual case study and call the case study we want to run. 

# Connecting with the individual libraries 
All the src files for the individual libraries can be found within src/ . These are built as individual libraries and one can simply use any of the libraries as needed for further extensions.

    (1) shs - contains the general model description for constructing a shs based on the input model structure; together with the simulator for shs.
    (2) FAUST - library for performing abstractions via MDPs
    (3) bmdp - library for performing abstractions via IMDPs 

# Benchmark for stochastic processes

We also provide a set of benchmarks for cyber-physical systems endowed with stochasticity (noise). These benchmarks serve as a means of constructing further models and test different verification/ synthesis
algorithms against models with different complexities.

The benchmarks can be found [here](https://gitlab.com/natchi92/BASBenchmarks), while the accompanying research paper describing these benchmarks can found is given [here](https://gitlab.com/natchi92/BASBenchmarks/blob/master/bench_ADHS.pdf).

# Contributions

StocHy is an open-source tool. We welcome feedback, issues, bug-fixes, extensions to other existing tools and other enhancements.

# Licence

The Stochastic Hybrid Systems tool : StocHy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation v3.0.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this toolbox (see LICENSE). If not, see https://www.gnu.org/licenses/.

It is the user's responsibility in assessing the correctness of the theory and software implementation before putting it to use in their own research or exploiting the results commercially. We are, however, very happy to answer any questions and investigate any bug reports.


# Credits

The principal author of this tool is Nathalie Cauchi (also primary maintainer).
However, it is a malgamation of collaboration with Kurt Degiorgio, Sadegh Soudjani (author of [FAUST^2](https://scholar.googleusercontent.com/scholar.bib?q=info:0oaUVF6-PBsJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAW-lG7SwJmkvp8LC2w3lA3JNYsi1S1AtU&scisf=4&ct=citation&cd=-1&hl=en) which inspired the MDP method in this work),  Morteza Lahijanian ([IMDP](http://sites.bu.edu/hyness/files/2015/08/TAC-Morteza-Stoch-2015.pdf) method), Luca Laurenti  ([IMDP](https://scholar.googleusercontent.com/scholar.bib?q=info:YqluvllRDOQJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAW-lG0s8rwD2JZuh8sD8Z6c92F-OfSbSO&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1) method), Sofie Haesaert ([Strategy synthesis via FAUST](http://www.cs.ox.ac.uk/publications/publication11228.bib)).
Please cite their relevant papers when using StocHy.


The author is a DPhil student of Prof. Alessandro Abate within the group [OxCAV](https://www.oxcav.com/).


A tool paper describing the features of StocHy will be presented at the 25th International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS). A copy of the submission can also be found within the repository.


A video demo of StocHy can be found at [OxCAV](https://www.oxcav.com/resources)
