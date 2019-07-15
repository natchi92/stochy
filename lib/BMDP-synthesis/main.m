%this is an example file to construct a BMDP model and compute for
%properties with (bounded/unbounded) UNTIL operator
addpath('PCTL_Control_Synthesis_Tool_BMDP/')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODEL CONSTRUCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% name your model
model.name = 'BMDPexample';

% number of actions per state (we assume the same number of actions per
% state)
model.actNum = 2;

% Steps matrix for lower transition prob bounds
% each row corresponds to a state-action pair
model.Stepsmin = [
    0         0.0500         0         0;
    0         0.0500         0         0;
    0         0.1200    0.1500    0.5700;
    0         0         0.5000    0.4400;
    0         0         1.0000         0;
    0.9800    0         0              0;
    0         0         0         1.0000;
    0         0         0         1.0000];

% Steps matrix for upper transition prob bounds
% each row corresponds to a state-action pair
model.Stepsmax = [
    0.0500    1.0000         0         0;
    0.0500    1.0000         0         0;
    0         0.2300    0.2000    0.6200;
    0         0         0.5600    0.5000;
    0         0         1.0000         0;
    1.0000    0         0.0500         0;
    0         0         0         1.0000;
    0         0         0        1.0000];

% labels of the states
% model.L: arrays where the first column is the label 
% and the second column is a of vector 0s and 1s, where 1 indicates the 
% label is satisfied at the corresponding state
model.L{1,1} = 'init';
model.L{1,2} = [1     0     0     0];

model.L{2,1} = 'R2';
model.L{2,2} = [0     0     1     0];

model.L{3,1} = 'R3';
model.L{3,2} = [0     0     0     1];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SYTHESIS / MODEL CHECKING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Property form: P_{~p} [phi1 U(<=k) phi2]
% - for bounded time properties, use k as the time step
% - for unbounded time step (infinite horrizon):
%           - MATLAB engine: k is value itiration tollerance, e.g., k = 1e-4
%           - C++ engine: use k = -1

phi1 = '~"R3"';
phi2 = '"R2"';
k = 5;
p = 0;

[probVmin probVmax Qsat Qpossible indVmin indVmax Qyes Qno policy] = BoundedUntil(model, phi1, phi2, k, '>=', .5)

%output:    -probVmax: 2-column matrix. 1st column is the state number of
%                   satisfying states and the second column is their
%                   probabilities (upper bound) using the upper bound
%                   probability matrix.
%
%           -probVmin: 2-column matrix. 1st column is the state number of
%                   satisfying states and the second column is their
%                   probabilities (lower bound) using the lower bound
%                   probability matrix.
%
%           -Qsat: the set of satisfying states
%
%           -Qpossible: the set of states (state numbers) that have the
%           possiblity of satisfying the formula but not definite
%
%           -indVmin: the vector of minimum probabilities
%
%           -indVmax: the vector of maximum probabilities
%
%           - Qyes: the set of states that satisfy the property with
%           probability 1.
%
%           - Qno: the set of states that satisfy the property with
%           probability 0.
%
%           -policy: column vector with the index of the optimal action
%



