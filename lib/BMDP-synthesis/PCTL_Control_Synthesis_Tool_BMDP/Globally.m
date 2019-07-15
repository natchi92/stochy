function [probVmin probVmax Qsat Qpossible indVmin indVmax Qyes Qno policy] = Globally(model, phi, k, bound, p)

%this function returns the vector of satisfying probabilities for the
%temporal operator G^{<= k} for Interval Markov Chains.

%input:     -model: model.Stepsmax: upper bound of tran probs
%                   model.Sptesmin: lower bound of tran probs
%                   model.L: arrays where the first column is the label
%                   and the second one is the vector where the label is
%                   satisfied
%           -Phi:   state formula on the Right-Hand-Side of G-operator
%                   OR vector of initial states on the RHS of U-operator
%           -k: time step. For unbounded (infinite horizon) property use:
%                   - faster engine c++, use k = -1
%                   - MATLAB engine, k is the tollerance, e.g., k = 1e-4
%           -bound: is one of the following {>,>=,<=,<,max,min}
%           -p: if bound is {>,>=,<=,<} then the bound value p is needed
%
%output:    -probVmax: 2-column matrix. 1st column is the state number of
%                   satisfying states and the second column is their
%                   probabilities (upper bound) using the upper bound
%                   probability matrix.
%           -probVmin: 2-column matrix. 1st column is the state number of
%                   satisfying states and the second column is their
%                   probabilities (lower bound) using the lower bound
%                   probability matrix.
%           -policy: 2-colum matrix. 1st column is the state number and
%           the 2nd column is the optimal control
%           -Qsat: the set of satisfying states
%           -Qpossible: the set of states (state numbers) that have the
%           possiblity of satisfying the formula but not definite

% P>=p [ G phi ] = P<= (1-p) [ true U !phi ];

phi1 = model.states;
phi2 = sprintf('~%s',phi);

% flip the bound
if strcmp(bound,'>=') 
    boundn = '<=';
elseif strcmp(bound,'>') 
    boundn = '<';
elseif strcmp(bound,'max')
    boundn = 'min';
elseif strcmp(bound,'<=') 
    boundn = '>=';
elseif strcmp(bound,'<') 
    boundn = '>';
elseif strcmp(bound,'min')
    boundn = 'max';
else
    pfritf('*************EEERRROOOOORRRRR*************')
    pfritf('problem with bound')
    pfritf('*************EEERRROOOOORRRRR*************')
    return
end

% p = 1-p;

[probVminU probVmaxU QsatU QpossibleU indVminU indVmaxU QyesU QnoU policy] = BoundedUntil(model, phi1, phi2, k, boundn, p);

indVmin = 1 - indVmaxU;
indVmax = 1 - indVminU;


% compare the computed probs with the prob bound
if strcmp(bound,'>')
    indMin = find(indVmin > p);
    indMax = find(indVmax > p);
elseif strcmp(bound,'>=')
    indMin = find(indVmin >= p);
    indMax = find(indVmax >= p);
elseif strcmp(bound,'<')
    indMin = find(indVmin < p);
    indMax = find(indVmax < p);
elseif strcmp(bound,'<=')
    indMin = find(indVmin <= p);
    indMax = find(indVmax <= p);
end

probVmin = [indMin indVmin(indMin)];
probVmax = [indMax indVmax(indMax)];

Q = 1:length(probVmin);
Qno = QyesU;
Qyes = QnoU;
Qsat = union(intersect(indMin,indMax),Qyes);
Qpossible = setdiff(union(indMin,indMax),  union(Qsat,Qno));
    
