function p = getTrueTranProb(pmin, pmax, indSorted)

% computes the true distribution given a range of distributions

% input:    pmin: lower bound transition probabilities
%           pmax: upper bound transition probabilities
%           indSorted: index of states according to their sorted list
% output:
%           p: true transition probability

p = zeros(1,size(pmax,2));
used = sum(pmin);
remain = 1 - used;

for i = indSorted'
    if pmax(i) <= (remain + pmin(i))
        p(i) = pmax(i);
    else
        p(i) = pmin(i) + remain;
    end
    remain = max([0, remain - (pmax(i) - pmin(i)) ]);    
end