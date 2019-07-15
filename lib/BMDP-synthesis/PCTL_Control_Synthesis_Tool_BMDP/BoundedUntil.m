function [probVmin probVmax Qsat Qpossible indVmin indVmax Qyes Qno policy] = BoundedUntil(model, phi1, phi2, k, bound, p)

%this function returns the vector of satisfying probabilities for the
%temporal operator U^{<= k} for Interval Markov Chains.

%input:     -model: model.Stepsmax: upper bound of tran probs
%                   model.Sptesmin: lower bound of tran probs
%                   model.L: arrays where the first column is the label
%                   and the second one is the vector where the label is
%                   satisfied
%           -Phi1: state formula on the Left-Hand-Side of U-operator
%                  OR vector of initial states on the LHS of U-operator
%           -Phi2: state formula on the Right-Hand-Side of U-operator
%                  OR vector of initial states on the RHS of U-operator
%           -k: time step. For unbounded (infinite horizon) property use:
%                   - faster engine c++, use k = -1
%                   - MATLAB engine, k is the tollerance, e.g., k = 1e-4
%           -bound: is one of the following {>,>=,<=,<}
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


actNum = model.actNum;
%find Qyes and Qno if phi1 is a string
%--------------------------------------------------------------------------
if isstr(phi1)
    ind = find(phi1=='"');
    for i = 1:2:length(ind);
        ind = find(phi1=='"');
        prop = phi1(ind(1)+1:ind(2)-1);
        phi1(ind(1):ind(1)+1) = '%s';
        phi1(ind(1)+2:ind(2))= ' ';
        
        flag = 0;
        j = 1;
        while j <= size(model.L,1) & flag ==0
            if isequal(model.L{j,1},prop);
                s = sprintf('[%s]',num2str(model.L{j,2}));
                phi1=sprintf(phi1,s);
                flag = 1;
            end
            j = j+1;
        end
    end
    phi1_indV = str2num(phi1);
else
    phi1_indV = zeros(1,size(model.Stepsmax,2));
    phi1_indV(phi1) = 1;
end

if isstr(phi2)
    ind = find(phi2=='"');
    for i = 1:2:length(ind);
        ind = find(phi2=='"');
        prop = phi2(ind(1)+1:ind(2)-1);
        phi2(ind(1):ind(1)+1) = '%s';
        phi2(ind(1)+2:ind(2))= ' ';
        
        flag = 0;
        j = 1;
        while j <= size(model.L,1) & flag ==0
            if isequal(model.L{j,1},prop);
                s = sprintf('[%s]',num2str(model.L{j,2}));
                phi2=sprintf(phi2,s);
                flag = 1;
            end
            j = j+1;
        end
    end
    phi2_indV = str2num(phi2);
else
    phi2_indV = zeros(1,size(model.Stepsmax,2));
    phi2_indV(phi2) = 1;
end

Qyes = find(phi2_indV);
Qno = find(~(phi1_indV|phi2_indV));


% %--------------------------------------------------------------------------
% For infinite horrizon (k = -1)

if k == -1
    % write the bmdp into a file named: bmdpModel
    
    % cd ~/BMDP-varification/PCTL_Control_Synthesis_Tool_BMDP/
    
    stateNum = size(model.Stepsmax,2);
    filename = sprintf('%s.txt',model.name);
    fileID = fopen(filename,'w');
    fprintf(fileID,'%d \n', stateNum);  %number of states
    fprintf(fileID,'%d \n', model.actNum);  %number of actions
    fprintf(fileID,'%d \n', length(Qyes));  %number of terminal states
    for i = 1:length(Qyes)
        fprintf(fileID,'%d ', Qyes(i)-1);  %terminal states
    end
    fprintf(fileID,'\n');
    for i = 1:stateNum
        if (~ismember(i,Qno))
            for a = 1:model.actNum
                ij = find(model.Stepsmax((i-1)*actNum+a,:));
                if sum(model.Stepsmax((i-1)*actNum+a,:)) < 1
                    remain = 1 - sum(model.Stepsmax((i-1)*actNum+a,:));
                    model.Stepsmax((i-1)*actNum+a,ij(end)) = model.Stepsmax((i-1)*actNum+a,ij(end)) + remain;
                end
                for j = ij
                    fprintf(fileID,'%d %d %d %f %f', i-1, a-1, j-1, full(model.Stepsmin((i-1)*actNum+a,j)), full(model.Stepsmax((i-1)*actNum+a,j)));  %number of actions
                    if (i < stateNum || j < ij(end) || a < actNum)
                        fprintf(fileID,' \n');
                    end
                end
            end
        else
            fprintf(fileID,'%d %d %d %f %f', i-1, 0, i-1, 1.0,1.0);  %number of actions
            if i < stateNum
                fprintf(fileID,' \n');
            end
        end
    end
    fclose(fileID);
    
    if strcmp(bound,'>=') || strcmp(bound,'>') || strcmp(bound,'max')
        minmax = 'maximize pessimistic';
    else
        minmax = 'minimize optimistic';
    end
    
    %s = sprintf('./synthesis maximize pessimistic %d 0.0001 %s', k, filename);
    s = sprintf('./synthesis %s %d 0.000001 %s', minmax, k, filename);
    
    [~, output] = system(s);
    output = str2num(output);
    policy = output(:,2) + 1;
    policy(find(policy == 0)) = 1;
    indVmin= output(:,3);
    indVmax= output(:,4);
    
    % cd ~/Desktop/Dell' Laptop Files'/My' Documents'/Academics/SwitchedStochSysSynthesis/
    %
    % %-------------------------------------------------------------------------
    % To compute unbounded property using matlab - k is the tollerance for
    % value itiration, e.g., k = 1e-4
elseif k < 1
    model.Stepsmax = full(model.Stepsmax);
    model.Stepsmin = full(model.Stepsmin);
    
    indVmin = zeros(size(model.Stepsmax,2),1);
    indVmin(Qyes) = 1;
    indVmax = indVmin;
    
    Pmax = zeros(size(model.Stepsmax,1)/actNum, size(model.Stepsmax,2));
    policy = ones(size(indVmin'));
    
    VIerr = 1; % value itiration error
    
    if strcmp(bound,'>=') || strcmp(bound,'>') || strcmp(bound,'max')
        while VIerr > k
            
            indVmin_old = indVmin;
            indVmax_old = indVmax;
            
            % the best and worst MDPs
            [sortVal, indAscend] = sort(indVmin,'ascend');
            [sortVal, indDescend] = sort(indVmin,'descend');
            tic;
            for i = 1:size(model.Stepsmax,1)
                % worst MDP (lower bound)
                minMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indAscend);
                tmin = toc;
                
                % best MDP (upper bound)
                maxMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indDescend);
                tmax = toc;
            end
            tall = toc;
            ind = minMDP * indVmin;
            
            if actNum>1
                % maximize the lower bound and find the policy
                [indVmin, ctrl] = max(reshape(ind,actNum,length(ind)/actNum));
                indVmin = indVmin';
            else
                indVmin = ind;
                ctrl = policy;
            end
            
            % construct the MaxTranProb matrix According to ctrl
            Pmax = maxMDP([0:length(ctrl)-1]*actNum+ctrl,:);
            indVmax = Pmax * indVmax;
            
            % update the probs of Qyes and Qno
            indVmin(Qyes) = 1;
            indVmin(Qno) = 0;
            
            indVmax(Qyes) = 1;
            indVmax(Qno) = 0;
            
            % compute the max difference between the new and old values
            VIerr = max(abs([indVmin; indVmax] - [indVmin_old; indVmax_old]));
        end
    else
        while VIerr > k
            
            indVmin_old = indVmin;
            indVmax_old = indVmax;
            
            % the best and worst MDPs
            [sortVal, indAscend] = sort(indVmax,'ascend');
            [sortVal, indDescend] = sort(indVmax,'descend');
            tic;
            for i = 1:size(model.Stepsmax,1)
                % worst MDP (lower bound)
                minMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indAscend);
                tmin = toc;
                
                % best MDP (upper bound)
                maxMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indDescend);
                tmax = toc;
            end
            tall = toc;
            ind = maxMDP * indVmax;
            
            if actNum>1
                % maximize the lower bound and find the policy
                [indVmax, ctrl] = min(reshape(ind,actNum,length(ind)/actNum));
                indVmax = indVmax';
            else
                indVmax = ind;
                ctrl = policy;
            end
            
            % construct the MaxTranProb matrix According to ctrl
            Pmin = minMDP([0:length(ctrl)-1]*actNum+ctrl,:);
            indVmin = Pmin * indVmin;
            
            % update the probs of Qyes and Qno
            indVmin(Qyes) = 1;
            indVmin(Qno) = 0;
            
            indVmax(Qyes) = 1;
            indVmax(Qno) = 0;
            
            % compute the max difference between the new and old values
            VIerr = max(abs([indVmin; indVmax] - [indVmin_old; indVmax_old]));
        end
    end
    
    
    indVmin = full(indVmin);
    indVmax = full(indVmax);
    policy = ctrl';
    
    
    % %-------------------------------------------------------------------------
    % bounded until with k >= 1
else
    model.Stepsmax = full(model.Stepsmax);
    model.Stepsmin = full(model.Stepsmin);
    
    indVmin = zeros(size(model.Stepsmax,2),1);
    indVmin(Qyes) = 1;
    indVmax = indVmin;
    
    Pmax = zeros(size(model.Stepsmax,1)/actNum, size(model.Stepsmax,2));
    policy = ones(size(indVmin'));
    policyHist = zeros(length(indVmin),k);
    
    if strcmp(bound,'>=') || strcmp(bound,'>') || strcmp(bound,'max')
        for tk = 1:k
            
            indVmin_old = indVmin;
            
            % the best and worst MDPs
            [sortVal, indAscend] = sort(indVmin,'ascend');
            [sortVal, indDescend] = sort(indVmin,'descend');
            tic;
            for i = 1:size(model.Stepsmax,1)
                % worst MDP (lower bound)
                minMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indAscend);
                tmin = toc;
                
                % best MDP (upper bound)
                maxMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indDescend);
                tmax = toc;
            end
            tall = toc;
            ind = minMDP * indVmin;
            
            if actNum>1
                % maximize the lower bound and find the policy
                [indVmin, ctrl] = max(reshape(ind,actNum,length(ind)/actNum));
                indVmin = indVmin';
            else
                indVmin = ind;
                ctrl = policy;
            end
            
            % save the controls that become available first
            %         policy_ind = find((indVmin > 0).*(indVmin_old == 0));
            %         policy(policy_ind) = ctrl(policy_ind);
            policyHist(:,tk) = ctrl';
            
            % construct the MaxTranProb matrix According to ctrl
            Pmax = maxMDP([0:length(ctrl)-1]*actNum+ctrl,:);
            indVmax = Pmax * indVmax;
            
            
            % update the probs of Qyes and Qno
            indVmin(Qyes) = 1;
            indVmin(Qno) = 0;
            
            indVmax(Qyes) = 1;
            indVmax(Qno) = 0;
        end
    else
        for tk = 1:k
            
            indVmax_old = indVmax;
            
            % the best and worst MDPs
            [sortVal, indAscend] = sort(indVmax,'ascend');
            [sortVal, indDescend] = sort(indVmax,'descend');
            tic;
            for i = 1:size(model.Stepsmax,1)
                % worst MDP (lower bound)
                minMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indAscend);
                tmin = toc;
                
                % best MDP (upper bound)
                maxMDP(i,:) = getTrueTranProb(model.Stepsmin(i,:), model.Stepsmax(i,:), indDescend);
                tmax = toc;
            end
            tall = toc;
            ind = maxMDP * indVmax;
            
            if actNum>1
                % maximize the lower bound and find the policy
                [indVmax, ctrl] = min(reshape(ind,actNum,length(ind)/actNum));
                indVmax = indVmax';
            else
                indVmax = ind;
                ctrl = policy;
            end
            
            % save the controls that become available first
            %         policy_ind = find((indVmin > 0).*(indVmin_old == 0));
            %         policy(policy_ind) = ctrl(policy_ind);
            policyHist(:,tk) = ctrl';
            
            % construct the MaxTranProb matrix According to ctrl
            Pmin = minMDP([0:length(ctrl)-1]*actNum+ctrl,:);
            indVmin = Pmin * indVmin;
            
            
            % update the probs of Qyes and Qno
            indVmin(Qyes) = 1;
            indVmin(Qno) = 0;
            
            indVmax(Qyes) = 1;
            indVmax(Qno) = 0;
        end
    end
    
    indVmin = full(indVmin);
    indVmax = full(indVmax);
    policy = policyHist;
    
end


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

Qsat = union(intersect(indMin,indMax),Qyes);
Qpossible = setdiff(union(indMin,indMax),  union(Qsat,Qno));