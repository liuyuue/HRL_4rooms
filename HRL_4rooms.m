%%%Hierarchical Reinforcement Learning for 4-room problem (larger version)%%%
%%%Yue Liu 2016/12/09%%%

clear all
close all

%% specify the task %%
start = [1,11];

%reward
rew = zeros(9,11,11);
rew(1,8,2) = 10000;
%pseudo reward
rew(2,3,6) = 100;
rew(3,6,2) = 100;
rew(4,3,6) = 100;
rew(5,7,9) = 100;
rew(6,6,2) = 100;
rew(7,10,6) = 100;
rew(8,10,6) = 100;
rew(9,7,9) = 100;


%%the grid world%%
wall = [[1,6];[2,6];[4,6];[5,6];[6,6];[7,6];[8,6];[9,6];[11,6];...
    [6,1];[6,3];[6,4];[6,5];...
    [7,7];[7,8];[7,10];[7,11];...
    [0,0];[0,1];[0,2];[0,3];[0,4];[0,5];[0,6];[0,7];[0,8];[0,9];[0,10];[0,11];[0,12];...
    [1,0];[2,0];[3,0];[4,0];[5,0];[6,0];[7,0];[8,0];[9,0];[10,0];[11,0];[12,0];...
    [12,1];[12,2];[12,3];[12,4];[12,5];[12,6];[12,7];[12,8];[12,9];[12,10];[12,11];[12,12];...
    [1,12];[2,12];[3,12];[4,12];[5,12];[6,12];[7,12];[8,12];[9,12];[10,12];[11,12]];
%upperleft = [[1,1];[1,2];[1,3];[1,4];[1,5];[2,1];[2,2];[2,3];[2,4];[2.5];[3,1];[3,2];[3,3];[3,4];[3,5];[4,1];[4,2];[4,3];[4,4];[4,5];[5,1];[5,2];[5,3];[5,4];[5,5]];
%upperleft = [[1,1];[1,2];[1,3];[1,4];[1,5];[2,1];[2,2];[2,3];[2,4];[2.5];[3,1];[3,2];[3,3];[3,4];[3,5];[4,1];[4,2];[4,3];[4,4];[4,5];[5,1];[5,2];[5,3];[5,4];[5,5]];
upperleft = [[1,1];[1,2];[1,3];[1,4];[1,5];[2,1];[2,2];[2,3];[2,4];[2,5];[3,1];[3,2];[3,3];[3,4];[3,5];[4,1];[4,2];[4,3];[4,4];[4,5];[5,1];[5,2];[5,3];[5,4];[5,5]];
upperright = [[1,7];[1,8];[1,9];[1,10];[1,11];[2,7];[2,8];[2,9];[2,10];[2,11];[3,7];[3,8];[3,9];[3,10];[3,11];[4,7];[4,8];[4,9];[4,10];[4,11];[5,7];[5,8];[5,9];[5,10];[5,11];[6,7];[6,8];[6,9];[6,10];[6,11]];
lowerleft = [[7,1];[7,2];[7,3];[7,4];[7,5];[8,1];[8,2];[8,3];[8,4];[8,5];[9,1];[9,2];[9,3];[9,4];[9,5];[10,1];[10,2];[10,3];[10,4];[10,5];[11,1];[11,2];[11,3];[11,4];[11,5]];
lowerright = [[8,7];[8,8];[8,9];[8,10];[8,11];[9,7];[9,8];[9,9];[9,10];[9,11];[10,7];[10,8];[10,9];[10,10];[10,11];[11,7];[11,8];[11,9];[11,10];[11,11]];
gate1 = [3,6];
gate2 = [6,2];
gate3 = [10,6];
gate4 = [7,9];
goal = [8,2];
boundaries = [gate1;gate2;gate3;gate4];


%% specify global parameters %%

alpha_v = 0.1;
alpha_a = 0.01;
gamma = 0.9;
episodes = 9000;
tau = 1; %temperature
runs = 1;
rectime = zeros(runs,200);
rectime2 = zeros(1,200);    %for options
rectime3 = zeros(1,200);
rectime4 = zeros(1,200);
rectime5 = zeros(1,200);
rectime6 = zeros(1,200);
rectime7 = zeros(1,200);
rectime8 = zeros(1,200);
rectime9 = zeros(1,200);

vv = zeros(9,11,11);


%% main loop %%


v = zeros(9,11,11);   %option-specific value function -- 9 options, 7*7 states
a = zeros(9,16,11,11);	%option-specific action value function -- 9 options, 16 actions, 7*7 states

%list of 9 options [primitive, A1, A2, B1, B4, C2, C3, D3, D4]
%list of 16 actions [N,NE,E,SE,S,SW,W,NW,A1,A2,B1,B4,C2,C3,D3,D4]




%% train options %%

for ep = 1:episodes/9   %train the 1st option (A1)
    ep
    %%%---reset---%%%
    o_ctrl = 2;
    t = 0;
    t2 = 0;
    s = datasample(upperleft,1);	%starting state
    
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    memory_s = zeros(1,2);
    memory_a = 0;
    r_cum = 0;
    
    t_option = 0;
    
    
    while ~isequal(ss,gate1)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate2],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            %disp([s(1),s(2),v(o_ctrl,s(1),s(2)), delta_op])    %for diagnosing
            
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime2(1,ep) = t;
end



for ep = episodes/9+1 : episodes*2/9   %train the 2nd option (A2)
    ep
    %%%---reset---%%%
    o_ctrl = 3;
    t = 0;
    t2 = 0;
    s = datasample(upperleft,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    memory_s = zeros(1,2);
    memory_a = 0;
    r_cum = 0;
    
    t_option = 0;
    
    
    while ~isequal(ss,gate2)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate1],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime3(ep-episodes/9) = t;
end

for ep = episodes*2/9+1:episodes*3/9   %train the 3rd option (B1)
    ep
    %%%---reset---%%%
    o_ctrl = 4;
    t = 0;
    t2 = 0;
    s = datasample(upperright,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    
    
    while ~isequal(ss,gate1)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate4],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime4(1,ep-episodes/9*2) = t;
end

for ep = episodes*3/9+1:episodes*4/9   %train the 4th option (B4)
    ep
    %%%---reset---%%%
    o_ctrl = 5;
    t = 0;
    t2 = 0;
    s = datasample(upperright,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    while ~isequal(ss,gate4)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate1],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime5(1,ep-episodes/9*3) = t;
end

for ep = episodes*4/9+1:episodes*5/9   %train the 5th option (C2)
    ep
    %%%---reset---%%%
    o_ctrl = 6;
    t = 0;
    t2 = 0;
    s = datasample(lowerleft,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    
    while ~isequal(ss,gate2)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate3],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime6(1,ep-episodes/9*4) = t;
end

for ep = episodes*5/9+1:episodes*6/9   %train the 6th option (C3)
    ep
    %%%---reset---%%%
    o_ctrl = 7;
    t = 0;
    t2 = 0;
    s = datasample(lowerleft,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    
    while ~isequal(ss,gate3)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate2],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime7(1,ep-episodes/9*5) = t;
end

for ep = episodes*6/9+1:episodes*7/9   %train the 7th option (D3)
    ep
    %%%---reset---%%%
    o_ctrl = 8;
    t = 0;
    t2 = 0;
    s = datasample(lowerright,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    memory_s = zeros(1,2);
    memory_a = 0;
    r_cum = 0;
    
    t_option = 0;
    
    
    while ~isequal(ss,gate3)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate4],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime8(1,ep-episodes/9*6) = t;
end

for ep = episodes*7/9+1:episodes*8/9   %train the 8th option (D4)
    ep
    %%%---reset---%%%
    o_ctrl = 9;
    t = 0;
    t2 = 0;
    s = datasample(lowerright,1);	%starting state
    ss = s;
    r = 1;  %parameter indicating which action the agent takes
    
    while ~isequal(ss,gate4)	%subgoal state
        %--update--%
        t = t+1;
        
        s = ss;  %update state
        
        
        values = zeros(1,8);
        for i = 1:8
            values(1,i) = a(o_ctrl,i,s(1),s(2));
        end
        p = softmax((values/tau)');
        r = randsample(1:numel(p),1,true,p);
        
        %move to a new state%
        if r == 1
            ss = s + [-1,0];
        elseif r == 2
            ss = s + [-1,1];
        elseif r == 3
            ss = s + [0,1];
        elseif r == 4
            ss = s + [1,1];
        elseif r == 5
            ss = s + [1,0];
        elseif r == 6
            ss = s + [1,-1];
        elseif r == 7
            ss = s + [0,-1];
        elseif r == 8
            ss = s + [-1,-1];
        end
        % if run into wall, stay
        if ismember(ss,[wall;gate3],'rows')
            ss = s;
        end
        
        %update option-specific value function%
        if norm(ss-s)~=0
            delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
            v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
            a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
        end
    end
    rectime9(1,ep-episodes/9*7) = t;
end


figure; %show the value function after pre-training
for i = 1:9
    subplot(3,3,i)
    imagesc(reshape(v(i,:,:),11,11)/runs);
    colorbar;
    title(sprintf('option value function for option %d',i));
end
v_trained = v;  %record the value functions after pre-training
a_trained = a;

%% actual simulation with options trained %%
for run = 1:runs
    v = v_trained;
    a = a_trained;
    
    for ep = episodes*8/9+1:episodes*8/9+2000
        for i = 1:9
            rew(i,8,2) = 10000;
        end
        
        % for diagnosing
        %if ep == episodes*8/9+2
        %  break
        %end
        
        %%%---reset---%%%
        alpha_v = 0.005;
        alpha_a = 0.001;
        t = 0;
        t2 = 0;
        s = start;	%starting state
        ss = start;
        r = 1;  %parameter indicating which action the agent takes
        o_ctrl = 1;
        o_ctrl_next = 1;	%starts off under primitive option
        
        memory_s = zeros(1,2);  %to memorize the state before entering option
        memory_a = 0;
        r_cum = 0;
        
        t_option = 0;
        
        while ~isequal(ss,goal)	%goal state
            %--update--%
            t = t+1;
            
            % for diagnosing
            if t>20000 && t<20100
                disp([s(1),s(2),o_ctrl]);
            end
            
            
            %(optional) track the time under the control of abstract option
            if o_ctrl ~= 1
                t_option = t_option+1;
            end
            
            o_ctrl = o_ctrl_next;
            s = ss;
            
            
            %%--for each current option, calculate probability for each action and move to next state (this can be packed into a function)--%%
            %1. current option is primitive option
            if o_ctrl == 1
                %upper-left room
                if ismember(s,upperleft,'rows')
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,9,s(1),s(2));
                    values(1,10) = a(1,10,s(1),s(2));
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 2;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 3;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        %disp('run into wall');
                        ss = s;
                    end
                end
                
                %upper-right room
                if ismember(s,upperright,'rows')
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,11,s(1),s(2));
                    values(1,10) = a(1,12,s(1),s(2));
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 4;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 5;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                end
                
                %lower-left room
                if ismember(s,lowerleft,'rows')
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,13,s(1),s(2));
                    values(1,10) = a(1,14,s(1),s(2));
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 6;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 7;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                end
                
                %lower-right room
                if ismember(s,lowerright,'rows')
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,15,s(1),s(2));
                    values(1,10) = a(1,16,s(1),s(2));
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 8;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 9;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                end
                
                
                
                %boundary cases, where the agent can adopt two options from
                %either room
                
                %boundary door 1
                if isequal(s,gate1)
                    
                    values = zeros(1,10);
                    
                    
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,10,s(1),s(2));
                    values(1,10) = a(1,12,s(1),s(2));   %can adopt options A2 and B4
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 3;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 5;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                    %}
                    
                end
                
                %boundary door 2
                if isequal(s,gate2)
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,9,s(1),s(2));
                    values(1,10) = a(1,14,s(1),s(2));   %can adopt options A1(#1) and C3(#6)
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 2;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 7;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                    %}
                    
                end
                
                %boundary door 3
                if isequal(s,gate3)
                    
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,13,s(1),s(2));
                    values(1,10) = a(1,16,s(1),s(2));   %can adopt options C2(#5) and D4(#8)
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 6;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 9;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                    %}
                    
                end
                
                %boundary door 4
                if isequal(s,gate4)
                    
                    values = zeros(1,10);
                    for i = 1:8
                        values(1,i) = a(1,i,s(1),s(2));
                    end
                    values(1,9) = a(1,11,s(1),s(2));
                    values(1,10) = a(1,15,s(1),s(2));   %can adopt options B1(#3) and D3(#7)
                    p = softmax((values/tau)');
                    r = randsample(1:numel(p),1,true,p);
                    
                    if r == 1
                        ss = s + [-1,0];
                    elseif r == 2
                        ss = s + [-1,1];
                    elseif r == 3
                        ss = s + [0,1];
                    elseif r == 4
                        ss = s + [1,1];
                    elseif r == 5
                        ss = s + [1,0];
                    elseif r == 6
                        ss = s + [1,-1];
                    elseif r == 7
                        ss = s + [0,-1];
                    elseif r == 8
                        ss = s + [-1,-1];
                    elseif r == 9
                        ss = s;
                        o_ctrl_next = 4;
                    elseif r == 10
                        ss = s;
                        o_ctrl_next = 8;
                    end
                    % if run into wall, stay
                    if ismember(ss,wall,'rows')
                        ss = s;
                    end
                    %}
                end
            end
            
            
            
            
            %2. current option is abstract option (for simplicity, assume cannot change abstract option during)
            if o_ctrl ~= 1
                values = zeros(1,8);
                for i = 1:8
                    values(1,i) = a(o_ctrl,i,s(1),s(2));
                end
                p = softmax((values/tau)');
                r = randsample(1:numel(p),1,true,p);
                
                if r == 1
                    ss = s + [-1,0];
                elseif r == 2
                    ss = s + [-1,1];
                elseif r == 3
                    ss = s + [0,1];
                elseif r == 4
                    ss = s + [1,1];
                elseif r == 5
                    ss = s + [1,0];
                elseif r == 6
                    ss = s + [1,-1];
                elseif r == 7
                    ss = s + [0,-1];
                elseif r == 8
                    ss = s + [-1,-1];
                end
                
                % if run into wall, stay.
                if ismember(ss,wall,'rows')
                    ss = s;
                end
                % if go to sub goal, exit option.
                if ismember(ss,boundaries,'rows')
                    o_ctrl_next = 1;
                end
                
            end
            
            
            
            %%--update stuff after moving to a new state--%%
            if o_ctrl == 1
                if o_ctrl_next == 1 && norm(ss-s)~=0     %if the agent does not move (run into wall), then do not update, if not then do one step TD learning
                    delta = rew(1,ss(1),ss(2)) + gamma*v(1,ss(1),ss(2)) - v(1,s(1),s(2));
                    v(1,s(1),s(2)) = v(1,s(1),s(2)) + alpha_v * delta;
                    a(1,r,s(1),s(2)) = a(1,r,s(1),s(2)) + alpha_a * delta;
                elseif o_ctrl_next~=1
                    %does not move, switch to abstract option
                    t2 = 0;
                    
                    memory_s = s;
                    memory_a = o_ctrl_next+7;   %convert between the index for option and for action
                    %disp('memory updated');
                    
                end
            end
            
            if o_ctrl ~= 1
                t2 = t2 + 1;
                if ~isequal(ss,s)
                    r_cum = r_cum + (gamma^t2)*rew(o_ctrl, ss(1),ss(2));
                    delta_op = rew(o_ctrl, ss(1),ss(2)) + gamma*v(o_ctrl,ss(1),ss(2)) - v(o_ctrl,s(1),s(2));
                    v(o_ctrl,s(1),s(2)) = v(o_ctrl,s(1),s(2)) + alpha_v * delta_op;
                    
                    a(o_ctrl,r,s(1),s(2)) = a(o_ctrl,r,s(1),s(2)) + alpha_a * delta_op;
                    %disp([o_ctrl,s(1),s(2)])
                end
                
                if (o_ctrl == 2 && isequal(ss,gate1)) || (o_ctrl == 3 && isequal(ss,gate2)) || (o_ctrl == 4 && isequal(ss,gate1)) || (o_ctrl == 5 && isequal(ss,gate4)) || (o_ctrl == 6 && isequal(ss,gate2)) || (o_ctrl == 7 && isequal(ss,gate3)) || (o_ctrl == 8 && isequal(ss,gate3)) || (o_ctrl == 9 && isequal(ss,gate4))
                    
                    %terminate, update the root value function
                    delta = r_cum + (gamma^t2)*v(1,ss(1),ss(2))-v(1,memory_s(1),memory_s(2));
                    v(1,memory_s(1),memory_s(2)) = v(1,memory_s(1),memory_s(2)) + alpha_v*(delta);
                    a(1,memory_a, memory_s(1),memory_s(2)) = a(1,memory_a,memory_s(1),memory_s(2)) + alpha_a*(delta);
                    r_cum = 0;  %reset cumulative reward
                    
                end
            end
        end
        disp([ep,t]);
        rectime(run,ep-episodes/9*8) = t;
    end
    for i = 1:9
        vv(i,:,:) = vv(i,:,:)+v(i,:,:);
    end
    
end

figure;
subplot(1,2,1);
hold on
plot((rectime2)/1);
plot((rectime3)/1);
plot((rectime4)/1);
plot((rectime5)/1);
plot((rectime6)/1);
plot((rectime7)/1);
plot((rectime8)/1);
plot((rectime9)/1);

subplot(1,2,2);
plot(sum(rectime)/runs);
xlim([0,2000]);
ylim([0,1400]);


figure;
for i = 1:9
    subplot(3,3,i)
    imagesc(reshape(vv(i,:,:),11,11)/runs);
    colorbar;
    title(sprintf('option value function for option %d',i));
end


