%% Flat RL 4rooms %%
%% Yue Liu 2016/12/11 %%

clear all
close all
%% specify the task %%
start = [1,11];

%reward
rew = zeros(9,11,11);
rew(1,8,2) = 10000;


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
upperight = [[1,7];[1,8];[1,9];[1,10];[1,11];[2,7];[2,8];[2,9];[2,10];[2,11];[3,7];[3,8];[3,9];[3,10];[3,11];[4,7];[4,8];[4,9];[4,10];[4,11];[5,7];[5,8];[5,9];[5,10];[5,11];[6,7];[6,8];[6,9];[6,10];[6,11]];
lowerleft = [[7,1];[7,2];[7,3];[7,4];[7,5];[8,1];[8,2];[8,3];[8,4];[8,5];[9,1];[9,2];[9,3];[9,4];[9,5];[10,1];[10,2];[10,3];[10,4];[10,5];[11,1];[11,2];[11,3];[11,4];[11,5]];
lowerright = [[8,7];[8,8];[8,9];[8,10];[8,11];[9,7];[9,8];[9,9];[9,10];[9,11];[10,7];[10,8];[10,9];[10,10];[10,11];[11,7];[11,8];[11,9];[11,10];[11,11]];
gate1 = [3,6];
gate2 = [6,2];
gate3 = [10,6];
gate4 = [7,9];
goal = [8,2];
boundaries = [gate1;gate2;gate3;gate4];



%% specify global parameters %%

alpha_v = 0.005;
alpha_a = 0.001;
gamma = 0.9;
episodes =3000;
runs = 1;
rectime = zeros(runs,episodes);
vv = zeros(11); %visualize value matrix
tau = 1;



%% main loop %%
for run = 1:runs
    v = zeros(9,11,11);   %option-specific value function -- 9 options, 7*7 states
    a = zeros(9,16,11,11);	%option-specific action value function -- 9 options, 16 actions, 7*7 states

    %list of 9 options [primitive, A1, A2, B1, B4, C2, C3, D3, D4]
    %list of 16 actions [N,NE,E,SE,S,SW,W,NW,A1,A2,B1,B4,C2,C3,D3,D4]
    for ep = 1:episodes
        %% -------------------------reset---------------------------- %%
        t = 0;
        s = [1,11];	%starting state
        ss = s;
        r = 1;  %parameter indicating which action the agent takes
        o_ctrl = 1;
        
        while ~isequal(ss,goal)	%goal state
            %% ------------------update internal parameters-------------- %%
            t = t+1;
            s = ss;
            
            %% -----------------go to a new state----------------- %%
            values = zeros(1,8);
            for i = 1:8
                values(1,i) = a(1,i,s(1),s(2));
            end
            
            p = softmax(values'/tau);
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
            % if run into wall, stay
            if ismember(ss,wall,'rows')
                %disp('run into wall');
                ss = s;
            end
            
            %% -----------------update action values----------------- %%
            if ~isequal(ss,s)  % if the agent does not move (run into wall, then do not update)
                
                delta = rew(1,ss(1),ss(2)) + gamma*v(1,ss(1),ss(2)) - v(1,s(1),s(2));   %gamma at wrong place before!
                v(1,s(1),s(2)) = v(1,s(1),s(2)) + alpha_v * delta;
                a(1,r,s(1),s(2)) = a(1,r,s(1),s(2)) + alpha_a * delta;
                
            end
        end
        %% -----------record time steps taken--------------- %%
        disp([ep,t]);
        rectime(run,ep) = t;
    end
    vvv(:,:) = v(1,:,:);
    vv = vv+vvv;
end
figure;
subplot(1,2,1);
title('learning curve, flatRL')
plot((rectime)/runs);
xlabel('episodes');
ylabel('time steps taken');

subplot(1,2,2);
title('state value function, flatRL')
imagesc(vv/runs);
colorbar;

