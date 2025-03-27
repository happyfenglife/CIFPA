% ----------------------------------------------------------------------- %
% Programmed by Dahai Li & Yunfeng Zhu                                    %
% ----------------------------------------------------------------------- %
function [bestFitValue, bestSolution, cg_curve] = CIFPA2(N, N_iter, lb, ub, D, fobj)
P = 0.5;
Po = 0.6;

LB = lb.*ones(1, D); % Lower bounds
UB = ub.*ones(1, D); % Upper bounds

Sol = initialization(N, D, UB, LB);
Fitness = zeros(1, N);

% 记录初始当前最佳个体位置以及其适应值
pbest = Sol(1,:);
Fitness(1) = fobj(Sol(1, :));
pmin = Fitness(1);

for i = 2:N
  Fitness(i) = fobj(Sol(i, :));
  
  if Fitness(i) < pmin
    pbest = Sol(i,:);
    pmin = Fitness(i);
  end
end

% 记录初始最佳个体位置以及其适应值和当前最佳相同
gbest = pbest;
gmin = pmin;

S_Qopposed = zeros(N, D);
S_QOTLBO = zeros(N, D);
cg_curve = zeros(1, N_iter);

for t = 1:N_iter
  % divide population
  [~, sortedIdxF] = sort(Fitness); % 按适应度值升序排序
  Fi = zeros(size(Fitness));
  Fi(sortedIdxF) = 1:length(Fitness); % 适应度排名
  
  % 计算相对距离排名 Qi
  L = zeros(N, 1);
  firstIdx = 1;
  for i = 1:N
    if Fi(i) == 1
      firstIdx = i;
      continue
    end
    
    dis = inf;
    for j = 1:N
      if Fi(j) < Fi(i)
        dis = dis + sqrt(sum((Sol(i, :) - Sol(j, :)).^2));
        % dis = min(dis, sqrt(sum((Sol(i, :) - Sol(j, :)).^2)));
      end
    end
    
    L(i) = mean(dis); % 取平均值作为相对距离
    % L(i) = min(dis); % 取最优值作为相对距离
  end
  L(firstIdx) = max(L); % 适应度值最优的个体 拥有最大距离
  
  [~, sortedIdxM] = sort(L, 'descend'); %
  Qi = zeros(size(Fitness));
  Qi(sortedIdxM) = 1:length(L); % 相对距离排名
  
  % 计算综合权重指标 Ei
  Ei = Fi + Qi;
  [~, sortedIdx] = sort(Ei);
  % Ei(sortedIdx) = 1:length(Ei);
  Sol(1:N,:) = Sol(sortedIdx(1:N),:);
  Fitness(1:N) = Fitness(sortedIdx(1:N));
  
  
  
  
  
  
  
  
  
  
  % Nc = 0.2 + 0.6 * (1 - exp(-2 * t / N_iter));
  % Ne = 1 - Nc;
  % eCount = ceil(Ne * N);
  eCount = N*P;
  
  for i = 1:N
    if i <= eCount
      % 精英花粉个体 -> 全局搜索
      L1 = Levy_Flight(D);
      % L1 = Brown(D);
      % L1 = LnF3(2/(1+sqrt(5)),0.05,1,1);
      L2 = Levy_Flight(D);
      % L2 = Brown(D);
      % L2 = LnF3(2/(1+sqrt(5)),0.05,1,1);
      % 增加了趋向当前最优的分量进行全局搜索
      S = Sol(i,:) + L1.*(gbest - Sol(i,:)) + L2.*(pbest - Sol(i,:));
    else
      % 普通花粉个体 -> 局部搜索
      c1 = rand(1,D);
      c2 = rand(1,D);
      c3 = rand(1,D);
      g = randperm(N,2);
      
      teta = rand*pi/2.1;  % 正切算法参数
      step = 10*sign(rand-0.5)*norm(gbest)*log(1+10*D/N_iter); % 正切算法参数
      
      % 增加了趋向全局最优和当前最优的分量进行局部搜索;
      S = Sol(i,:) ...
        + c1.*(Sol(g(1), :) - Sol(g(2), :)) ...
        + c2.*(gbest - Sol(i, :)) + c3.*(pbest - Sol(i, :));
      
      % S = Sol(i,:) ...
      %   + c1.*(Sol(g(1), :) - Sol(i, :)) ...
      %   + c2.*(Sol(g(2), :) - Sol(i, :));
      
      % S = gbest + step*tan(teta).*(gbest - Sol(i,:)); % 对精英个体在正切范围内搜索
    end
    
    %     Sol(i,:) = simplebounds(S,Lb,Ub);
    %     Fitness(i) = Fun(Sol(i,:));
    S = simplebounds(S, LB, UB);
    Fnew = fobj(S);
    if Fnew <= Fitness(i)
      Sol(i,:) = S;
      Fitness(i) = Fnew;
    end
  end
  
  
  
  
  
  
  
  
  
  % 基于加权中心的反向学习方法，先计算各个花粉的权值；
  Gcenter = zeros(1, D);
  worstF = max(Fitness);
  sum_f = sum(Fitness);
  for i = 1:N
    Gcenter = Gcenter + ((worstF - Fitness(i)) / (N * worstF - sum_f)) * Sol(i, :);
  end
  
  % Gcenter = (UB + LB) / 2;
  % % 计算所有花粉个体的反向解；
  % for i = 1:N
  %   for j = 1:D
  %     S_Qopposed(i, j) = 2 * Gcenter(j) - Sol(i, j);
  
  %     if Sol(i, j) < Gcenter(j)
  %       S_QOTLBO(i, j) = Gcenter(j) + (S_Qopposed(i, j) - Gcenter(j))*rand(1);
  %     else
  %       S_QOTLBO(i, j) = S_Qopposed(i, j) + (Gcenter(j) - S_Qopposed(i, j))*rand(1);
  %     end
  %   end
  
  %   S_QOTLBO(i, :) = simplebounds(S_QOTLBO(i, :), LB, UB);
  %   Fnew = fobj(S_QOTLBO(i, :));
  %   if Fnew <= Fitness(i)
  %     Sol(i,:) = S_QOTLBO(i, :);
  %     Fitness(i) = Fnew;
  %   end
  % end
  
  % 基于加权中心的反向学习方法，先计算各个花粉的权值；
  % fworst = max(Fitness)*ones(1, N);
  % %计算各个花粉个体的权重；
  % weights = fworst-Fitness(1:N);
  % weights = weights/sum(weights);
  % %对权值向量进行扩展，将其扩展维矩阵形式；
  % weights = (ones(D, 1)*weights)';
  % %计算精英花粉个体的加权几何重心；
  % Gcenter = sum(weights.*Sol(1:N,:),1);
  
  % ln_term = log((N_iter - t) ./ N_iter ./ 2);
  % sqrt_term = sqrt(4 .* t ./ N_iter);
  % Po = exp(ln_term - sqrt_term) + 0.15;
  
  % 计算所有花粉个体的反向解；
  for i=1:N
    c = rand(1,D);
    bc = c <= Po;
    RSol = Sol(i,:).*~bc+bc.*(2.0*Gcenter-Sol(i,:));  %计算随机反向解；
    RSol = simplebounds(RSol,LB,UB);%检查反向解的越界情况；
    Rfitness = fobj(RSol);
    %进行反向解和原解的优劣比较，并选择较优优的个体进行替换；
    if Rfitness < Fitness(i)
      Fitness(i) = Rfitness;
      Sol(i,:) = RSol;
    end
  end
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  % if t > 3 && cg_curve(t-3) == cg_curve(t-1)
  %   [~,sortidx] = sort(Fitness);
  
  %   for i = 1:round(size(Sol,1)/2)
  %     teta=rand*pi/2.1;
  %     step = 1*sign(rand-0.5)*norm(gbest)*log(1+10*D/N_iter);
  %     if isequal(Sol(sortidx(i),:) , gbest)
  %       S(sortidx(i),:) = gbest + step*tan(teta).*(rand.*gbest-Sol(sortidx(i),:));
  %     else
  %       S(sortidx(i),:) = gbest + step*tan(teta).*(gbest-Sol(sortidx(i),:));
  %     end
  
  %     S(sortidx(i),:) = simplebounds(S(sortidx(i),:), UB, LB);
  %     tempFitness = fobj(S(sortidx(i),:));
  %     if tempFitness<Fitness(sortidx(i))
  %       Fitness(sortidx(i))=tempFitness;
  %       Sol(sortidx(i),:)= S(sortidx(i),:);
  %     end
  %   end
  
  %   for i = round(size(Sol,1)/2)+1:size(Sol,1)
  %     r1=randi([1,size(Sol,1)],1,1);
  %     while(r1==i)
  %       r1=randi([1,size(Sol,1)],1,1);
  %     end
  %     r2=randi([1,size(Sol,1)],1,1);
  %     while(r2==r1)||(r2==i)
  %       r2=randi([1,size(Sol,1)],1,1);
  %     end
  %     r3=randi([1,size(Sol,1)],1,1);
  %     while(r3==i)||(r3==r2)||(r3==r1)
  %       r3=randi([1,size(Sol,1)],1,1);
  %     end
  
  %     F0=0.4;
  %     S(sortidx(i),:)=Sol(r1,:)+F0*(Sol(r2,:)-Sol(r3,:));
  
  %     S(sortidx(i),:) = simplebounds(S(sortidx(i),:), UB, LB);
  %     tempFitness = fobj(S(sortidx(i),:));
  %     if tempFitness<Fitness(sortidx(i))
  %       Fitness(sortidx(i))=tempFitness;
  %       Sol(sortidx(i),:)= S(sortidx(i),:);
  %     end
  %   end
  % end
  
  for i = 1:N
    if Fitness(i) <= pmin
      pbest = Sol(i, :);
      pmin = Fnew;
    end
  end
  
  if pmin < gmin
    gbest = pbest;
    gmin = pmin;
  end
  
  
  cg_curve(t) = gmin;
end

bestFitValue = gmin;
bestSolution = gbest;
end

% -------------------------------------------------------------------------
function result = initialization(N, D, UB, LB)
% result = repmat(LB, N, 1) + chaos(N, D).*repmat((UB - LB), N, 1);
result = repmat(LB, N, 1) + rand(N, D).*repmat((UB - LB), N, 1);
end

% -------------------------------------------------------------------------
function result = chaos(N, D)
cubic = 3;
Cubic = rand(N, D);

for i=1:N
  for j=2:D
    Cubic(i, j) = cubic.*Cubic(i, j - 1).*(1 - Cubic(i, j - 1).^2);
  end
end

result = Cubic;
end

% -------------------------------------------------------------------------
function Y = LnF3(alpha, sigma, m, N)
Z = laplacernd(m, N);
Z = sign(rand(m,N)-0.5) .* Z;
U = rand(m, N);
R = sin(0.5*pi*alpha) .* tan(0.5*pi*(1-alpha*U)) - cos(0.5*pi*alpha);
Y = sigma * Z .* (R) .^ (1/alpha);
end

% -------------------------------------------------------------------------
function x = laplacernd(m, N)
u1 = rand(m, N);
u2 = rand(m, N);
x = log(u1./u2);
end