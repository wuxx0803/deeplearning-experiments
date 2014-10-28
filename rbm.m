classdef rbm < handle
properties(Constant)
  sigmoid = @(x)(1./(1+exp(-x)));
  p_hidden = @(vis,W,b) rbm.sigmoid(bsxfun(@plus, vis*W, b));
  p_visible = @(hid,W,a) rbm.sigmoid(bsxfun(@plus, hid*W',a));
  bsample = @(x) (double(x > rand(size(x))));
end
properties(SetAccess=private)
  n_vis;
  n_hid;
end
properties
  W = [];
  a = [];
  b = [];
end
methods
    function this = rbm(n_vis, n_hid, n_level)
    this.n_vis = n_vis;
    this.n_hid = n_hid;
    k = .1;
    this.W = k*(2*rand(this.n_vis, this.n_hid)-1);
    this.b = k*(2*rand(1, this.n_hid)-1);
    this.a = k*(2*rand(1, this.n_vis)-1);
    end
  
    function [dW, db, da] = contrastive_divergence(this, vis, N)
    % Computes the gradient of the parameters using contrastive divergence
    hid = this.bsample(this.p_hidden(vis,this.W,this.b));
    p_v = vis;
    p_h = hid;
    for cnt = 1:N
      p_v = this.bsample(this.p_visible(this.bsample(p_h),this.W,this.a));
      p_h = this.bsample(this.p_hidden(this.bsample(p_v),this.W,this.b));
    end
    b_hid = this.bsample(hid);
    b_p_h = this.bsample(p_h);
    b_vis = this.bsample(vis);
    b_p_v = this.bsample(p_v);
    dW = ( b_vis'*b_hid - b_p_v'*b_p_h )/ size(b_vis,1);
    da = mean(b_vis-b_p_v,1);
    db = mean(b_hid-b_p_h,1);
    end
 
    function learning_step(this, vis, e, nbatch, Ncd)
    % Make one learning step
    ids = randperm(size(vis,1));
    for cnt = 1:ceil((size(vis,1)-1)/nbatch)
      id1 = nbatch*(cnt-1)+1;
      id2 = nbatch*cnt;
      if id2 > length(ids) - nbatch + 1;
        id2 = length(ids);
      end
      [dW, db, da] = this.contrastive_divergence(vis(ids(id1:id2),:), Ncd);
      this.W = this.W + e*dW;
      this.a = this.a + e*da;
      this.b = this.b + e*db;
    end
    end
  
    function learn(this, dat, n_step, e, nbatch, Ncd)
    if nargin < 5 | isempty(nbatch)
      nbatch = 128;
    end
    if nargin < 6 | isempty(N)
      N = 1;
    end
    for cnt = 1:n_step
      this.learning_step(dat, e, nbatch, N);
    end
    end

    function rdat = gibbs_sampling(this, dat, Ncd)
    vis = dat;
    hid = this.p_hidden(vis,this.W,this.b);
    p_v = vis;
    p_h = hid;
    for cnt = 1:Ncd
      p_v = this.p_visible(this.bsample(p_h),this.W,this.a);
      p_h = this.p_hidden(p_v,this.W,this.b);
    end
    [~, rdat] = max(p_v,[],3);
    rdat = p_v;
    end
  
    function showweights(this, nx, ny)
    cnt = 0;
    if nargin ~= 3
      nx = size(this.W,1);
      ny = 1;
    end
    for cnt1 = 1:size(this.W,3)
      for cnt2 = 1:size(this.W,2)
        cnt = cnt + 1;
        subplot(size(this.W,3), size(this.W,2), cnt);
        imagesc(reshape(this.W(:,cnt2,cnt1),nx,ny)');
        set(gca, 'DataAspectRatio', [1 1 1]);
      end
    end
    end
  
    function h = hidden(this, dat)
    h = this.p_hidden(dat,this.W,this.b);
    end
end
end