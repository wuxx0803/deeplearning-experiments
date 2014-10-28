% rnn = rneuralnet(n_input, n_dynamic, n_output, p)
classdef rneuralnet < handle
% x = sigma(Wx + Uy + b)
properties(Constant)
  logistic = @(x) (1./(1+exp(-x)));
  inv_logistic = @(z) (z.*(1-z));
  softmax = @(x) (bsxfun(@times, exp(x), 1./sum(exp(x))));
  inv_softmax = @(z) (diag(z)-z*z');
  times_element = @(x,y)(x.*y);
  times_matrix = @(x,y)(x*y);
end
properties(SetAccess=private)
  n_input = [];
  n_dynamic = [];
  n_output = [];

  p = [];

  W = [];
  V = [];
  U = [];
  b = [];
  
  sigma  = [];
  isigma = [];
  
  y  = [];
  z0 = [];
  x  = [];
  z  = [];

  S = [];
  iS = [];

  dx_dw = [];
  dx_du = [];
  dx_db = [];
  
  dE_dw = [];
  dE_du = [];
  dE_db = [];
end

methods
  function o = rneuralnet(n_input, n_dynamic, n_output, p)
  o.n_input   = n_input;
  o.n_dynamic = n_dynamic;
  o.n_output  = n_output;
  o.p = p;
  
  %rand('seed', 3895719837);
  k = 10;
  o.W = k * (-1 + 2*rand(o.n_dynamic, o.n_dynamic));
  %o.W = o.W - diag(diag(o.W)); % Be sure that the agent does not
                               % project onto itself since this can
                               % change the characteristic time of
                               % the neuron
  o.U = k * (-1 + 2*rand(o.n_dynamic, o.n_input  ));
  o.V = k * (-1 + 2*rand(o.n_output,  o.n_dynamic));
  o.b = k * (-1 + 2*rand(o.n_dynamic, 1));
  
  o.U = zeros(size(o.U));
  o.U(1,1) = 1;
  o.V = zeros(size(o.V));
  o.V(1,end) = 1;
  
  o.sigma  = @(x)(-1+2*rneuralnet.logistic(x));
  o.isigma = @(z)(2*rneuralnet.inv_logistic(.5*(1+z)));
  end

  function o.setweights(o, nlayer, W, b)
  o.W = W;
  o.b = b;
  end

  function set_learing_data(o, y, z0)
  o.y  = y; % input
  o.z0 = z0;% desired output
  end

  function E = forward_prop(o)
  o.x = zeros(o.n_dynamic, size(o.y,2)); % internal state
  o.z = zeros(o.n_output, size(o.y,2));  % nn output
  o.dx_dw = zeros(o.n_dynamic, o.n_dynamic, o.n_dynamic, size(o.y,2));
  o.dx_db = zeros(o.n_dynamic, o.n_dynamic, size(o.y,2));
  o.dx_du = zeros(o.n_dynamic, o.n_dynamic, o.n_output, size(o.y,2));
  o.S  = zeros(size(o.x));
  o.iS = zeros(size(o.S));
  for cnt = 2:size(o.x,2)
    o.S(:,cnt-1)  = o.sigma(o.W*o.x(:,cnt-1) + o.U*o.y(:,cnt-1) + o.b);
    o.iS(:,cnt-1) = o.isigma(o.S(:,cnt-1));
    o.x(:,cnt) = (1-o.p)*o.x(:,cnt-1) + o.p*o.S(:,cnt-1); % <-- forward prop
    o.z(:,cnt) = o.V * o.x(:,cnt);
    % for backprop
    o.dx_dw(:,:,:,cnt) = (1-o.p)*o.dx_dw(:,:,:,cnt-1);
    for k = 1:o.n_dynamic
      o.dx_dw(:,k,:,cnt) = o.dx_dw(:,k,:,cnt) + permute(bsxfun(@times,o.W*permute(o.dx_dw(:,k,:,cnt-1),[1,3,2,4]),o.p*o.iS(:,cnt-1)),[1,3,2,4]);
      o.dx_dw(k,k,:,cnt) = o.dx_dw(k,k,:,cnt) + permute(o.p*o.iS(k,cnt-1)*o.x(:,cnt-1),[2,3,1]);
    end

    % o.dx_du(:,:,:,cnt) = (1-o.p)*o.dx_du(:,:,:,cnt-1);
    % for k = 1:o.n_dynamic
    %   for l = 1:o.n_output
    %     o.dx_du(:,k,l,cnt) = o.dx_du(:,k,l,cnt) + o.p*o.iS(:,cnt-1).*(o.W*o.dx_du(:,k,l,cnt-1));
    %   end
    %   o.dx_du(k,k,:,cnt) = o.dx_du(k,k,:,cnt) + permute(o.p*o.iS(k,cnt-1)*o.y(:,cnt-1),[2,3,1]);
    % end
    
    o.dx_db(:,:,cnt) = (1-o.p)*o.dx_db(:,:,cnt-1);
    o.dx_db(:,:,cnt) = o.dx_db(:,:,cnt) + bsxfun(@times, o.W*o.dx_db(:,:,cnt-1), o.p*o.iS(:,cnt-1));
    o.dx_db(:,:,cnt) = o.dx_db(:,:,cnt) + diag(o.p*o.iS(:,cnt-1));
  end
  E = .5*sum( (o.z-o.z0).^2, 1 );
  end



  
  function back_prop(o)
  
  o.dE_dw = zeros(o.n_dynamic, o.n_dynamic, size(o.y,2));
  o.dE_du = zeros(o.n_dynamic, o.n_output, size(o.y,2));
  o.dE_db = zeros(o.n_dynamic, size(o.y,2));
  for i = 1:o.n_dynamic
    for j = 1:o.n_dynamic
      o.dE_dw(i,j,:) = sum((o.z-o.z0) .* (o.V * squeeze(o.dx_dw(:,i,j,:))),1);
    end
  end
  % for i = 1:o.n_dynamic
  %   for j = 1:o.n_output
  %     o.dE_du(i,j,:) = sum((o.z-o.z0) .* (o.V * squeeze(o.dx_du(:,i,j,:))),1);
  %   end
  % end
  for i = 1:o.n_dynamic
    o.dE_db(i,:) = sum((o.z-o.z0) .* (o.V * squeeze(o.dx_db(:,i,:))),1);
  end
  % For debugging!!!
  % d = 1e-6;
  % E0 = o.forward_prop(); W = o.W; dE_dw = []; for i = 1:o.n_dynamic; for j = 1:o.n_dynamic; o.W = W; o.W(i,j) = W(i,j) + d; E1 = o.forward_prop(); dE_dw(i,j,:) = (E1-E0)/d; end; end; o.W = W; o.forward_prop();
  % E0 = o.forward_prop(); b = o.b; dE_db = []; for i = 1:o.n_dynamic; o.b = b; o.b(i) = b(i) + d; E1 = o.forward_prop(); dE_db(i,:) = (E1-E0)/d; end; o.b = b; o.forward_prop();
  % x0 = o.x; W = o.W; dx_dw = []; for i = 1:o.n_dynamic; for j = 1:o.n_dynamic; o.W = W; o.W(i,j) = o.W(i,j) + d; o.forward_prop(); x1 = o.x; dx_dw(:,i,j,:) = permute((x1-x0)/d,[1,3,4,2]); end; end; o.W = W;o.forward_prop();
  % keyboard;
  
  end

  function [z, E] = learn(o, y, z, N, eta)
  
  E = [];
  o.set_learing_data(y,z);
  for cnt = 1:N
    err = o.forward_prop();
    E = [E; mean(err)];
    o.back_prop();
    ids = 1:size(o.y,2);
    o.W = o.W - eta*mean(o.dE_dw(:,:,ids),3);
    o.b = o.b - eta*mean(o.dE_db(:,ids),2);
    if length(E) >= 10
      if max(abs(diff(E(end-9:end)))) < 1e-5
        o.W = o.W + 10*(-1+2*rand(size(o.W)));
        o.b = o.b + 10*(-1+2*rand(size(o.b)));
        disp('Randomize!');
      end
    end
  end
  z = o.z;
  disp(mean(err));
  end

end
end