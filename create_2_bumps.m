function x = create_2_bumps(N, Ngrid)
% x = create_2_bumps(N)
%
% Creates a set x of N random points drawn from 2d GMM with 2
% well-separated bumps.
%
% Y = create_2_bumps(N,grid)
%
% Creates a set of N random Ngrid x Ngrid images. Each image is
% produced from x row from the previous usage using gaussian
% excitation function for its pixels.



sigma = .03;
n_sigma = 100;

bmp = [.3, .3; .7, .6];

x = [];
for cnt = 1:size(bmp,1)
  x = [x; bsxfun(@plus, sigma*random('norm', 0, 1, N, 2), bmp(cnt,:))];
end

if nargin == 2
  [g1, g2] = meshgrid(1:Ngrid,1:Ngrid);
  Rdecay = 1000/Ngrid;
  Y = zeros(size(x,1), Ngrid^2);
  for cnt = 1:size(x,1)
    y = zeros(10,10);
    d = exp( - Rdecay * ( (g1/Ngrid-x(cnt,1)).^2 + (g2/Ngrid-x(cnt,2)).^2 ) );
    y(sub2ind(size(y),g1,g2)) = d;
    Y(cnt,:) = y(:)';
  end
  x = Y;
end