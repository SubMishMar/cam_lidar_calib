X = csvread('csv_out.csv');
data = [];
for i = 1:20:size(X, 1)
   for j = 1:20:size(X, 2)
	disp = X(i, j)/16;
	if(disp > 10 && disp < 192)
	        depth = 1.6124943463258518e+02/disp;
		plot3(i, j, depth, '*')
		hold on;
	end
   end
end
grid;
hold off;

