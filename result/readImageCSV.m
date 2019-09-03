clear all
close all
X = csvread('csv_out.csv');
figure(1)
data_disp_stereo = [];
data_disp = [];
for i = 1:20:size(X, 1)
   for j = 1:20:size(X, 2)
	disp = X(i, j)/16;
	if(disp > 0 && disp < 192)
	        %depth = 1.6124943463258518e+02/disp;
		plot3(j, i, disp, '*');
	        data_disp_stereo = [data_disp_stereo; disp];
		hold on;
	end
   end
end
grid;
hold off;
data_disp_lidar = [];
figure(2)
Data = csvread('csv_lidar_out.csv');
i = Data(:,1);
j = Data(:,2);
Z = Data(:,3);
plot3(i,j,1.6124943463258518e+02./Z,'.');
data_disp_lidar = 1.6124943463258518e+02./Z;
grid;

figure(3)
hist(data_disp_stereo, 100);
grid;
figure(4)
hist(data_disp_lidar, 100);
grid;

min(Z)

figure(5)
plot(data_disp_stereo)
figure(6)
plot(data_disp_lidar)
