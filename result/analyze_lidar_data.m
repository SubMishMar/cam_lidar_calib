Data = csvread('csv_lidar_out.csv');
i = Data(:,1);
j = Data(:,2);
Z = Data(:,3);
plot3(i,j,Z,'.');
grid;
