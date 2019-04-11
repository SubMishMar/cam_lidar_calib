//
// Created by usl on 4/10/19.
//

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <tf/transform_listener.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <iostream>
#include <fstream>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CameraInfo,
        sensor_msgs::PointCloud2,
        sensor_msgs::Image> SyncPolicy;

class lidarImageProjection {
private:

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::CameraInfo> *camInfo_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;

    Eigen::MatrixXd C_T_L;

    std::vector<cv::Point3d> lidar_pts_in_fov;

    std::string result_str;

public:
    lidarImageProjection() {

        camInfo_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, "/pylon_camera_node/cam_info", 1);
        cloud_sub =  new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/velodyne_points", 1);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, "/pylon_camera_node/image_raw", 1);

        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *camInfo_sub, *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&lidarImageProjection::callback, this, _1, _2, _3));

        C_T_L = Eigen::MatrixXd(3, 4);

        result_str = readParam<std::string>(nh, "result_file");

        std::ifstream myReadFile(result_str.c_str());
        std::string word;
        int i = 0;
        int j = 0;
        while (myReadFile >> word){
            C_T_L(i, j) = atof(word.c_str());
            j++;
            if(j>3) {
                j = 0;
                i++;
            }
        }
    }

    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name)
    {
        T ans;
        if (n.getParam(name, ans))
        {
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        }
        else
        {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    void callback(const sensor_msgs::CameraInfoConstPtr &camInfo_msg,
                  const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg) {

        lidar_pts_in_fov.clear();

        Eigen::Matrix3d camMat;
        camMat << camInfo_msg->P[0], camInfo_msg->P[1], camInfo_msg->P[2],
                camInfo_msg->P[4], camInfo_msg->P[5], camInfo_msg->P[6],
                camInfo_msg->P[8], camInfo_msg->P[9], camInfo_msg->P[10];

        double fov_x, fov_y;
        fov_x = 2*atan2(camInfo_msg->height, 2*camInfo_msg->P[0])*180/CV_PI;
        fov_y = 2*atan2(camInfo_msg->width, 2*camInfo_msg->P[5])*180/CV_PI;

        pcl::PCLPointCloud2 *cloud_in = new pcl::PCLPointCloud2;
        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        pcl_conversions::toPCL(*cloud_msg, *cloud_in);
        pcl::fromPCLPointCloud2(*cloud_in, *in_cloud);

        double max_range, min_range;
        max_range = 0;
        min_range = 1000000;
        for(size_t i = 0; i < in_cloud->points.size(); i++) {

            // Reject points behind the LiDAR
            if(in_cloud->points[i].x < 0)
                continue;

            Eigen::Vector4d pointCloud_L;
            pointCloud_L[0] = in_cloud->points[i].x;
            pointCloud_L[1] = in_cloud->points[i].y;
            pointCloud_L[2] = in_cloud->points[i].z;
            pointCloud_L[3] = 1;

            Eigen::Vector3d pointCloud_C;
            pointCloud_C = C_T_L*pointCloud_L;


            double X = pointCloud_C[0];
            double Y = pointCloud_C[1];
            double Z = pointCloud_C[2];

            double Xangle = atan2(X, Z)*180/CV_PI;
            double Yangle = atan2(Y, Z)*180/CV_PI;

            if(Xangle < -fov_x/2 || Xangle > fov_x/2)
                continue;

            if(Yangle < -fov_y/2 || Yangle > fov_y/2)
                continue;

            double x_1 = X/Z;
            double y_1 = Y/Z;

            double range = sqrt(X*X + Y*Y + Z*Z);

            if(range > max_range) {
                max_range = range;
            }
            if(range < min_range) {
                min_range = range;
            }
//            double r = x_1*x_1 + y_1*y_1;

//        x_1 = (x_1 * (1.0 + Dist(0) * r * r + Dist(1) * r * r * r * r +
//                      Dist(4) * r * r * r * r * r * r) +
//               2 * Dist(2) * x_1 * y_1 + Dist(3) * (r * r + 2 * x_1));
//
//        y_1 = (y_1 * (1.0 + Dist(0) * r * r + Dist(1) * r * r * r * r +
//                      Dist(4) * r * r * r * r * r * r) +
//               2 * Dist(3) * x_1 * y_1 + Dist(2) * (r * r + 2 * y_1));

            Eigen::Vector3d x1y1w;
            x1y1w << x_1, y_1, 1;
            Eigen::Vector3d uvw = camMat*x1y1w;
            lidar_pts_in_fov.push_back(cv::Point3d(uvw(0), uvw(1), range));
        }

        cv::Mat image_in = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        if(lidar_pts_in_fov.size() > 0) {
            for(size_t i = 0; i < lidar_pts_in_fov.size(); i++) {
                double range = lidar_pts_in_fov[i].z;
                double red_field = 255*(range - min_range)/(max_range - min_range);
                double green_field = 255*(max_range - range)/(max_range - min_range);
                cv::circle(image_in, cv::Point2d(lidar_pts_in_fov[i].x,
                                                 lidar_pts_in_fov[i].y), 4, CV_RGB(red_field, green_field, 0), -1, 8, 0);}
        } else {
            ROS_WARN("No lidar points in FOV");
        }
        cv::Mat image_resized;
        cv::resize(image_in, image_resized, cv::Size(), 0.25, 0.25);
        cv::imshow("view", image_resized);
        cv::waitKey(10);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cam_lidar_proj");
    cv::namedWindow("view");
    cv::startWindowThread();
    lidarImageProjection lip;
    ros::spin();
    cv::destroyWindow("view");
    return 0;
}