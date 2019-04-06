#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/CameraInfo.h"
#include "sensor_msgs/Image.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <velodyne_pointcloud/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "opencv2/opencv.hpp"

#include <cv_bridge/cv_bridge.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CameraInfo,
        sensor_msgs::PointCloud2,
        sensor_msgs::Image> SyncPolicy;

class camLidarCalib {
private:
    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::CameraInfo> *caminfo_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud_sub;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;

    cv::Mat image_in;
    cv::Mat image_resized;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    bool boardDetectedInCam;
public:


    camLidarCalib() {
        caminfo_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, "/pylon_camera_node/cam_info", 1);
        cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/velodyne_points", 1);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, "/pylon_camera_node/image_raw", 1);
        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *caminfo_sub, *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2, _3));
        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(4, 1, CV_64F);
        boardDetectedInCam = false;

        for(int i = 0; i <9; i++){
            for (int j = 0; j < 6; j++) {
                
            }
        }
    }

    void callback(const sensor_msgs::CameraInfoConstPtr &camInfo_msg,
                  const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg){

        projection_matrix.at<double>(0, 0) = camInfo_msg->K[0];
        projection_matrix.at<double>(0, 1) = camInfo_msg->K[1];
        projection_matrix.at<double>(0, 2) = camInfo_msg->K[2];

        projection_matrix.at<double>(1, 0) = camInfo_msg->K[3];
        projection_matrix.at<double>(1, 1) = camInfo_msg->K[4];
        projection_matrix.at<double>(1, 2) = camInfo_msg->K[5];

        projection_matrix.at<double>(2, 1) = camInfo_msg->K[6];
        projection_matrix.at<double>(2, 1) = camInfo_msg->K[7];
        projection_matrix.at<double>(2, 2) = camInfo_msg->K[8];

        distCoeff.at<double>(0) = camInfo_msg->D[0];
        distCoeff.at<double>(1) = camInfo_msg->D[1];
        distCoeff.at<double>(2) = camInfo_msg->D[2];
        distCoeff.at<double>(3) = camInfo_msg->D[3];

        try {
            image_in = cv_bridge::toCvShare(image_msg, "bgr8")->image;
            boardDetectedInCam = cv::findChessboardCorners(image_in, cv::Size(9, 6),
                    image_points, cv::CALIB_CB_ADAPTIVE_THRESH+
                    cv::CALIB_CB_NORMALIZE_IMAGE);
            ROS_INFO_STREAM("boardDetectedInCam = " << boardDetectedInCam);
            ROS_INFO_STREAM("No of corners = " << image_points.size());
            cv::drawChessboardCorners(image_in, cv::Size(9, 6),
                    image_points, boardDetectedInCam);

            cv::resize(image_in, image_resized, cv::Size(), 0.25, 0.25);
            cv::imshow("view", image_resized);
            cv::waitKey(10);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.",
                    image_msg->encoding.c_str());
        }

    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "CameraLidarCalib_node");
    camLidarCalib cLC;
    ros::spin();
    return 0;
}