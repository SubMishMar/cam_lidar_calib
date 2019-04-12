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

#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

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

    ros::Publisher cloud_pub;

    Eigen::MatrixXd C_T_L;
    cv::Mat c_R_l, tvec;
    std::vector<cv::Point3d> lidar_pts_in_fov;

    std::string result_str;

    bool project_only_plane;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;

    std::vector<cv::Point3d> objectPoints;
    std::vector<cv::Point2d> imagePoints;

    sensor_msgs::PointCloud2 out_cloud_ros;

public:
    lidarImageProjection() {

        camInfo_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, "/pylon_camera_node/cam_info", 1);
        cloud_sub =  new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, "/velodyne_points", 1);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, "/pylon_camera_node/image_raw", 1);
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_out_cloud", 1);

        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *camInfo_sub, *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&lidarImageProjection::callback, this, _1, _2, _3));

        C_T_L = Eigen::MatrixXd(3, 4);
        c_R_l = cv::Mat::zeros(3, 3, CV_64F);
        tvec = cv::Mat::zeros(3, 1, CV_64F);

        result_str = readParam<std::string>(nh, "result_file");
        project_only_plane = readParam<bool>(nh, "project_only_plane");

        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(4, 1, CV_64F);

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
        Eigen::Matrix3d C_R_L = C_T_L.block(0, 0, 3, 3);
        Eigen::Vector3d C_t_L = C_T_L.block(0, 3, 3, 1);
        cv::eigen2cv(C_R_L, c_R_l);
        cv::eigen2cv(C_t_L, tvec);
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

    pcl::PointCloud<pcl::PointXYZ >::Ptr planeFilter(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *in_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_x(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_y(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ >::Ptr plane_filtered(new pcl::PointCloud<pcl::PointXYZ>);

        /// Pass through filters
        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(in_cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(0.0, 5.0);
        pass_x.filter(*cloud_filtered_x);
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud_filtered_x);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(-1.25, 1.25);
        pass_y.filter(*cloud_filtered_y);

        /// Plane Segmentation
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_y));
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_p);
        ransac.setDistanceThreshold(0.01);
        ransac.computeModel();
        std::vector<int> inliers_indicies;
        ransac.getInliers(inliers_indicies);
        pcl::copyPointCloud<pcl::PointXYZ>(*cloud_filtered_y, inliers_indicies, *plane);

        /// Statistical Outlier Removal
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(plane);
        sor.setMeanK (50);
        sor.setStddevMulThresh (1);
        sor.filter (*plane_filtered);

        return plane_filtered;
    }

    void callback(const sensor_msgs::CameraInfoConstPtr &camInfo_msg,
                  const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg) {
        objectPoints.clear();
        imagePoints.clear();

        cv::Mat image_in = cv_bridge::toCvShare(image_msg, "bgr8")->image;


        double fov_x, fov_y;
        fov_x = 2*atan2(camInfo_msg->height, 2*camInfo_msg->P[0])*180/CV_PI;
        fov_y = 2*atan2(camInfo_msg->width, 2*camInfo_msg->P[5])*180/CV_PI;

        size_t k = 0;
        for(size_t i = 0 ; i < 3; i++)
            for(size_t j = 0; j < 3; j++) {
                projection_matrix.at<double>(i, j) = camInfo_msg->K[k++];
            }

        for(size_t i = 0; i < 4; i++)
            distCoeff.at<double>(i) = camInfo_msg->D[i];

        cv::Mat rvec;
        cv::Rodrigues(c_R_l, rvec);

        pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if(project_only_plane) {
            in_cloud = planeFilter(cloud_msg);

            for(size_t i = 0; i < in_cloud->points.size(); i++) {
                objectPoints.push_back(cv::Point3d(in_cloud->points[i].x, in_cloud->points[i].y, in_cloud->points[i].z));
            }
            cv::Mat rvec;
            cv::Rodrigues(c_R_l, rvec);
            cv::projectPoints(objectPoints, rvec, tvec, projection_matrix, distCoeff, imagePoints, cv::noArray());
        } else {
            pcl::PCLPointCloud2 *cloud_in = new pcl::PCLPointCloud2;
            pcl_conversions::toPCL(*cloud_msg, *cloud_in);
            pcl::fromPCLPointCloud2(*cloud_in, *in_cloud);

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

                objectPoints.push_back(cv::Point3d(pointCloud_L[0],
                                                       pointCloud_L[1],
                                                       pointCloud_L[2]));
            }
            cv::projectPoints(objectPoints, rvec, tvec, projection_matrix, distCoeff, imagePoints, cv::noArray());
            pcl::PointCloud<pcl::PointXYZRGB> out_cloud_pcl;
            out_cloud_pcl.resize(objectPoints.size());

            for(size_t i = 0; i < objectPoints.size(); i++) {
                out_cloud_pcl.points[i].x = objectPoints[i].x;
                out_cloud_pcl.points[i].y = objectPoints[i].y;
                out_cloud_pcl.points[i].z = objectPoints[i].z;
//                uint8_t R = image_in.at<cv::Vec3b>(cv::Point(imagePoints[i].y, imagePoints[i].x))[0];
//                out_cloud_pcl.points[i].r = image_in.at<cv::Vec3b>(cv::Point(imagePoints[i].x, imagePoints[i].y))[0];
            }

            pcl::toROSMsg(out_cloud_pcl, out_cloud_ros);
            out_cloud_ros.header.frame_id = cloud_msg->header.frame_id;
            out_cloud_ros.header.stamp = cloud_msg->header.stamp;

            cloud_pub.publish(out_cloud_ros);
        }

        for(size_t i = 0; i < imagePoints.size(); i++)
            cv::circle(image_in, imagePoints[i], 4, CV_RGB(0, 255, 0), -1, 8, 0);

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