//
// Created by usl on 4/6/19.
//

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
#include <pcl/filters/passthrough.h>

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>

#include <calibration_error_term.h>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "ceres/rotation.h"
#include "ceres/covariance.h"

#include <fstream>
#include <iostream>

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
    ros::Publisher cloud_pub;

    cv::Mat image_in;
    cv::Mat image_resized;
    cv::Mat projection_matrix;
    cv::Mat distCoeff;
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> projected_points;
    bool boardDetectedInCam;
    double dx, dy;
    int checkerboard_rows, checkerboard_cols;
    int min_points_on_plane;
    cv::Mat tvec, rvec;
    cv::Mat C_R_W;
    Eigen::Matrix3d c_R_w;
    Eigen::Vector3d c_t_w;
    Eigen::Vector3d r3;
    Eigen::Vector3d r3_old;
    Eigen::Vector3d Nc;

    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<std::vector<Eigen::Vector3d> > all_lidar_points;
    std::vector<Eigen::Vector3d> all_normals;

    sensor_msgs::PointCloud2 out_cloud;
    std::string result_str;

    std::string camera_in_topic;
    std::string lidar_in_topic;
    std::string camera_info_in_topic;

    int num_views;
public:


    camLidarCalib() {
        camera_info_in_topic = readParam<std::string>(nh, "camera_info_in_topic");
        camera_in_topic = readParam<std::string>(nh, "camera_in_topic");
        lidar_in_topic = readParam<std::string>(nh, "lidar_in_topic");

        caminfo_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, camera_info_in_topic, 1);
        cloud_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, lidar_in_topic, 1);
        image_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, camera_in_topic, 1);
        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), *caminfo_sub, *cloud_sub, *image_sub);
        sync->registerCallback(boost::bind(&camLidarCalib::callback, this, _1, _2, _3));
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("velodyne_points_out", 1);
        projection_matrix = cv::Mat::zeros(3, 3, CV_64F);
        distCoeff = cv::Mat::zeros(4, 1, CV_64F);
        boardDetectedInCam = false;
        tvec = cv::Mat::zeros(3, 1, CV_64F);
        rvec = cv::Mat::zeros(3, 1, CV_64F);
        C_R_W = cv::Mat::eye(3, 3, CV_64F);
        c_R_w = Eigen::Matrix3d::Identity();

        dx = readParam<double>(nh, "dx");
        dy = readParam<double>(nh, "dy");
        checkerboard_rows = readParam<int>(nh, "checkerboard_rows");
        checkerboard_cols = readParam<int>(nh, "checkerboard_cols");
        min_points_on_plane = readParam<int>(nh, "min_points_on_plane");
        num_views = readParam<int>(nh, "num_views");

        for(int i = 0; i < checkerboard_rows; i++)
            for (int j = 0; j < checkerboard_cols; j++)
                object_points.emplace_back(cv::Point3f(i*dx, j*dy, 0.0));

        result_str = readParam<std::string>(nh, "result_file");
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

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {

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
        pass_x.setFilterLimits(0.0, 6.0);
        pass_x.filter(*cloud_filtered_x);
        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud_filtered_x);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(-1, 1);
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

        /// Store the points lying in the filtered plane in a vector
        lidar_points.clear();
        for (size_t i = 0; i < plane_filtered->points.size(); i++) {
            double X = plane_filtered->points[i].x;
            double Y = plane_filtered->points[i].y;
            double Z = plane_filtered->points[i].z;
            lidar_points.push_back(Eigen::Vector3d(X, Y, Z));
        }
        pcl::toROSMsg(*plane_filtered, out_cloud);
        cloud_pub.publish(out_cloud);
    }

    void imageHandler(const sensor_msgs::CameraInfoConstPtr &camInfo_msg,
                      const sensor_msgs::ImageConstPtr &image_msg) {
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
            boardDetectedInCam = cv::findChessboardCorners(image_in,
                                                           cv::Size(checkerboard_cols, checkerboard_rows),
                                                           image_points,
                                                           cv::CALIB_CB_ADAPTIVE_THRESH+
                                                           cv::CALIB_CB_NORMALIZE_IMAGE);
            cv::drawChessboardCorners(image_in,
                                      cv::Size(checkerboard_cols, checkerboard_rows),
                                      image_points,
                                      boardDetectedInCam);
            if(image_points.size() == object_points.size()){
                cv::solvePnP(object_points, image_points, projection_matrix, distCoeff, rvec, tvec, false, CV_ITERATIVE);
                projected_points.clear();
                cv::projectPoints(object_points, rvec, tvec, projection_matrix, distCoeff, projected_points, cv::noArray());
                for(int i = 0; i < projected_points.size(); i++){
                    cv::circle(image_in, projected_points[i], 16, cv::Scalar(0, 255, 0), 10, cv::LINE_AA, 0);
                }
                cv::Rodrigues(rvec, C_R_W);
                cv::cv2eigen(C_R_W, c_R_w);
                c_t_w = Eigen::Vector3d(tvec.at<double>(0),
                                        tvec.at<double>(1),
                                        tvec.at<double>(2));

                r3 = c_R_w.block<3,1>(0,2);
                Nc = (r3.dot(c_t_w))*r3;
            }
            cv::resize(image_in, image_resized, cv::Size(), 0.25, 0.25);
            cv::imshow("view", image_resized);
            cv::waitKey(10);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("Could not convert from '%s' to 'bgr8'.",
                      image_msg->encoding.c_str());
        }
    }

    void runSolver() {
        if (lidar_points.size() > min_points_on_plane && boardDetectedInCam) {
            if (r3.dot(r3_old) < 0.9) {
                r3_old = r3;
                all_normals.push_back(Nc);
                all_lidar_points.push_back(lidar_points);
                ROS_ASSERT(all_normals.size() == all_lidar_points.size());
                ROS_INFO_STREAM("Recording View number: " << all_normals.size());
                if (all_normals.size() >= num_views) {
                    ROS_INFO_STREAM("Starting optimization...");

                    /// Start Optimization here

                    /// Step 1: Initialization
                    Eigen::Matrix3d Rotn;
                    Rotn(0, 0) = 1; Rotn(0, 1) = 0; Rotn(0, 2) = 0;
                    Rotn(1, 0) = 0; Rotn(1, 1) = 1; Rotn(1, 2) = 0;
                    Rotn(2, 0) = 0; Rotn(2, 1) = 0; Rotn(2, 2) = 1;
                    Eigen::Vector3d axis_angle;
                    ceres::RotationMatrixToAngleAxis(Rotn.data(), axis_angle.data());

                    Eigen::Vector3d Translation = Eigen::Vector3d(0, 0, 0);
                    Eigen::VectorXd R_t(6);
                    R_t(0) = axis_angle(0);
                    R_t(1) = axis_angle(1);
                    R_t(2) = axis_angle(2);
                    R_t(3) = Translation(0);
                    R_t(4) = Translation(1);
                    R_t(5) = Translation(2);
                    /// Step2: Defining the Loss function (Can be NONE)
                    ceres::LossFunction *loss_function = NULL;

                    /// Step 3: Form the Optimization Problem
                    ceres::Problem problem;
                    problem.AddParameterBlock(R_t.data(), 6);
                    for (int i = 0; i < all_normals.size(); i++) {
                        Eigen::Vector3d normal_i = all_normals[i];
                        std::vector<Eigen::Vector3d> lidar_points_i
                                = all_lidar_points[i];
                        for (int j = 0; j < lidar_points_i.size(); j++) {
                            Eigen::Vector3d lidar_point = lidar_points_i[j];
                            ceres::CostFunction *cost_function = new
                                    ceres::AutoDiffCostFunction<CalibrationErrorTerm, 1, 6>
                                    (new CalibrationErrorTerm(lidar_point, normal_i));
                            problem.AddResidualBlock(cost_function, loss_function, R_t.data());
//                            problem.SetParameterization(quatn.coeffs().data(), quaternion_local_parameterization);
                        }
                    }

                    /// Step 4: Solve it
                    ceres::Solver::Options options;
                    options.max_num_iterations = 200;
                    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
                    options.minimizer_progress_to_stdout = true;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    std::cout << summary.FullReport() << '\n';



                    /// Printing and Storing C_T_L in a file
                    ceres::AngleAxisToRotationMatrix(R_t.data(), Rotn.data());
                    Eigen::MatrixXd C_T_L(3, 4);
                    C_T_L.block(0, 0, 3, 3) = Rotn;
                    C_T_L.block(0, 3, 3, 1) = Eigen::Vector3d(R_t[3], R_t[4], R_t[5]);

                    std::cout << "C_T_L = " << std::endl;
                    std::cout << C_T_L << std::endl;

                    /// Step 5: Covariance Estimation
                    ceres::Covariance::Options options_cov;
                    ceres::Covariance covariance(options_cov);
                    std::vector<std::pair<const double*, const double*> > covariance_blocks;
                    covariance_blocks.push_back(std::make_pair(R_t.data(), R_t.data()));
                    CHECK(covariance.Compute(covariance_blocks, &problem));
                    double covariance_xx[6 * 6];
                    covariance.GetCovarianceBlock(R_t.data(),
                                                  R_t.data(),
                                                  covariance_xx);

                    Eigen::MatrixXd cov_mat_RotTrans(6, 6);
                    cv::Mat cov_mat_cv = cv::Mat(6, 6, CV_64F, &covariance_xx);
                    cv::cv2eigen(cov_mat_cv, cov_mat_RotTrans);

                    Eigen::MatrixXd cov_mat_TransRot(6, 6);
                    cov_mat_TransRot.block(0, 0, 3, 3) = cov_mat_RotTrans.block(3, 3, 3, 3);
                    cov_mat_TransRot.block(3, 3, 3, 3) = cov_mat_RotTrans.block(0, 0, 3, 3);
                    cov_mat_TransRot.block(0, 3, 3, 3) = cov_mat_RotTrans.block(3, 0, 3, 3);
                    cov_mat_TransRot.block(3, 0, 3, 3) = cov_mat_RotTrans.block(0, 3, 3, 3);

                    double  sigma_xx = sqrt(cov_mat_TransRot(0, 0));
                    double  sigma_yy = sqrt(cov_mat_TransRot(1, 1));
                    double  sigma_zz = sqrt(cov_mat_TransRot(2, 2));

                    double sigma_rot_xx = sqrt(cov_mat_TransRot(3, 3));
                    double sigma_rot_yy = sqrt(cov_mat_TransRot(4, 4));
                    double sigma_rot_zz = sqrt(cov_mat_TransRot(5, 5));

                    std::cout << "sigma_xx = " << sigma_xx << "\t"
                              << "sigma_yy = " << sigma_yy << "\t"
                              << "sigma_zz = " << sigma_zz << std::endl;

                    std::cout << "sigma_rot_xx = " << sigma_rot_xx*180/M_PI << "\t"
                              << "sigma_rot_yy = " << sigma_rot_yy*180/M_PI << "\t"
                              << "sigma_rot_zz = " << sigma_rot_zz*180/M_PI << std::endl;

                    std::ofstream results;
                    results.open(result_str);
                    results << C_T_L;
                    results.close();
                    ros::shutdown();
                }
            } else {
                ROS_WARN_STREAM("Not enough Rotation, view not recorded");
            }
        } else {
            if(!boardDetectedInCam)
                ROS_WARN_STREAM("Checker-board not detected in Image.");
            else {
                ROS_WARN_STREAM("Checker Board Detected in Image?: " << boardDetectedInCam << "\t" <<
                "No of LiDAR pts: " << lidar_points.size() << " (Check if this is less than threshold) ");
            }
        }
    }

    void callback(const sensor_msgs::CameraInfoConstPtr &camInfo_msg,
                  const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                  const sensor_msgs::ImageConstPtr &image_msg) {
        imageHandler(camInfo_msg, image_msg);
        cloudHandler(cloud_msg);
        runSolver();
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "CameraLidarCalib_node");
    camLidarCalib cLC;
    ros::spin();
    return 0;
}