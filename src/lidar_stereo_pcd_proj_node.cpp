//
// Created by subodh on 8/28/19.
//

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
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
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <iostream>
#include <fstream>

typedef message_filters::sync_policies::ApproximateTime
        <sensor_msgs::PointCloud2,
         sensor_msgs::PointCloud2> SyncPolicy;

class projectClouds {
private:

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud1_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *cloud2_sub;
    message_filters::Synchronizer<SyncPolicy> *sync;

    ros::Publisher cloud_pub;

    std::string cloud1_input_topic;
    std::string cloud2_input_topic;
    std::string cloud2_output_topic;
    std::string result_str;

    Eigen::Matrix4d C_T_L;
    Eigen::Matrix4d S_T_C;
    Eigen::Matrix4d S_T_L;

public:

    projectClouds() {
        cloud1_input_topic = readParam<std::string>(nh, "cloud1_input_topic");
        cloud2_input_topic = readParam<std::string>(nh, "cloud2_input_topic");
        cloud2_output_topic = readParam<std::string>(nh, "cloud2_output_topic");

        cloud1_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, cloud1_input_topic, 1);
        cloud2_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh, cloud2_input_topic, 1);
        sync = new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10),
                                                             *cloud1_sub,
                                                             *cloud2_sub);
        sync->registerCallback(boost::bind(&projectClouds::callback, this, _1, _2));


        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(cloud2_output_topic, 1);

        result_str = readParam<std::string>(nh, "result_file");
        std::ifstream myReadFile(result_str.c_str());
        std::string word;
        int i = 0;
        int j = 0;

        C_T_L = S_T_C = Eigen::Matrix4d::Identity();
        S_T_C.block(0, 0, 3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0;
        while (myReadFile >> word){
            C_T_L(i, j) = atof(word.c_str());
            j++;
            if(j>3) {
                j = 0;
                i++;
            }
        }
        S_T_L = S_T_C*C_T_L;
    }

    template <typename T>
    T readParam(ros::NodeHandle &n, std::string name){
        T ans;
        if (n.getParam(name, ans)){
            ROS_INFO_STREAM("Loaded " << name << ": " << ans);
        } else {
            ROS_ERROR_STREAM("Failed to load " << name);
            n.shutdown();
        }
        return ans;
    }

    void callback(const sensor_msgs::PointCloud2ConstPtr &cloud1_msg,
                  const sensor_msgs::PointCloud2ConstPtr &cloud2_msg) {
        double time1 = cloud1_msg->header.stamp.toSec();
        double time2 = cloud1_msg->header.stamp.toSec();
        double time_diff = time1 - time2;
        if(fabs(time1-time2)<0.001) {
            std::string frame_id_ref = cloud1_msg->header.frame_id;
            pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud2(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud2(new pcl::PointCloud<pcl::PointXYZI>);

            in_cloud2->points.clear();
            transformed_cloud2->points.clear();

            pcl::fromROSMsg(*cloud2_msg, *in_cloud2);
            pcl::transformPointCloud(*in_cloud2, *transformed_cloud2, S_T_L);

            sensor_msgs::PointCloud2 out_cloud_ros;
            pcl::toROSMsg(*transformed_cloud2, out_cloud_ros);
            out_cloud_ros.header.frame_id = frame_id_ref;
            out_cloud_ros.header.stamp = cloud1_msg->header.stamp;
            cloud_pub.publish(out_cloud_ros);
        } else {
            ROS_WARN_STREAM("Time diff higher than threshold");
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lidar_stereo_pcd_proj_node");
    projectClouds pC;
    ros::spin();
    return 0;
}