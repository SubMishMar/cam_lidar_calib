//
// Created by usl on 4/9/19.
//

#ifndef CAM_LIDAR_CALIB_CALIBRATION_ERROR_TERM_H
#define CAM_LIDAR_CALIB_CALIBRATION_ERROR_TERM_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/autodiff_cost_function.h>
class CalibrationErrorTerm {
private:
    const Eigen::Vector3d laser_point_;
    const Eigen::Vector3d normal_to_plane_;

public:
    CalibrationErrorTerm(const Eigen::Vector3d& laser_point,
                         const Eigen::Vector3d& normal_to_plane):
                         laser_point_(laser_point), normal_to_plane_(normal_to_plane)
                         {}

    template  <typename  T>
    bool operator() (const T* const c_t_l_ptr,
                     const T* const c_q_l_ptr,
                     T* residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1> > c_t_l(c_t_l_ptr);
        Eigen::Map<const Eigen::Quaternion<T> > c_q_l(c_q_l_ptr);
        residual[0] = (normal_to_plane_.template cast<T>()).normalized().dot(c_q_l.normalized()
                *(laser_point_.template cast<T>()) + c_t_l) -  (normal_to_plane_.template cast<T>()).norm();
        return true;
    }
};
#endif //CAM_LIDAR_CALIB_CALIBRATION_ERROR_TERM_H
