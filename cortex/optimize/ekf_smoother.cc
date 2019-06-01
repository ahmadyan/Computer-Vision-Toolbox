#include <Eigen/Core>
#include <ceres/ceres.h>
#include <iostream>
#include <stdio.h>
#include <deque>
#include <sophus/se3.hpp>

#include "../geometry/rotation.h"


class EKF {
public:
	EKF(const double acc_cov, const double gyro_cov, const double mag_cov, const double process_cov)
	: acc_cov_(acc_cov), gyro_cov_(gyro_cov), magnet_cov_(mag_cov), process_cov_(process_cov) ,
	    Q (Eigen::Matrix<double,12,12>::Identity() * process_cov),
	    R_acc(Eigen::Matrix<double,3,3>::Identity() * acc_cov),
	    R_gyro(Eigen::Matrix<double,3,3>::Identity() * gyro_cov),
	    R_mag(Eigen::Matrix<double,3,3>::Identity() * mag_cov) {

        x.setZero();
        initialized_ = false;

        R_imu.setZero();
        R_imu.block<3, 3>(0, 0) = R_gyro;
        R_imu.block<3, 3>(3, 3) = R_acc;
        R_imu.block<3, 3>(6, 6) = R_mag;

        P.setZero();
        P.block<3, 3>(0, 0) = R_gyro;
        P.block<3, 3>(3, 3) = R_gyro;
        P.block<3, 3>(6, 6) = R_acc;
        P.block<3, 3>(9, 9) = R_mag;
    }

    EKF(EKF&) = default;
	~EKF() = default;

    // a priori prediction
	void Predict(double t){
        const double dt = t - time;
        Eigen::Vector3d w = x.head(3);             // Gyro
        Eigen::Vector3d wa = x.segment<3>(3);      // Accel
        Eigen::Vector3d ra = x.segment<3>(6);      // Gravity
        Eigen::Vector3d rm = x.tail(3);            // Magnet

        // Predict the state estimate x_k+1 = f(x_k)
        x.segment<3>(0) += wa*dt;
        x.segment<3>(6) += -w.cross(ra)*dt;
        x.segment<3>(9) += -w.cross(rm)*dt;

        // Compute state transition A_k at the linearization point
        Eigen::Matrix<double, 12, 12> A = Eigen::Matrix<double, 12, 12>::Identity(12, 12);
        A.block<3, 3>(0, 3) += Eigen::Matrix3d::Identity() * dt;
       	A.block<3, 3>(6, 6) += hat(w)*-dt;
	    A.block<3, 3>(9, 9) += hat(w)*-dt;
	    A.block<3, 3>(6, 0) += hat(ra)*dt;
	    A.block<3, 3>(9, 0) += hat(rm)*dt;

        // Project the error covariance: P_k+1 = A_k P_k A_k' + Q_k
        P = A * P * A.transpose() + Q;
        time = t;
    }

    void Update_magnetic(const Eigen::Vector3d& mag, double t) {
        Predict(t);

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 12);
        Eigen::VectorXd z = mag;
        H.block<3, 3>(0, 9) = Eigen::Matrix3d::Identity();

        Eigen::MatrixXd Pi = P.inverse();
        Eigen::MatrixXd K = Pi * H.transpose() * (H * Pi * H.transpose() + R_mag).inverse();
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(12, 12);

        x = x + K*(z - H*x);
        P = (I- K*H)*Pi;
    }

    void Initialize(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, const Eigen::Vector3d& magnet, double t) {
        time = t;
        x << gyro, Eigen::Vector3d::Zero(), acc, magnet;
        initialized_ = true;
    }

    void Update(const Eigen::Vector3d&gyro, const Eigen::Vector3d& acc, const Eigen::Vector3d& magnet, double t) {
        if(!initialized_) {
            Initialize(gyro, acc, magnet, t);
            return;
        }

        Predict(t);

        Eigen::Matrix<double, 9, 12> H = Eigen::Matrix<double, 9, 12>::Zero();
        Eigen::Matrix<double, 9, 1> z;
        z << gyro, acc, magnet;

        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(6, 9) = Eigen::Matrix3d::Identity();  // Magnet

        Eigen::Matrix<double, 12, 12> Pi = P.inverse();

        // Compute Kalman gain (P H' S^-1), where S (residual covariance) is HPH'+R
        Eigen::MatrixXd K = Pi * H.transpose() * (H * Pi * H.transpose() + R_imu).inverse();

        // Update the state and covariance estimate
        x = x + K * (z - H * x);
        P = (Eigen::Matrix<double,12,12>::Identity() - K * H) * Pi;
    }

	Eigen::Quaterniond Attitude(){
        if (!initialized_) {
            return Eigen::Quaterniond(1.f, 0.f, 0.f, 0.f);
        }

        Eigen::Matrix3d rotation;

        Eigen::Vector3d z_dir = x.segment<3>(6);              // Gravity
        Eigen::Vector3d y_dir = z_dir.cross(x.segment<3>(9)); // Magnet
        Eigen::Vector3d x_dir = y_dir.cross(z_dir);

        // Make sure rotation matrix is normalized
        rotation.row(0) = x_dir.normalized();
        rotation.row(1) = y_dir.normalized();
        rotation.row(2) = z_dir.normalized();

        return rotation_matrix_to_quaternion(rotation);
    }

private:
	double time;
	const double acc_cov_, gyro_cov_, magnet_cov_, process_cov_;

	/// @brief the state of the filter, arranged as anguler velocity, angular acceleration velocity, gravity field, magnetic field
	Eigen::Matrix<double, 12, 1> x;

	/// @brief Covariance estimate
	Eigen::Matrix<double, 12, 12> P;

	const Eigen::Matrix<double, 12, 12> Q;
	const Eigen::Matrix<double, 3, 3> R_acc, R_gyro, R_mag;
	Eigen::Matrix<double, 9, 9> R_imu;

	bool initialized_;
};

extern "C" void* setup(double* initial_covariance) {
    const double acc_cov = initial_covariance[0];
    const double gyro_cov = initial_covariance[1];
    const double mag_cov = initial_covariance[2];
    const double process_cov = initial_covariance[3];

    EKF* ekf = new EKF(acc_cov, gyro_cov, mag_cov, process_cov);
    return (void*) ekf;
}

extern "C" void release(void* ekf_ptr) {
    EKF* ekf = reinterpret_cast<EKF*>(ekf_ptr);
    delete ekf;
}

extern "C" int update(void* ekf_ptr, double* gyro, double* acc, double* magnet, double timestamp) {
    try {
        EKF* ekf = reinterpret_cast<EKF*>(ekf_ptr);
        ekf->Update(Eigen::Map<Eigen::Vector3d>(gyro),
                    Eigen::Map<Eigen::Vector3d>(acc),
                    Eigen::Map<Eigen::Vector3d>(magnet),
                    timestamp);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int attitude(void* ptr, double* state) {
    try {
        EKF* ekf = reinterpret_cast<EKF*>(ptr);
        Eigen::Map<Eigen::Quaterniond> q(state);
        q = ekf->Attitude();
        return 0;
    } catch (...) {
        return 1;
    }
}
