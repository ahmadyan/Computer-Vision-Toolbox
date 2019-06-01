#include <Eigen/Core>
#include <ceres/ceres.h>
#include <iostream>
#include <stdio.h>
#include <deque>
#include <sophus/se3.hpp>

class CeresAdapter {
    public:
    CeresAdapter() {}
    ~CeresAdapter() {
        delete problem_;
    }
    ceres::Problem* problem_;
    std::deque<Sophus::SE3d> poses_;
};

extern "C" void* setup() {
    CeresAdapter* adapter = new CeresAdapter();
    adapter->problem_ = new ceres::Problem();
    return (void*) adapter;
}

extern "C" int add_poses(void* ptr, double* poses, int number_poses) {
    try {
        CeresAdapter* adapter = reinterpret_cast<CeresAdapter*>(ptr);
        for (int i = 0; i < number_poses; ++i) {
            Eigen::Map<const Eigen::Matrix<double, 7, 1>> pose(&poses[i * 7]);
            std::cout << i << " " << pose.transpose() << "\n";
        }
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int add_imu_residuals(void* ptr) {
    try {
        CeresAdapter* adapter = reinterpret_cast<CeresAdapter*>(ptr);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" int optimize(void* ptr) {
    try {
        CeresAdapter* adapter = reinterpret_cast<CeresAdapter*>(ptr);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" void release(void* adapter_pointer) {
    CeresAdapter* adapter = reinterpret_cast<CeresAdapter*>(adapter_pointer);
    delete adapter;
}
