#pragma once

#include <Eigen/Core>

template <typename T = double>
Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1>& v) {
	Eigen::Matrix<T, 3, 3> m;
	m << 0, -v(2), v(1),
		 v(2), 0,  -v(0),
		 -v(1), v(0), 0;
	return m;
}

// Eigen tend to use the double cover of quaternion when the rotation
// trace is negative. This hack will avoid that issue
template <typename T = double>
Eigen::Quaternion<T> rotation_matrix_to_quaternion(const Eigen::Matrix<T,3,3>& m){
    Eigen::Quaternion<T> q;
    T t = m.trace();
    t = sqrt(t + double(1.0));
    q.w() = 0.5 * t;
    t = 0.5/t;
    q.x() = (m.coeff(2,1) - m.coeff(1,2)) * t;
    q.y() = (m.coeff(0,2) - m.coeff(2,0)) * t;
    q.z() = (m.coeff(1,0) - m.coeff(0,1)) * t;
    return q;
}
