######################################################################
# Automatically generated by qmake (2.01a) Fri Jul 24 21:08:11 2015
######################################################################

TEMPLATE = app
TARGET = 
DEPENDPATH += . \
              g2o/core \
              g2o/stuff \
              g2o/solvers/cholmod \
              g2o/solvers/dense \
              g2o/types/sba \
              g2o/types/sim3 \
              g2o/types/slam3d
INCLUDEPATH += . \
               g2o/core \
               g2o/stuff \
               g2o/solvers/cholmod \
               g2o/solvers/dense \
               g2o/types/sba \
               g2o/types/slam3d \
               g2o/types/sim3

# Input
HEADERS += config.h \
           g2o/core/base_binary_edge.h \
           g2o/core/base_binary_edge.hpp \
           g2o/core/base_edge.h \
           g2o/core/base_multi_edge.h \
           g2o/core/base_multi_edge.hpp \
           g2o/core/base_unary_edge.h \
           g2o/core/base_unary_edge.hpp \
           g2o/core/base_vertex.h \
           g2o/core/base_vertex.hpp \
           g2o/core/batch_stats.h \
           g2o/core/block_solver.h \
           g2o/core/block_solver.hpp \
           g2o/core/cache.h \
           g2o/core/creators.h \
           g2o/core/estimate_propagator.h \
           g2o/core/factory.h \
           g2o/core/g2o_core_api.h \
           g2o/core/hyper_dijkstra.h \
           g2o/core/hyper_graph.h \
           g2o/core/hyper_graph_action.h \
           g2o/core/jacobian_workspace.h \
           g2o/core/linear_solver.h \
           g2o/core/marginal_covariance_cholesky.h \
           g2o/core/matrix_operations.h \
           g2o/core/matrix_structure.h \
           g2o/core/openmp_mutex.h \
           g2o/core/optimizable_graph.h \
           g2o/core/optimization_algorithm.h \
           g2o/core/optimization_algorithm_dogleg.h \
           g2o/core/optimization_algorithm_factory.h \
           g2o/core/optimization_algorithm_gauss_newton.h \
           g2o/core/optimization_algorithm_levenberg.h \
           g2o/core/optimization_algorithm_property.h \
           g2o/core/optimization_algorithm_with_hessian.h \
           g2o/core/parameter.h \
           g2o/core/parameter_container.h \
           g2o/core/robust_kernel.h \
           g2o/core/robust_kernel_factory.h \
           g2o/core/robust_kernel_impl.h \
           g2o/core/solver.h \
           g2o/core/sparse_block_matrix.h \
           g2o/core/sparse_block_matrix.hpp \
           g2o/core/sparse_block_matrix_ccs.h \
           g2o/core/sparse_block_matrix_diagonal.h \
           g2o/core/sparse_optimizer.h \
           g2o/core/sparse_optimizer_terminate_action.h \
           g2o/stuff/color_macros.h \
           g2o/stuff/command_args.h \
           g2o/stuff/filesys_tools.h \
           g2o/stuff/g2o_stuff_api.h \
           g2o/stuff/macros.h \
           g2o/stuff/misc.h \
           g2o/stuff/opengl_primitives.h \
           g2o/stuff/opengl_wrapper.h \
           g2o/stuff/os_specific.h \
           g2o/stuff/property.h \
           g2o/stuff/sampler.h \
           g2o/stuff/scoped_pointer.h \
           g2o/stuff/sparse_helper.h \
           g2o/stuff/string_tools.h \
           g2o/stuff/tictoc.h \
           g2o/stuff/timeutil.h \
           g2o/stuff/unscented.h \
           g2o/solvers/cholmod/linear_solver_cholmod.h \
           g2o/solvers/dense/linear_solver_dense.h \
           g2o/types/sba/g2o_types_sba_api.h \
           g2o/types/sba/sbacam.h \
           g2o/types/sba/types_sba.h \
           g2o/types/sba/types_six_dof_expmap.h \
           g2o/types/sim3/sim3.h \
           g2o/types/sim3/types_seven_dof_expmap.h \
           g2o/types/slam3d/dquat2mat.h \
           g2o/types/slam3d/edge_se3.h \
           g2o/types/slam3d/edge_se3_offset.h \
           g2o/types/slam3d/edge_se3_pointxyz.h \
           g2o/types/slam3d/edge_se3_pointxyz_depth.h \
           g2o/types/slam3d/edge_se3_pointxyz_disparity.h \
           g2o/types/slam3d/edge_se3_prior.h \
           g2o/types/slam3d/g2o_types_slam3d_api.h \
           g2o/types/slam3d/isometry3d_gradients.h \
           g2o/types/slam3d/isometry3d_mappings.h \
           g2o/types/slam3d/parameter_camera.h \
           g2o/types/slam3d/parameter_se3_offset.h \
           g2o/types/slam3d/parameter_stereo_camera.h \
           g2o/types/slam3d/se3_ops.h \
           g2o/types/slam3d/se3_ops.hpp \
           g2o/types/slam3d/se3quat.h \
           g2o/types/slam3d/types_slam3d.h \
           g2o/types/slam3d/vertex_pointxyz.h \
           g2o/types/slam3d/vertex_se3.h \
           g2o/types/slam3d/dquat2mat_maxima_generated.cpp \
           g2o/types/slam3d/dquat2mat.cpp
SOURCES += g2o/core/batch_stats.cpp \
           g2o/core/cache.cpp \
           g2o/core/estimate_propagator.cpp \
           g2o/core/factory.cpp \
           g2o/core/hyper_dijkstra.cpp \
           g2o/core/hyper_graph.cpp \
           g2o/core/hyper_graph_action.cpp \
           g2o/core/jacobian_workspace.cpp \
           g2o/core/marginal_covariance_cholesky.cpp \
           g2o/core/matrix_structure.cpp \
           g2o/core/optimizable_graph.cpp \
           g2o/core/optimization_algorithm.cpp \
           g2o/core/optimization_algorithm_dogleg.cpp \
           g2o/core/optimization_algorithm_factory.cpp \
           g2o/core/optimization_algorithm_gauss_newton.cpp \
           g2o/core/optimization_algorithm_levenberg.cpp \
           g2o/core/optimization_algorithm_with_hessian.cpp \
           g2o/core/parameter.cpp \
           g2o/core/parameter_container.cpp \
           g2o/core/robust_kernel.cpp \
           g2o/core/robust_kernel_factory.cpp \
           g2o/core/robust_kernel_impl.cpp \
           g2o/core/solver.cpp \
           g2o/core/sparse_block_matrix_test.cpp \
           g2o/core/sparse_optimizer.cpp \
           g2o/core/sparse_optimizer_terminate_action.cpp \
           g2o/stuff/command_args.cpp \
           g2o/stuff/filesys_tools.cpp \
           g2o/stuff/opengl_primitives.cpp \
           g2o/stuff/os_specific.c \
           g2o/stuff/property.cpp \
           g2o/stuff/sampler.cpp \
           g2o/stuff/sparse_helper.cpp \
           g2o/stuff/string_tools.cpp \
           g2o/stuff/tictoc.cpp \
           g2o/stuff/timeutil.cpp \
           g2o/solvers/cholmod/solver_cholmod.cpp \
           g2o/solvers/dense/solver_dense.cpp \
           g2o/types/sba/types_sba.cpp \
           g2o/types/sba/types_six_dof_expmap.cpp \
           g2o/types/sim3/types_seven_dof_expmap.cpp \
           g2o/types/slam3d/dquat2mat.cpp \
           g2o/types/slam3d/dquat2mat_maxima_generated.cpp \
           g2o/types/slam3d/edge_se3.cpp \
           g2o/types/slam3d/edge_se3_offset.cpp \
           g2o/types/slam3d/edge_se3_pointxyz.cpp \
           g2o/types/slam3d/edge_se3_pointxyz_depth.cpp \
           g2o/types/slam3d/edge_se3_pointxyz_disparity.cpp \
           g2o/types/slam3d/edge_se3_prior.cpp \
           g2o/types/slam3d/isometry3d_gradients.cpp \
           g2o/types/slam3d/isometry3d_mappings.cpp \
           g2o/types/slam3d/parameter_camera.cpp \
           g2o/types/slam3d/parameter_se3_offset.cpp \
           g2o/types/slam3d/parameter_stereo_camera.cpp \
           g2o/types/slam3d/types_slam3d.cpp \
           g2o/types/slam3d/vertex_pointxyz.cpp \
           g2o/types/slam3d/vertex_se3.cpp
