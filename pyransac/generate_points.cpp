#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/eigen.h>


#include <primitives/PointCloud.h>
#include <primitives/RansacShapeDetector.h>
#include <primitives/PlanePrimitiveShapeConstructor.h>
#include <primitives/CylinderPrimitiveShapeConstructor.h>
#include <primitives/SpherePrimitiveShapeConstructor.h>
#include <primitives/ConePrimitiveShapeConstructor.h>
#include <primitives/TorusPrimitiveShapeConstructor.h>
#include<iostream>


#include <primitives/ConePrimitiveShape.h>
#include <primitives/SpherePrimitiveShape.h>
#include <primitives/TorusPrimitiveShape.h>
#include <primitives/PlanePrimitiveShape.h>
#include <primitives/CylinderPrimitiveShape.h>


// ----------------
// Python interface
// ----------------

namespace py = pybind11;

// wrap C++ function with NumPy array IO
py::dict py_fit(py::array_t<double, py::array::c_style | py::array::forcecast> points,
                    py::array_t<double, py::array::c_style | py::array::forcecast> normals,
                    float percentage,
                    int type,
                    float epsilon = 0.01f,          // distance threshold as fraction of bounding box
                    float bitmap_epsilon = 0.02f,   // bitmap resolution as fraction of bounding box
                    float normal_thresh = 0.6f,     // cos of max normal deviation
                    float probability = 0.5f,        // probability of overlooking a primitive
                    int min_points = 30            // minimum number of points for a primitive
)
{
    if ( points.ndim()     != 2 )
        throw std::runtime_error("Input should be 2-D NumPy array");
    if ( normals.ndim()     != 2 )
        throw std::runtime_error("Input normal should be 2-D NumPy array");
    int point_number = points.shape()[0] ;
    int normal_number = normals.shape()[0] ;
    if (point_number!=normal_number)
        throw std::runtime_error("Input size should be size");
    int point_dim = points.shape()[1] ;
    int normal_dim = normals.shape()[1] ;
    if (point_dim!=3)
        throw std::runtime_error("Input dim should be 3");
    if (normal_dim!=3)
        throw std::runtime_error("Input normal dim should be 3");

    PointCloud input;
    for(int i=0;i<point_number;i++) {
        input.push_back(Point(Vec3f(points.at(i, 0), points.at(i, 1), points.at(i, 2)))) ;
    }
    for(int i=0;i<point_number;i++) {
        input.at(i).normal =  Vec3f(normals.at(i, 0), normals.at(i, 1), normals.at(i, 2)) ;
    }
    input.setBBox(Vec3f(-1,-1,-1), Vec3f(1,1,1));

    RansacShapeDetector::Options ransacOptions;
    ransacOptions.m_epsilon = epsilon * input.getScale();
    // NOTE: Internally the distance threshold is taken as 3 * ransacOptions.m_epsilon!!!
    ransacOptions.m_bitmapEpsilon = bitmap_epsilon * input.getScale();
    ransacOptions.m_normalThresh = normal_thresh;
    ransacOptions.m_minSupport = std::max(min_points, int(point_number * percentage));
    ransacOptions.m_probability = probability;

    RansacShapeDetector detector(ransacOptions);
    if (type == 0)
        detector.Add(new PlanePrimitiveShapeConstructor());
    if (type == 1)
        detector.Add( new CylinderPrimitiveShapeConstructor()) ;
    if (type == 2)
        detector.Add( new ConePrimitiveShapeConstructor()) ;
    if (type == 3)
        detector.Add(new SpherePrimitiveShapeConstructor());
    if (type == 4)
        detector.Add( new TorusPrimitiveShapeConstructor()) ;

    MiscLib::Vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > shapes;
    size_t remaining = detector.Detect(input, 0, input.size(), &shapes); // run detection

    py::dict result;


    for(int i=0;i<shapes.size();i++)
    {
        std::string desc;
        shapes[i].first->Description(&desc);

        PrimitiveShape* currentshape =  &(*(shapes[i].first)) ;
        ConePrimitiveShape* coneptr = dynamic_cast<ConePrimitiveShape*>(currentshape);
        PlanePrimitiveShape* planeptr = dynamic_cast<PlanePrimitiveShape*>(currentshape);
        CylinderPrimitiveShape* cylinderptr = dynamic_cast<CylinderPrimitiveShape*>(currentshape);
        TorusPrimitiveShape* torusptr = dynamic_cast<TorusPrimitiveShape*>(currentshape);
        SpherePrimitiveShape* sphereptr = dynamic_cast<SpherePrimitiveShape*>(currentshape);


        if(coneptr != nullptr)
        {
            std::cout << "parentPtr points to a Cone object." << std::endl;
            Cone cone = coneptr->Internal();
            Vec3f center = cone.Center() ;
            Vec3f axis = cone.AxisDirection() ;
            double angle = cone.Angle() ;
            result[py::str("cone_center" +  std::to_string(i))] =  Eigen::Vector3f(center.getValue()[0], center.getValue()[1], center.getValue()[2]) ;
            result[py::str("cone_axisDir" + std::to_string(i))] =  Eigen::Vector3f(axis.getValue()[0], axis.getValue()[1], axis.getValue()[2]) ;
            result[py::str("cone_angle" + std::to_string(i))] =  angle ;

            std::vector<float> error ;
            for(int pi = 0; pi < point_number; pi++)
            {
                error.push_back(coneptr->Distance(input[pi].pos)) ;
            }
            result[py::str("cone" + std::to_string(i))] = error ;
        }


        if(planeptr != nullptr)
        {
            std::cout << "parentPtr points to a Plane object." << std::endl;
            Plane plane = planeptr->Internal() ;
            Vec3f normal = plane.getNormal() ;
            Vec3f position = plane.getPosition() ;
            result[py::str("plane_normal" +  std::to_string(i))] =  Eigen::Vector3f(normal.getValue()[0], normal.getValue()[1], normal.getValue()[2]) ;
            result[py::str("plane_position" + std::to_string(i))] =  Eigen::Vector3f(position.getValue()[0], position.getValue()[1], position.getValue()[2]) ;

            std::vector<float> error ;
            for(int pi = 0; pi < point_number; pi++)
            {
                error.push_back(planeptr->Distance(input[pi].pos)) ;
            }
            result[py::str("plane" + std::to_string(i))] = error ;

        }


        if(cylinderptr != nullptr)
        {
            std::cout << "parentPtr points to a cylinderptr object." << std::endl;
            Cylinder cylinder = cylinderptr->Internal();
            Vec3f axis = cylinder.AxisDirection() ;
            Vec3f position = cylinder.AxisPosition() ;
            double radius = cylinder.Radius() ;
            result[py::str("cylinder_axis" +  std::to_string(i))] =  Eigen::Vector3f(axis.getValue()[0], axis.getValue()[1], axis.getValue()[2]) ;
            result[py::str("cylinder_position" + std::to_string(i))] =  Eigen::Vector3f(position.getValue()[0], position.getValue()[1], position.getValue()[2]) ;
            result[py::str("cylinder_radius" + std::to_string(i))] = radius ;

            std::vector<float> error ;
            for(int pi = 0; pi < point_number; pi++)
            {
                error.push_back(cylinderptr->Distance(input[pi].pos)) ;
            }
            result[py::str("cylinder" + std::to_string(i))] = error ;
            std::cout<<"finish fitting for " <<i<<std::endl;
        }
        if(torusptr != nullptr)
        {

            std::cout << "parentPtr points to a torusptr object." << std::endl;
            Torus torus = torusptr->Internal() ;

            double small_radius = torus.MinorRadius() ;
            double big_radius = torus.MajorRadius() ;
            Vec3f center = torus.Center() ;
            Vec3f normal = torus.AxisDirection() ;
            result[py::str("torus_normal" +  std::to_string(i))] =  Eigen::Vector3f(normal.getValue()[0], normal.getValue()[1], normal.getValue()[2]) ;
            result[py::str("torus_center" + std::to_string(i))] =  Eigen::Vector3f(center.getValue()[0], center.getValue()[1], center.getValue()[2]) ;
            result[py::str("torus_small_radius" + std::to_string(i))] = small_radius ;
            result[py::str("torus_big_radius" + std::to_string(i))] = big_radius ;

            std::vector<float> error ;
            for(int pi = 0; pi < point_number; pi++)
            {
                error.push_back(torusptr->Distance(input[pi].pos)) ;
            }
            result[py::str("torus" + std::to_string(i))] = error ;

        }

        if(sphereptr != nullptr)
        {
            std::cout << "parentPtr points to a sphereptr object." << std::endl;
            Sphere sph = sphereptr->Internal() ;
            double raidus = sph.Radius() ;
            Vec3f center = sph.Center() ;
            result[py::str("sphere_center" + std::to_string(i))] =  Eigen::Vector3f(center.getValue()[0], center.getValue()[1], center.getValue()[2]) ;
            result[py::str("sphere_radius" + std::to_string(i))] = raidus ;

            std::vector<float> error ;
            for(int pi = 0; pi < point_number; pi++)
            {
                error.push_back(sphereptr->Distance(input[pi].pos)) ;
            }
            result[py::str("sphere" + std::to_string(i))] = error ;

        }

        std::cout << "shape " << i << " consists of " << shapes[i].second << " points, it is a " << desc << std::endl;
    }

    // return 2-D NumPy array
    return result ;
}

namespace py = pybind11;
// wrap as Python module
PYBIND11_MODULE(fitpoints,m)
{
    m.doc() = "get fit result";
    m.def("py_fit", &py_fit, "Calculate the length of an array of vectors",
          py::arg("points"),
          py::arg("normals"), 
          py::arg("percentage"),
          py::arg("type"),
          py::arg("epsilon") = 0.01f,
          py::arg("bitmap_epsilon") = 0.02f,
          py::arg("normal_thresh") = 0.6f,
          py::arg("probability") = 0.5f,
          py::arg("min_points") = 30
    );
    m.def("py_project", &py_fit, "Calculate the length of an array of vectors");
}
