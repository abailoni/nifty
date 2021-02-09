#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <typeinfo> // to debug atm

#include "xtensor-python/pyarray.hpp"
#include "nifty/tools/make_dense.hxx"

namespace py = pybind11;



namespace nifty{
namespace tools{

    template<class T>
    void exportMakeDenseT(py::module & toolsModule) {

        toolsModule.def("makeDense",
        [](const xt::pyarray<T> & dataIn){

            typedef typename xt::pyarray<T>::shape_type ShapeType;
            ShapeType shape(dataIn.shape().begin(), dataIn.shape().end());
            xt::pyarray<T> dataOut(shape);
            {
                py::gil_scoped_release allowThreads;
                tools::makeDense(dataIn, dataOut);
            }
            return dataOut;
        });

    }

    template<class T>
    void exportFromAdjMatrixToEdgeListT(py::module & toolsModule) {
        toolsModule.def("fromAdjMatrixToEdgeList",
                        [](const xt::pyarray<T> & adjMatrix,
                           xt::pyarray<T> & edgeOut,
                           const int nbThreads
                                ){
                            int nb_edges;
                            {
                                py::gil_scoped_release allowThreads;
                                nb_edges = tools::fromAdjMatrixToEdgeList(adjMatrix, edgeOut, nbThreads);
                            }
                            return nb_edges;
                        });
        toolsModule.def("fromEdgeListToAdjMatrix",
                        [](xt::pyarray<T> & A_p,
                           xt::pyarray<T> & A_n,
                           const xt::pyarray<T> & edgeList,
                           const int nbThreads
                        ){
                            {
                                py::gil_scoped_release allowThreads;
                                tools::fromEdgeListToAdjMatrix(A_p, A_n, edgeList, nbThreads);
                            }
                        });

    }


    void exportMakeDense(py::module & toolsModule) {

        exportMakeDenseT<uint32_t>(toolsModule);
        exportMakeDenseT<uint64_t>(toolsModule);
        exportMakeDenseT<int32_t>(toolsModule);


        exportFromAdjMatrixToEdgeListT<int32_t>(toolsModule);
        exportFromAdjMatrixToEdgeListT<int64_t>(toolsModule);
        exportFromAdjMatrixToEdgeListT<double_t>(toolsModule);
        exportFromAdjMatrixToEdgeListT<float_t>(toolsModule);

        //exportMakeDenseT<float   , false>(toolsModule);
        exportMakeDenseT<int64_t>(toolsModule);
    }

}
}
