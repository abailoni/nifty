#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <algorithm>
#include <iostream>

//#include "xtensor/xexpression.hpp"
//#include "xtensor/xview.hpp"
//#include "xtensor/xmath.hpp"
//#include "xtensor-python/pyarray.hpp"
//#include "xtensor-python/pytensor.hpp"
//#include "xtensor-python/pyvectorize.hpp"

#include "nifty/tools/for_each_coordinate.hxx"
#include "nifty/parallel/threadpool.hxx"

#include "nifty/python/converter.hxx"


#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"
#include "nifty/xtensor/xtensor.hxx"


#ifdef WITH_HDF5

#include "nifty/hdf5/hdf5_array.hxx"
#include "nifty/graph/rag/grid_rag_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_stacked_2d_hdf5.hxx"
#include "nifty/graph/rag/grid_rag_labels_hdf5.hxx"
#endif


#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/grid_rag_accumulate.hxx"
#include "nifty/ufd/ufd.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"
#include "nifty/python/graph/edge_contraction_graph.hxx"



namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;


    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeMeanAndLength",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){

            typename xt::pytensor<DATA_T, 2>::shape_type shape = {int64_t(rag.edgeIdUpperBound()+1), int64_t(2)};
            xt::pytensor<DATA_T, 2> out(shape);
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeMeanAndLength(rag, data, blockShape, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag").noconvert(),
        py::arg("data").noconvert(),
        py::arg("blockShape")=array::StaticArray<int64_t, DIM>(100),
        py::arg("numberOfThreads")=-1
        );
    }


    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateGeometricEdgeFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateGeometricEdgeFeatures",
        [](
            const RAG & rag,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){

            xt::pytensor<DATA_T, 2> out({int64_t(rag.edgeIdUpperBound()+1), int64_t(17)});
            {
                py::gil_scoped_release allowThreads;
                accumulateGeometricEdgeFeatures(rag, blockShape, out, numberOfThreads);
            }
            return out;
        },
        py::arg("rag"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }


    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateMeanAndLength(
        py::module & ragModule
    ){
        ragModule.def("accumulateMeanAndLength",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads,
            const bool saveMemory
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(2)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                accumulateMeanAndLength(rag, data, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1,
        py::arg_t<bool>("saveMemory",false)
        );
    }

    #ifdef WITH_HDF5
    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateMeanAndLengthHdf5(
        py::module & ragModule
    ){
        ragModule.def("accumulateMeanAndLength",
        [](
            const RAG & rag,
            const nifty::hdf5::Hdf5Array<DATA_T> & data,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads,
            const bool saveMemory
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(2)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(2)});
            {
                py::gil_scoped_release allowThreads;
                accumulateMeanAndLength(rag, data, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);;
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1,
        py::arg_t<bool>("saveMemory",false)
        );
    }
    #endif




    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateStandartFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateStandartFeatures",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(9)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateStandartFeatures(rag, data, minVal, maxVal, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }

    #ifdef WITH_HDF5
    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateStandartFeaturesHdf5(
        py::module & ragModule
    ){
        ragModule.def("accumulateStandartFeatures",
        [](
            const RAG & rag,
            const nifty::hdf5::Hdf5Array<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2> edgeOut({int64_t(rag.edgeIdUpperBound()+1), int64_t(9)});
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateStandartFeatures(rag, data, minVal, maxVal, blockShape, edgeOut, nodeOut, numberOfThreads);
            }
            return std::make_pair(edgeOut, nodeOut);
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }

    #endif




    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateNodeStandartFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateNodeStandartFeatures",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2>nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(9)});
            {
                py::gil_scoped_release allowThreads;
                accumulateNodeStandartFeatures(rag, data, minVal, maxVal, blockShape, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }

    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateEdgeStandartFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateEdgeStandartFeatures",
        [](
            const RAG & rag,
            const xt::pyarray<DATA_T> & data,
            const double minVal,
            const double maxVal,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2>edgeOut({int64_t(rag.edgeIdUpperBound()+1), 9L});
            {
                py::gil_scoped_release allowThreads;
                accumulateEdgeStandartFeatures(rag, data, minVal, maxVal, blockShape, edgeOut, numberOfThreads);
            }
            return edgeOut;
        },
        py::arg("rag"),
        py::arg("data").noconvert(),
        py::arg("minVal"),
        py::arg("maxVal"),
        py::arg("blockShape") = array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")= -1
        );
    }



    template<std::size_t DIM, class RAG, class DATA_T>
    void exportAccumulateGeometricNodeFeatures(
        py::module & ragModule
    ){
        ragModule.def("accumulateGeometricNodeFeatures",
        [](
            const RAG & rag,
            array::StaticArray<int64_t, DIM> blockShape,
            const int numberOfThreads
        ){
            xt::pytensor<DATA_T, 2> nodeOut({int64_t(rag.nodeIdUpperBound()+1), int64_t(3*DIM+1)});
            {
                py::gil_scoped_release allowThreads;
                accumulateGeometricNodeFeatures(rag, blockShape, nodeOut, numberOfThreads);
            }
            return nodeOut;
        },
        py::arg("rag"),
        py::arg("blockShape")=array::StaticArray<int64_t,DIM>(100),
        py::arg("numberOfThreads")=-1
        );
    }



    void exportAccumulate(py::module & ragModule) {

        //explicit
        {

            typedef xt::pytensor<uint32_t, 2> ExplicitLabels2D;
            typedef GridRag<2, ExplicitLabels2D> Rag2d;
            typedef xt::pytensor<uint32_t, 3> ExplicitLabels3D;
            typedef GridRag<3, ExplicitLabels3D> Rag3d;


            exportAccumulateEdgeMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeMeanAndLength<3, Rag3d, float>(ragModule);

            exportAccumulateMeanAndLength<2, Rag2d, float>(ragModule);
            exportAccumulateMeanAndLength<3, Rag3d, float>(ragModule);

            exportAccumulateStandartFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateStandartFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateNodeStandartFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateNodeStandartFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateEdgeStandartFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateEdgeStandartFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateGeometricNodeFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateGeometricNodeFeatures<3, Rag3d, float>(ragModule);

            exportAccumulateGeometricEdgeFeatures<2, Rag2d, float>(ragModule);
            exportAccumulateGeometricEdgeFeatures<3, Rag3d, float>(ragModule);

            #ifdef WITH_HDF5
            typedef nifty::hdf5::Hdf5Array<uint32_t> H5Labels;
            typedef GridRag<3, H5Labels> RagH53d;
            exportAccumulateMeanAndLengthHdf5<3,RagH53d, float>(ragModule);
            exportAccumulateStandartFeaturesHdf5<3, RagH53d, uint8_t>(ragModule);
            #endif

        }
    }

} // end namespace graph
} // end namespace nifty
