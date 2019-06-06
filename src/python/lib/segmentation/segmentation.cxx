#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"


namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);


namespace nifty{
    namespace segmentation{
        void exportAffinities(py::module &);
        void exportConnectedComponents(py::module &);
        void exportMutexWatershed(py::module &);
    }
}




PYBIND11_MODULE(_segmentation, segmentationModule) {

    xt::import_numpy();

    py::options options;
    options.disable_function_signatures();
    segmentationModule.doc() = "segmentation submodule of nifty";

    using namespace nifty::segmentation;

    exportAffinities(segmentationModule);
    exportConnectedComponents(segmentationModule);
    exportMutexWatershed(segmentationModule);
}
