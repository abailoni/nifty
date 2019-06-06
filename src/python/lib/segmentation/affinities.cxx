#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "nifty/segmentation/affinities.hxx"

namespace py = pybind11;

namespace nifty {
    namespace segmentation {


        template<typename T>
        void export_affinities_T(py::module & m) {
            m.def("compute_affinities", [](const xt::pyarray<T> & labels,
                                           const std::vector<std::vector<int>> & offsets,
                                           const bool have_ignore_label,
                                           const T ignore_label) {
                    // compute the out shape
                    typedef typename xt::pyarray<float>::shape_type ShapeType;
                    const auto & shape = labels.shape();
                    const unsigned ndim = labels.dimension();
                    ShapeType out_shape(ndim + 1);
                    out_shape[0] = offsets.size();
                    for(unsigned d = 0; d < ndim; ++d) {
                        out_shape[d + 1] = shape[d];
                    }

                    // allocate the output
                    xt::pyarray<float> affs = xt::zeros<float>(out_shape);
                    xt::pyarray<uint8_t> mask = xt::zeros<uint8_t>(out_shape);
                    {
                        py::gil_scoped_release allowThreads;
                        compute_affinities(labels, offsets,
                                                       affs, mask,
                                                       have_ignore_label, ignore_label);
                    }
                    return std::make_pair(affs, mask);
                }, py::arg("labels").noconvert(),
                   py::arg("offset"),
                   py::arg("have_ignore_label")=false,
                   py::arg("ignore_label")=0);
        }

        void exportAffinities(py::module & m) {
            export_affinities_T<bool>(m);
            export_affinities_T<uint64_t>(m);
            export_affinities_T<int64_t>(m);
        }

    }
}

