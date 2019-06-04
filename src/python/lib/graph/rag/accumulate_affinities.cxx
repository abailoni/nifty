#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "xtensor-python/pytensor.hpp"

#include "nifty/graph/rag/grid_rag.hxx"
#include "nifty/graph/rag/feature_accumulation/grid_rag_affinity_features.hxx"
#include "nifty/graph/rag/feature_accumulation/lifted_nh.hxx"

#include "nifty/python/graph/undirected_grid_graph.hxx"
#include "nifty/python/graph/undirected_list_graph.hxx"

namespace py = pybind11;


namespace nifty{
namespace graph{

    using namespace py;

    template<class RAG, unsigned DIM>
    void exportAccumulateAffinityFeaturesT(
        py::module & ragModule
    ){
        ragModule.def("computeFeaturesAndNhFromAffinities",
        [](
            const RAG & rag,
            xt::pytensor<float, DIM+1> affinities,
            const std::vector<std::vector<int>> & offsets,
            const int numberOfThreads
        ){

            // TODO
            //{
            //    py::gil_scoped_release allowThreads;
            //}
            LiftedNh<RAG> lnh(
                rag, offsets.begin(), offsets.end(), numberOfThreads
            );

            const int64_t nLocal  = rag.numberOfEdges();
            int64_t nLifted = lnh.numberOfEdges();

            bool haveLifted = true;
            if(nLifted == 0) {
                nLifted += 1;
                haveLifted = false;
            }

            xt::pytensor<double, 2> outLocal({nLocal, int64_t(10)});
            xt::pytensor<double, 2> outLifted({nLifted, int64_t(10)});
            {
                py::gil_scoped_release allowThreads;
                accumulateLongRangeAffinities(rag, lnh, affinities, 0., 1., outLocal, outLifted, numberOfThreads);
            }
            xt::pytensor<int64_t, 2> lnhOut({nLifted, int64_t(2)});
            if(haveLifted){
                py::gil_scoped_release allowThreads;
                for(std::size_t e = 0; e < nLifted; ++e) {
                    lnhOut(e, 0) = lnh.u(e);
                    lnhOut(e, 1) = lnh.v(e);
                }
            } else {
                lnhOut(0, 0) = -1;
                lnhOut(0, 1) = -1;
            }
            return std::make_tuple(lnhOut, outLocal, outLifted);
        },
        py::arg("rag"),
        py::arg("affinities"),
        py::arg("offsets"),
        py::arg("numberOfThreads")= -1
        );

        ragModule.def("accumulateAffinityStandartFeatures",
        [](
            const RAG & rag,
            const xt::pytensor<float, DIM+1> & affinities,
            const std::vector<std::array<int, 3>> & offsets,
            const float min, const float max,
            const int numberOfThreads
        ){

            int64_t nEdges = rag.numberOfEdges();
            typename xt::pytensor<double, 2>::shape_type shape = {nEdges, int64_t(10)};
            xt::pytensor<double, 2> out(shape);
            {
                py::gil_scoped_release allowThreads;
                accumulateAffinities(rag, affinities, offsets, out, min, max, numberOfThreads);
            }
            return out;
        }, py::arg("rag"),
           py::arg("affinities"),
           py::arg("offsets"),
           py::arg("min")=0., py::arg("max")=1.,
           py::arg("numberOfThreads")=-1
        );
    }

    // -----------------------------------------------------------------------------------------------
    // Old version with additional features:
    //  - does not require a RAG but simply a graph and a label image (can include long-range edges)
    //  - performs weighted average

    // TODO: combine with polished version by Constantin (or move implementation to headers)
    template<std::size_t DIM, class GRAPH, class DATA_T>
    void exportAccumulateAffinitiesMeanAndLength(
            py::module & ragModule
    ) {
        ragModule.def("accumulateAffinitiesMeanAndLength",
                      [](
                              const GRAPH &graph,
                              // TODO: generalize for more label types:
                              xt::pytensor<int64_t, DIM> labels, // Labels less then zero are ignored
                              xt::pytensor<DATA_T, DIM + 1> affinities,
                              xt::pytensor<int, 2> offsets,
                              xt::pytensor<DATA_T, 1> affinities_weights,
                              const int numberOfThreads
                      ) {
                          array::StaticArray<uint64_t, DIM> shape;

//                          std::array<int,DIM> shape;
                          // Check inputs:
                          for(auto d=0; d<DIM; ++d){
                              shape[d] = labels.shape()[d];
                              NIFTY_CHECK_OP(shape[d],==,affinities.shape()[d], "affinities have wrong shape");
                          }
                          NIFTY_CHECK_OP(offsets.shape()[0],==,affinities.shape()[DIM], "Affinities and offsets do not match");
                          NIFTY_CHECK_OP(offsets.shape()[0],==,affinities_weights.shape()[0], "Affinities weights and offsets do not match");


                          // Create thread pool:
                          nifty::parallel::ParallelOptions pOpts(numberOfThreads);
                          nifty::parallel::ThreadPool threadpool(pOpts);
                          const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();

                          // Create temporary arrays
                          const auto nb_edges = uint64_t(graph.edgeIdUpperBound()+1);
                          xt::pytensor<DATA_T, 2> accAff = xt::zeros<DATA_T>({actualNumberOfThreads, nb_edges});
                          xt::pytensor<DATA_T, 2> maxAff = xt::zeros<DATA_T>({actualNumberOfThreads, nb_edges});
                          xt::pytensor<DATA_T, 2> counter = xt::zeros<DATA_T>({actualNumberOfThreads, nb_edges});
                          {
                              py::gil_scoped_release allowThreads;
                              nifty::tools::parallelForEachCoordinate(threadpool,
                                                                      shape,
                                                                      [&](const auto threadId, const auto & coordP){
                                                                          const auto u = labels[coordP];
                                                                          for(auto i=0; i<offsets.shape()[0]; ++i){
                                                                              auto coordQ = coordP;
                                                                              for (auto d=0; d<DIM; ++d) {
                                                                                  coordQ[d] += offsets(i,d);
                                                                              }
                                                                              if(coordQ.allInsideShape(shape)){
                                                                                  const auto v = labels[coordQ];
                                                                                  if(u != v && u > 0 && v > 0){
                                                                                      const auto edge = graph.findEdge(u,v);
                                                                                      if (edge >=0 ){
                                                                                          array::StaticArray<int64_t, DIM + 1> affIndex;
                                                                                          std::copy(std::begin(coordP), std::end(coordP), std::begin(affIndex));
                                                                                          affIndex[DIM] = i;
                                                                                          const auto aff_value = affinities[affIndex];
                                                                                          if (aff_value > maxAff(threadId, edge))
                                                                                              maxAff(threadId, edge) = aff_value;
                                                                                          counter(threadId,edge) += affinities_weights(i);
                                                                                          accAff(threadId,edge) += aff_value*affinities_weights(i);
                                                                                      }
                                                                                  }
                                                                              }
                                                                          }
                                                                      });
                          }

                          // Create outputs:
                          typedef xt::pytensor<DATA_T, 1> OutArrayType;
                          typedef std::tuple<OutArrayType, OutArrayType, OutArrayType>  OutType;
                          OutArrayType accAff_out = xt::zeros<DATA_T>({nb_edges});
                          OutArrayType counter_out = xt::zeros<DATA_T>({nb_edges});
                          OutArrayType maxAff_out = xt::zeros<DATA_T>({nb_edges});

                          // Normalize:
                          for(auto i=0; i<nb_edges; ++i){
                              maxAff_out(i) = maxAff(0,i);
                              for(auto thr=1; thr<numberOfThreads; ++thr){
                                  counter(0,i) += counter(thr,i);
                                  accAff(0,i) += accAff(thr,i);
                                  if (maxAff(thr,i) > maxAff_out(i))
                                      maxAff_out(i) = maxAff(thr,i);
                              }
                              if(counter(0,i)>0.5){
                                  accAff_out(i) = accAff(0,i) / counter(0,i);
                                  counter_out(i) = counter(0,i);
                              } else {
                                  accAff_out(i) = 0.;
                                  counter_out(i) = 0.;
                              }
                          }
                          return OutType(accAff_out, counter_out, maxAff_out);;
                      },
                      py::arg("graph"),
                      py::arg("labels"),
                      py::arg("affinities"),
                      py::arg("offsets"),
                      py::arg("affinitiesWeights"),
                      py::arg("numberOfThreads") = -1

        );
    }


    void exportAccumulateAffinityFeatures(py::module & ragModule) {
        typedef xt::pytensor<uint32_t, 3> ExplicitPyLabels3D;
        typedef GridRag<3, ExplicitPyLabels3D> Rag3d;
        exportAccumulateAffinityFeaturesT<Rag3d, 3>(ragModule);

        typedef PyUndirectedGraph GraphType;
        typedef PyUndirectedGraph GraphType;
        exportAccumulateAffinitiesMeanAndLength<3, GraphType, float>(ragModule);
    }

}
}
