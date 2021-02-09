#pragma once
#include <unordered_map>


#include "nifty/xtensor/xtensor.hxx"
#include "nifty/parallel/threadpool.hxx"
#include "nifty/tools/for_each_coordinate.hxx"

namespace nifty{
namespace tools{

    template<class ARRAY>
    void makeDense(xt::xexpression<ARRAY> & dataExp){
        typedef typename ARRAY::value_type DataType;
        auto & data = dataExp.derived_cast();

        std::unordered_map<DataType, DataType> hmap;
        for(auto s=0; s<data.size(); ++s){
            const auto val = data(s);
            hmap[val] = DataType();
        }
        DataType c = 0;
        for(auto & kv : hmap){
            kv.second = c;
            ++c;
        }
        for(auto s=0; s<data.size(); ++s){
            const auto val = data(s);
            data(s) = hmap[val];
        }
    }

    template<class ARRAY1, class ARRAY2>
    void makeDense(
        const xt::xexpression<ARRAY1> & dataInExp,
        xt::xexpression<ARRAY2> & dataOutExp
    ){
        const auto & dataIn = dataInExp.derived_cast();
        auto & dataOut = dataOutExp.derived_cast();
        for(auto s=0; s<dataIn.size(); ++s){
            dataOut(s) = dataIn(s);
        }
        makeDense(dataOut);
    }


    template<class ARRAY1, class ARRAY2>
    int fromAdjMatrixToEdgeList(
            const xt::xexpression<ARRAY1> & adjMatrixExp,
            xt::xexpression<ARRAY2> & edgeOutExp,
            const int numberOfThreads
    ){
        const auto & adjMatrix = adjMatrixExp.derived_cast();
        auto & edgeOut = edgeOutExp.derived_cast();


        // nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        // nifty::parallel::ThreadPool threadpool(pOpts);
        // const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();
        int counter = 0;
        typedef array::StaticArray<int64_t, 2>  ShapeType;
        ShapeType shape;
        for(auto d=0; d<2; ++d){
            shape[d] = adjMatrix.shape()[d];
        }
        nifty::tools::forEachCoordinate(shape,
                                                [&](const auto & coordP) {
            //  TODO: generalize, atm copies half of the matrix and ignores diagonal (and zeros)

            if (adjMatrix[coordP] > 0. || adjMatrix[coordP] < 0.) {
                // Only check on one side of the matrix:
                    if (coordP[0] > coordP[1] ) {
                        edgeOut(counter, 0) = coordP[0];
                        edgeOut(counter, 1) = coordP[1];
                        edgeOut(counter, 2) = adjMatrix[coordP];
                        counter++;
                    }
                }
        });
        return counter;
    }

    template<class ARRAY1, class ARRAY2>
    void fromEdgeListToAdjMatrix(
            xt::xexpression<ARRAY1> & A_p_Exp,
            xt::xexpression<ARRAY1> & A_n_Exp,
            const xt::xexpression<ARRAY2> & edgeListExp,
            const int numberOfThreads
    ){
        auto & A_p = A_p_Exp.derived_cast();
        auto & A_n = A_n_Exp.derived_cast();
        const auto & edgeList = edgeListExp.derived_cast();

        //  TODO: generalize, atm creates symmetric (positive) matrices (negative edge weights in A_n and positive ones in A_p)
        // nifty::parallel::ParallelOptions pOpts(numberOfThreads);
        // nifty::parallel::ThreadPool threadpool(pOpts);
        // const std::size_t actualNumberOfThreads = pOpts.getActualNumThreads();
        // int counter = 0;
        const auto nb_edges = edgeList.shape()[0];
        for(auto s=0; s<edgeList.shape()[0]; ++s){
            if (edgeList(s,2)>0.) {
                A_p(size_t(edgeList(s,0)), size_t(edgeList(s,1))) = edgeList(s,2);
                A_p(size_t(edgeList(s,1)), size_t(edgeList(s,0))) = edgeList(s,2);
            } else if (edgeList(s,2)<0.) {
                A_n(size_t(edgeList(s,0)), size_t(edgeList(s,1))) = -edgeList(s,2);
                A_n(size_t(edgeList(s,1)), size_t(edgeList(s,0))) = -edgeList(s,2);
            }
        }
        const auto nb_nodes = A_p.shape()[0];
        for(auto n=0; n<nb_nodes; ++n){
            A_p(n,n) = 1.;
            // A_n(n,n) = 1.;
        }
    }


}
}

