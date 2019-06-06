#include "boost/pending/disjoint_sets.hpp"
#include "xtensor/xtensor.hpp"
#include "nifty/tools/for_each_coordinate.hxx"

#include <boost/container/flat_set.hpp>
#include <functional>


namespace nifty {
namespace segmentation {


    template<class AFFS, class LABELS>
    inline size_t connected_components(const xt::xexpression<AFFS> & affinities_exp,
                                       xt::xexpression<LABELS> & labels_exp,
                                       const float threshold) {

        typedef typename LABELS::value_type LabelType;
        const auto & affs = affinities_exp.derived_cast();
        auto & labels = labels_exp.derived_cast();

        const auto & shape = labels.shape();
        const unsigned dim = shape.size();

        // create and initialise union find
        const size_t n_nodes = labels.size();
        std::vector<LabelType> rank(n_nodes);
        std::vector<LabelType> parent(n_nodes);
        boost::disjoint_sets<LabelType*, LabelType*> sets(&rank[0], &parent[0]);
        for(LabelType node_id = 0; node_id < n_nodes; ++node_id) {
            sets.make_set(node_id);
        }

        // First pass:
        // iterate over each coordinate and create new label at coordinate
        // or assign representative of the neighbor label
        LabelType current_label = 0;
        nifty::tools::forEachCoordinate(shape, [&](const xt::xindex & coord){

            // get the spatial part of the affinity coordinate
            xt::xindex aff_coord(affs.dimension());
            std::copy(coord.begin(), coord.end(), aff_coord.begin() + 1);

            // iterate over all neighbors with smaller coordiates
            // (corresponding to affinity neighbors) and collect the labels
            // if neighbors that are connected
            std::set<LabelType> ngb_labels;
            for(unsigned d = 0; d < dim; ++d) {
                xt::xindex ngb_coord = coord;
                ngb_coord[d] -= 1;
                // perform range check
                if(ngb_coord[d] < 0 || ngb_coord[d] >= shape[d]) {
                    continue;  // continue if out of range
                }
                // set the proper dimension in the affinity coordinate
                aff_coord[0] = d;
                // check if the neighbor is connected and appen its label if so
                if(affs[aff_coord] > threshold) {
                    ngb_labels.insert(labels[ngb_coord]);
                }
            }

            // check if we are connected to any of the neighbors
            // and if the neighbor labels need to be merged
            if(ngb_labels.size() == 0) {
                // no connection -> make new label @ current pixel
                labels[coord] = ++current_label;
            } else if (ngb_labels.size() == 1) {
                // only single label -> we assign its representative to the current pixel
                labels[coord] = sets.find_set(*ngb_labels.begin());
            } else {
                // multiple labels -> we merge them and assign representative to the current pixel
                std::vector<LabelType> tmp_labels(ngb_labels.begin(), ngb_labels.end());
                for(unsigned ii = 1; ii < tmp_labels.size(); ++ii) {
                    sets.link(tmp_labels[ii - 1], tmp_labels[ii]);
                }
                labels[coord] = sets.find_set(tmp_labels[0]);
            }
        });

        // Second pass:
        // Assign representative to each pixel
        nifty::tools::forEachCoordinate(shape, [&](const xt::xindex & coord){
            labels[coord] = sets.find_set(labels[coord]);
        });

        // FIXME this is not necessarily the correct max value !!!
        return current_label;
    }

    template<class EDGE_ARRAY, class WEIGHT_ARRAY, class NODE_ARRAY>
    void compute_single_linkage_clustering(const size_t number_of_labels,
                                           const xt::xexpression<EDGE_ARRAY> & uvs_exp,
                                           const xt::xexpression<EDGE_ARRAY> & mutex_uvs_exp,
                                           const xt::xexpression<WEIGHT_ARRAY> & weights_exp,
                                           const xt::xexpression<WEIGHT_ARRAY> & mutex_weights_exp,
                                           xt::xexpression<NODE_ARRAY> & node_labeling_exp) {

        // casts
        const auto & uvs = uvs_exp.derived_cast();
        const auto & mutex_uvs = mutex_uvs_exp.derived_cast();
        const auto & weights = weights_exp.derived_cast();
        const auto & mutex_weights = mutex_weights_exp.derived_cast();
        auto & node_labeling = node_labeling_exp.derived_cast();

        // make ufd
        std::vector<uint64_t> ranks(number_of_labels);
        std::vector<uint64_t> parents(number_of_labels);
        boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
        for(uint64_t label = 0; label < number_of_labels; ++label) {
            ufd.make_set(label);
        }

        // determine number of edge types
        const size_t num_edges = uvs.shape()[0];
        const size_t num_mutex = mutex_uvs.shape()[0];

        // argsort ALL edges
        // we sort in ascending order
        std::vector<size_t> indices(num_edges + num_mutex);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](const size_t a, const size_t b){
            const double val_a = (a < num_edges) ? weights(a) : -mutex_weights(a - num_edges);
            const double val_b = (b < num_edges) ? weights(b) : -mutex_weights(b - num_edges);
            return val_a > val_b;
        });


        // iterate over all edges
        int counter = 0;
        for(const size_t edge_id : indices) {
            counter++;

            // check whether this edge is mutex via the edge offset
            const bool is_mutex_edge = edge_id >= num_edges;

            if (is_mutex_edge) {
                break;
            }

            // find the edge-id or mutex id and the connected nodes
            const size_t id = is_mutex_edge ? edge_id - num_edges : edge_id;
            const uint64_t u = is_mutex_edge ? mutex_uvs(id, 0) : uvs(id, 0);
            const uint64_t v = is_mutex_edge ? mutex_uvs(id, 1) : uvs(id, 1);

            // find the current representatives
            uint64_t ru = ufd.find_set(u);
            uint64_t rv = ufd.find_set(v);

            // if the nodes are not connected yet, merge
            if(ru != rv) {
                // link the nodes and merge their mutex constraints
                ufd.link(u, v);
                // check  if we have to swap the roots
                if(ufd.find_set(ru) == rv) {
                    std::swap(ru, rv);
                }
            }

        }
        // get node labeling into output
        for(size_t label = 0; label < number_of_labels; ++label) {
            node_labeling[label] = ufd.find_set(label);
        }
    }

}
}
