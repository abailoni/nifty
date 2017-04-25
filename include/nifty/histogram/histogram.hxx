#pragma once

#include <vector>
#include <array>

namespace nifty{
namespace histogram{


    template<class T, class BINCOUNT=float>
    class Histogram{
    public:
        typedef BINCOUNT BincountType;
        Histogram(
            const T minVal, 
            const T maxVal,
            const size_t bincount
        )
        :   counts_(bincount),
            minVal_(minVal),
            maxVal_(maxVal),
            sum_(0)
        {
        }


        const BincountType & operator[](const size_t i)const{
            return counts_[i];
        }
        size_t numberOfBins()const{
            return counts_.size();
        }
        BincountType sum()const{
            return sum_;
        }



        // insert     
        void insert(const T & value, const double w = 1.0){
            const auto b = this->fbin(value);
            const auto low  = std::floor(b);
            const auto high = std::ceil(b);

            // low and high are the same
            if(low + 0.5 >= high){
                counts_[size_t(low)] += w;
            }
            // low and high are different
            else{
                const auto wLow  = high - b;
                const auto wHigh = double(b) - low;

                counts_[size_t(low)]  += w*wLow;
                counts_[size_t(high)] += w*wHigh;
            }
            sum_ += w;
        }

        void normalize(){
            for(auto & v: counts_)
                v/=sum_;
            sum_ = 1.0;
        }

        void clear(){
            for(auto & v: counts_)
                v = 0;
            sum_ = 0.0;
        }

        double binToValue(const double fbin)const{
            return this->fbinToValue(fbin);
        }
    private:

        double fbinToValue(const double fbin){
            // todo
        }

        /**
         * @brief      get the floating point bin index
         *
         * @param[in]  val   value which to put in a bin
         *
         * @return     the floating point bin in [0,numberOfBins()-1]
         */
        float fbin(T val)const{
            // truncate
            val = std::max(minVal_, val);
            val = std::min(maxVal_, val);

            // normalize
            val -= minVal_;
            val /= (maxVal_ - minVal_);

            return val*(this->numberOfBins()-1);
        }


        std::vector<BincountType> counts_;
        T minVal_;
        T maxVal_;
        BincountType sum_;
    };



    template<class HISTOGRAM, size_t N>
    void quantiles(
        const HISTOGRAM & histogram,
        std::array<float, N>
    ){
    }


    template<class HISTOGRAM, class RANK_ITER, class OUT_ITER>
    void quantiles(
        const HISTOGRAM & histogram,
        RANK_ITER ranksBegin,
        RANK_ITER ranksEnd,
        OUT_ITER outIter
    ){

        const auto nQuantiels = std::distance(ranksBegin, ranksEnd);
        const auto s = histogram.sum();


        double csum = 0.0;
        auto qi = 0;
        for(auto bin=0; bin<histogram.numberOfBins(); ++bin){
            const double newcsum = csum  + histogram[bin];
            const auto  quant = ranksBegin[qi] * s;
            while(qi < nQuantiels && csum <= quant && newcsum >= quant ){
                if(bin == 0 ){
                    outIter[qi] = histogram.binToValue(0.0);
                }
                // linear interpolate the bin index    
                else{
                    const auto lbin  = double(bin - 1);
                    const auto hbin =  double(bin);
                    const auto m = histogram[bin];
                    const auto c = newcsum - hbin*m;
                    outIter[qi] = histogram.binToValue((quant - c)/m);
                }
                ++qi;
            }
            csum = newcsum;
        }
    }


    




}
}