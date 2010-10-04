#include "decision_stump.h"

#include <limits>
#include <vector>

using std::vector;

REGISTER_CLASSIFIER(mll::roizner::DecisionStump,
                    "DecisionStump",
                    "Decision-stump classifier");

namespace mll {
namespace roizner {

//! Choose the best class label for the weight sums on the one side of a threshold.
//! Returns the overall penalty for the selected class label
double SelectClassLabel(const vector<double>& classWeightSums,
                        const IMetaData& metaData,
                        int* classLabel,
                        vector<double>* confidences) {
    double minPenalty = std::numeric_limits<double>::max();
    double weightSum = 0;
    int classCount = metaData.GetClassCount();
    confidences->resize(classCount);
    for (int label = 0; label < classCount; ++label) {
        weightSum += classWeightSums[label];
    }
    for (int label = 0; label < classCount; ++label) {
        double penalty = 0;
        for (int label1 = 0; label1 < classCount; ++label1) {
            penalty += classWeightSums[label1] * metaData.GetPenalty(label1, label);
        }
        if (penalty < minPenalty) {
            minPenalty = penalty;
            *classLabel = label;
        }
        confidences->at(label) = classWeightSums[label] / weightSum;
    }
    return minPenalty;
}

void DecisionStump::Learn(IDataSet* data) {
    double minPenalty = std::numeric_limits<double>::max();
    vector<double> belowThresholdConfidences, aboveThresholdConfidences;
    // Iterating by the feature
    for (int featureIndex = 0; featureIndex < data->GetFeatureCount(); ++featureIndex) {
        vector<double> belowThresholdWeightSums(data->GetClassCount());
        vector<double> aboveThresholdWeightSums(data->GetClassCount());
        // Initializing weight sums
        for (int objectIndex = 0; objectIndex < data->GetObjectCount(); ++objectIndex) {
            aboveThresholdWeightSums[data->GetTarget(objectIndex)] += data->GetWeight(objectIndex);
        }
        // Sorting by the feature
        data->SortObjectsByFeature(featureIndex);
        // Choosing best threshold
        for (int objectIndex = 0; objectIndex < data->GetObjectCount(); ++objectIndex) {
            belowThresholdWeightSums[data->GetTarget(objectIndex)] += data->GetWeight(objectIndex);
            aboveThresholdWeightSums[data->GetTarget(objectIndex)] -= data->GetWeight(objectIndex);
            int belowThresholdClass, aboveThresholdClass;
            double penalty =
                SelectClassLabel(
                    belowThresholdWeightSums,
                    data->GetMetaData(),
                    &belowThresholdClass,
                    &belowThresholdConfidences) +
                SelectClassLabel(
                    aboveThresholdWeightSums,
                    data->GetMetaData(),
                    &aboveThresholdClass,
                    &aboveThresholdConfidences);
            if (penalty < minPenalty) {
                minPenalty = penalty;
                separatingFeatureIndex_ = featureIndex;
                belowThresholdClass_ = belowThresholdClass;
                aboveThresholdClass_ = aboveThresholdClass;
                belowThresholdConfidences_.assign(
                    belowThresholdConfidences.begin(),
                    belowThresholdConfidences.end());
                aboveThresholdConfidences_.assign(
                    aboveThresholdConfidences.begin(),
                    aboveThresholdConfidences.end());
                double feature = data->GetFeature(objectIndex, featureIndex);
                threshold_ = 
                    objectIndex + 1 < data->GetObjectCount()
                        ? (feature + data->GetFeature(objectIndex + 1, featureIndex)) / 2
                        : feature + 1.0;
            }
        }
    }
}

void DecisionStump::Classify(IDataSet* data) const {
    for (int i = 0; i < data->GetObjectCount(); ++i) {
        bool below = data->GetFeature(i, separatingFeatureIndex_) < threshold_;
        data->SetTarget(i, below ? belowThresholdClass_ : aboveThresholdClass_);
        for (int label = 0; label < data->GetClassCount(); ++label) {
            data->SetConfidence(
                i, label,
                below ? belowThresholdConfidences_.at(label)
                      : aboveThresholdConfidences_.at(label));
        }
    }
}

} // namespace roizner
} // namespace mll
