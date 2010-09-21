#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <strstream>

#include <tclap/CmdLine.h>

#include "classifier.h"
#include "cross_validation.h"
#include "dataset.h"
#include "factories.h"
#include "tester.h"
#include "logger.h"
#include "dataset_wrapper.h"

using std::cout;
using std::cerr;
using std::endl;
using std::list;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

using namespace mll;

void ListClassifiers() {
    cout << "Registered classifiers:" << endl;
    vector<ClassifierFactory::Entry> classifiers;
    ClassifierFactory::Instance().GetEntries(&classifiers);
    for (vector<ClassifierFactory::Entry>::const_iterator it = classifiers.begin();
         it != classifiers.end();
         ++it) {
        cout << "\t" << it->GetName() << ": " 
             << it->GetDescription() << endl;
    }
    cout << endl;
}

void ListTesters() {
    cout << "Registered testers:" << endl;
    vector<TesterFactory::Entry> testers;
    TesterFactory::Instance().GetEntries(&testers);
    for (vector<TesterFactory::Entry>::const_iterator it = testers.begin();
         it != testers.end();
         ++it) {
        cout << "\t" << it->GetName() << ": " 
             << it->GetDescription() << endl;
    }
    cout << endl;
}

void LoadDataSet(DataSet* dataSet, const string& dataFileName) {
    if (!dataSet->Load(dataFileName)) {
        throw std::logic_error("Could not load dataset");
    }
    cout << "Dataset '" << dataSet->GetName() << "' loaded: "
         << dataSet->GetFeatureCount() << " features, "
         << dataSet->GetObjectCount() << " objects, "
         << dataSet->GetClassCount() << " classes." << endl;
    cout << endl;
}

sh_ptr<IClassifier> CreateClassifier(const string& classifierName) {
    sh_ptr<IClassifier> classifier = ClassifierFactory::Instance().Create(classifierName);
    if (classifier.get() == NULL) {
        throw std::logic_error("No classifier with this name registered");
    }
    return classifier;
}

sh_ptr<ITester> CreateTester(const string& testerName) {
    sh_ptr<ITester> tester = TesterFactory::Instance().Create(testerName);
    if (tester.get() == NULL) {
        throw std::logic_error("No tester with this name registered");
    }
    cout << "Tester: " << testerName << endl;
    cout << "Parameters:" << endl;
    tester->PrintParameters(true, true);
    cout << endl;
    return tester;
}

void ReadIndexes(const string& filename, vector<int>* indexes) {
	ifstream input(filename.c_str());
	if (!input.is_open()) {
		throw std::invalid_argument("File doesn't exist!");
	}
    indexes->clear();
	int index;
	while (input >> index) {
		indexes->push_back(index);
	}
	input.close();
}

sh_ptr<IDataSet> GetDataSet(const IDataSet& originalDataSet, const vector<int>& indexes) {
    DataSetWrapper* wrapper = new DataSetWrapper(&originalDataSet);
    wrapper->SetObjectIndexes(indexes.begin(), indexes.end());
    return sh_ptr<IDataSet>(wrapper);
}

void OutputTargets(const string& filename, const IDataSet& dataSet) {
    ofstream output(filename.c_str());
    for (int objectIndex = 0; objectIndex < dataSet.GetObjectCount(); ++objectIndex) {
        output << dataSet.GetTarget(objectIndex) << endl;
    }
}

void OutputConfidences(const string& filename, const IDataSet& dataSet) {
    ofstream output(filename.c_str());
    for (int objectIndex = 0; objectIndex < dataSet.GetObjectCount(); ++objectIndex) {
        for (int target = 0; target < dataSet.GetClassCount(); ++target) {
            if (target != 0) {
                output << "\t";
            }
            output << dataSet.GetConfidence(objectIndex, target);
        }
        output << endl;
    }
}

void OutputWeights(const string& filename, const IDataSet& dataSet) {
    ofstream output(filename.c_str());
    for (int objectIndex = 0; objectIndex < dataSet.GetObjectCount(); ++objectIndex) {
         output << dataSet.GetWeight(objectIndex) << endl;
    }
}

void LoadPenalties(DataSet* dataSet, const string& filename) {
    ifstream input(filename.c_str());
    if (!input.is_open()) {
        return;
    }
    MetaData& metaData = dataSet->GetMetaData();
    string line;
    for (int i = 0; i < metaData.GetClassCount(); ++i) {
        getline(input, line);
        std::stringstream lineStream;
        lineStream << line;
        double penalty;
        for (int j = 0; j < metaData.GetClassCount(); ++j) {
            lineStream >> penalty;
            metaData.SetPenalty(i, j, penalty);
        }
        if (lineStream >> penalty) {
            metaData.SetPenalty(i, Refuse, penalty);
        }
    }
}

void LoadParameters(sh_ptr<IClassifier> classifier, const string& filename) {
    ifstream input(filename.c_str());
    string line;
    while (input >> line) {
        int index = line.find('=');
        if (index == string::npos) {
            throw std::logic_error("Incorrect algorithm parameter");
        }
        string paramaterName = line.substr(0, index);
        string parameterValue = line.substr(index + 1);
        if (!classifier->SetParameter(paramaterName, parameterValue)) {
            cerr << "Parameter " << paramaterName
                 << " could not be set to " << parameterValue << endl;
        }
    }
}

int main(int argc, char** argv) {
    try {
		typedef TCLAP::ValueArg<string> StringArg;
        TCLAP::CmdLine cmd("Command description message", ' ', "0.1");

		TCLAP::UnlabeledValueArg<string> commandTypeArg(
			"command", "Type of command", true, "", "string", cmd);
		StringArg classifierArg(
			"c", "classifier", "Name of classifier", false, "", "string", cmd);
		StringArg fullDataArg(
			"", "data", "File with full data", false, "", "string", cmd);
		StringArg testDataArg(
			"", "trainData", "File with train data", false, "", "string", cmd);
		StringArg trainDataArg(
			"", "testData", "File with test data", false, "", "string", cmd);
		StringArg testIndexesArg(
			"", "testIndexes", "File with test indexes", false, "", "string", cmd);
		StringArg trainIndexesArg(
			"", "trainIndexes", "File with train indexes", false, "", "string", cmd);
		StringArg penaltiesArg(
			"", "penalties", "File with penalties", false, "", "string", cmd);
        StringArg classifierParametersArg(
            "", "parameters", "File with algorithm parameters", false, "", "string", cmd);
		
		StringArg testTargetOutputArg(
			"", "testTargetOutput", "File to write targets of test set", 
			false, "", "string", cmd);
		StringArg testConfidencesOutputArg(
			"", "testConfidencesOutput", "File to write confidences of test set", 
			false, "", "string", cmd);
		StringArg testObjectsWeightsOutputArg(
			"", "testFeatureWeightsOutput", "File to write objects weights of test set", 
			false, "", "string", cmd);

		StringArg trainTargetOutputArg(
			"", "trainTargetOutput", "File to write target of test set", 
			false, "", "string", cmd);
		StringArg trainConfidencesOutputArg(
			"", "trainConfidencesOutput", "File to write confidences of test set", 
			false, "", "string", cmd);
		StringArg trainObjectsWeightsOutputArg(
			"", "trainFeatureWeightsOutput", "File to write objects weights of test set", 
			false, "", "string", cmd);

		StringArg featureWeightsOutputArg(
			"", "featureWeightsOutput", "File to write feature weights", 
			false, "", "string", cmd);

		cmd.parse(argc, argv);

		if (!classifierArg.isSet()) {
			throw TCLAP::ArgException("Classifier is not specified");
		}

        DataSet trainDataSet;
        DataSet testDataSet;
		sh_ptr<IDataSet> learningTrainData;
        sh_ptr<IDataSet> testingTrainData;
		sh_ptr<IDataSet> testData;
		if (fullDataArg.isSet()) {
            if (testDataArg.isSet() || trainDataArg.isSet()) {
                throw TCLAP::ArgException("Too many datasets specified");
            } 
			if (!testIndexesArg.isSet() || !trainIndexesArg.isSet()) {
				throw TCLAP::ArgException("Dataset indices is not specified");
			}
			LoadDataSet(&trainDataSet, fullDataArg.getValue());
            vector<int> indexes;
            ReadIndexes(trainIndexesArg.getValue(), &indexes);
            learningTrainData = GetDataSet(trainDataSet, indexes);
            testingTrainData = GetDataSet(trainDataSet, indexes);
            ReadIndexes(testIndexesArg.getValue(), &indexes);
            testData = GetDataSet(trainDataSet, indexes);
		} else {
			if (!testDataArg.isSet() || !trainDataArg.isSet()) {
				throw TCLAP::ArgException("Dataset is not specified");
			}
            LoadDataSet(&trainDataSet, trainDataArg.getValue());
            LoadDataSet(&testDataSet, testDataArg.getValue());
            if (trainIndexesArg.isSet()) {
                vector<int> indexes;
                ReadIndexes(trainIndexesArg.getValue(), &indexes);
                learningTrainData = GetDataSet(trainDataSet, indexes);
                testingTrainData = GetDataSet(trainDataSet, indexes);
            } else {
                learningTrainData.set(new DataSetWrapper(&trainDataSet));
                testingTrainData.set(new DataSetWrapper(&trainDataSet));
            }
			if (testIndexesArg.isSet()) {
                vector<int> indexes;
                ReadIndexes(testIndexesArg.getValue(), &indexes);
                testData = GetDataSet(testDataSet, indexes);
            } else {
                testData.set(new DataSetWrapper(&testDataSet));
            }
			
		}

		sh_ptr<IClassifier> classifier = CreateClassifier(classifierArg.getValue());

        if (classifierParametersArg.isSet()) {
            LoadParameters(classifier, classifierParametersArg.getValue());
        }

        if (penaltiesArg.isSet()) {
			LoadPenalties(&trainDataSet, penaltiesArg.getValue());
			LoadPenalties(&testDataSet, penaltiesArg.getValue());
		}

		classifier->Learn(learningTrainData.get());

		if (trainTargetOutputArg.isSet() || 
			trainConfidencesOutputArg.isSet() ||
			trainObjectsWeightsOutputArg.isSet()) {
			classifier->Classify(testingTrainData.get());
			if (trainTargetOutputArg.isSet()) {
				OutputTargets(trainTargetOutputArg.getValue(), *(testingTrainData.get()));
			}
			if (trainConfidencesOutputArg.isSet()) {
				OutputConfidences(trainConfidencesOutputArg.getValue(), *(testingTrainData.get()));
			}
			if (trainObjectsWeightsOutputArg.isSet()) {
				OutputWeights(trainObjectsWeightsOutputArg.getValue(), *(testingTrainData.get()));
			}
		}

		classifier->Classify(testData.get());
		if (testTargetOutputArg.isSet()) {
			OutputTargets(testTargetOutputArg.getValue(), *(testData.get()));
		}
		if (testConfidencesOutputArg.isSet()) {
			OutputConfidences(testConfidencesOutputArg.getValue(), *(testData.get()));
		}
		if (testObjectsWeightsOutputArg.isSet()) {
			OutputWeights(testObjectsWeightsOutputArg.getValue(), *(testData.get()));
		}
    } catch (TCLAP::ArgException &e) { 
        cerr << "Error: " << e.error() << " for arg " << e.argId() << endl; 
    } catch (const std::exception& ex) {
		cerr << "Exception occurred: " << ex.what() << endl;
        return EXIT_FAILURE;
    } catch (...) {
		cerr << "Unhandled error occurred" << endl;
		exit(EXIT_FAILURE);
	}
}
