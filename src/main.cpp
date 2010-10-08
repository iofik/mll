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

void LoadParameters(IConfigurable* configurable, const string& filename) {
    ifstream input(filename.c_str());
    if (!input.is_open()) {
        return;
    }
    string line;
    while (input >> line) {
        int index = line.find('=');
        if (index == string::npos) {
            throw std::logic_error("Incorrect algorithm parameter");
        }
        string paramaterName = line.substr(0, index);
        string parameterValue = line.substr(index + 1);
        if (!configurable->SetParameter(paramaterName, parameterValue)) {
            cerr << "Warning: Parameter " << paramaterName
                 << " could not be set to " << parameterValue << endl;
        }
    }
}

void ClearTargets(IDataSet* dataSet) {
    for (int i = 0; i < dataSet->GetObjectCount(); ++i) {
        dataSet->SetTarget(i, Refuse);
    }
}

int main(int argc, char** argv) {
    try {
		typedef TCLAP::ValueArg<string> StringArg;
        TCLAP::CmdLine cmd("Command description message", ' ', "0.1");
        
        vector<string> commands;
        commands.push_back("classify");
        commands.push_back("listc");
        commands.push_back("listt");
        commands.push_back("test");
        TCLAP::ValuesConstraint<string> constraint(commands);
        TCLAP::UnlabeledValueArg<string> commandTypeArg(
            "command", "Command", true, "", &constraint, cmd);
		StringArg classifierArg(
			"c", "classifier", "Classifier name", false, "", "string", cmd);
        StringArg testerArg(
            "t", "tester", "Tester name", false, "", "string", cmd);
		StringArg fullDataArg(
			"", "data", "File with full data", false, "", "string", cmd);
		StringArg trainDataArg(
			"", "trainData", "File with train data", false, "", "string", cmd);
		StringArg testDataArg(
			"", "testData", "File with test data", false, "", "string", cmd);
		StringArg testIndexesArg(
			"", "testIndexes", "File with test indexes", false, "", "string", cmd);
		StringArg trainIndexesArg(
			"", "trainIndexes", "File with train indexes", false, "", "string", cmd);
		StringArg penaltiesArg(
			"", "penalties", "File with penalties", false, "", "string", cmd);
        StringArg classifierParametersArg(
            "", "parameters", "File with algorithm parameters", false, "", "string", cmd);
        StringArg testerParametersArg(
            "", "testerParameters", "File with tester parameters", false, "", "string", cmd);
		
		StringArg testTargetOutputArg(
			"", "testTargetOutput", "File to write targets of test set", 
			false, "", "string", cmd);
		StringArg testConfidencesOutputArg(
			"", "testConfidencesOutput", "File to write confidences of test set", 
			false, "", "string", cmd);
		StringArg testObjectsWeightsOutputArg(
			"", "testObjectWeightsOutput", "File to write objects weights of test set", 
			false, "", "string", cmd);

		StringArg trainTargetOutputArg(
			"", "trainTargetOutput", "File to write target of train set", 
			false, "", "string", cmd);
		StringArg trainConfidencesOutputArg(
			"", "trainConfidencesOutput", "File to write confidences of train set", 
			false, "", "string", cmd);
		StringArg trainObjectsWeightsOutputArg(
			"", "trainObjectWeightsOutput", "File to write objects weights of train set", 
			false, "", "string", cmd);

		StringArg featureWeightsOutputArg(
			"", "featureWeightsOutput", "File to write feature weights", 
			false, "", "string", cmd);

		cmd.parse(argc, argv);

        cout << "\t*\t*\t*\tWelcome to MLL!\t\t*\t*\t*" << endl;
        if (commandTypeArg.getValue() == "listc") {
            ListClassifiers();
        } else if (commandTypeArg.getValue() == "listt") {
            RegisterCVTesters();
            ListTesters();
        } else if (commandTypeArg.getValue() == "classify") {
		    if (!classifierArg.isSet()) {
			    throw TCLAP::ArgException("Classifier is not specified", "classifier");
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
				    throw TCLAP::ArgException("Dataset indices not specified");
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
                LoadParameters(classifier.get(), classifierParametersArg.getValue());
            }

            if (penaltiesArg.isSet()) {
			    LoadPenalties(&trainDataSet, penaltiesArg.getValue());
			    LoadPenalties(&testDataSet, penaltiesArg.getValue());
		    }

            cout << "Training classifier..." << endl;
		    classifier->Learn(learningTrainData.get());

		    if (trainTargetOutputArg.isSet() || 
			    trainConfidencesOutputArg.isSet() ||
			    trainObjectsWeightsOutputArg.isSet()) {
                cout << "Classifying train dataset..." << endl;
                ClearTargets(testingTrainData.get());
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

            ClearTargets(testData.get());
			cout << "Classifying test dataset..." << endl;
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
        } else if (commandTypeArg.getValue() == "test") {
            if (!classifierArg.isSet()) {
			    throw TCLAP::ArgException("Classifier is not specified", "classifier");
		    }
            if (!testerArg.isSet()) {
			    throw TCLAP::ArgException("Tester is not specified", "tester");
		    }
            if (!fullDataArg.isSet()) {
                throw TCLAP::ArgException("Data is not specified", "data");
            }
            DataSet dataSet;
            LoadDataSet(&dataSet, fullDataArg.getValue());
            if (penaltiesArg.isSet()) {
			    LoadPenalties(&dataSet, penaltiesArg.getValue());
		    }

		    sh_ptr<IClassifier> classifier = CreateClassifier(classifierArg.getValue());
            if (classifierParametersArg.isSet()) {
                LoadParameters(classifier.get(), classifierParametersArg.getValue());
            }
            sh_ptr<ITester> tester = CreateTester(testerArg.getValue());
            if (classifierParametersArg.isSet()) {
                LoadParameters(tester.get(), testerParametersArg.getValue());
            }

            cout << "Testing classifier..." << endl;
            double loss = tester->Test(*classifier, &dataSet);

            cout << "Classification loss: " << loss << endl;
        }

        cout << "Goodbye!" << endl;
    } catch (TCLAP::ArgException &e) { 
        cerr << "Usage error: " << e.error();
        if (e.argId()[0] != ' ') {
            cerr << " (argument: " << e.argId() << ")";
        }
        cerr << endl;
        exit(EXIT_FAILURE);
    } catch (const std::exception& ex) {
		cerr << "Exception occurred: " << ex.what() << endl;
        exit(EXIT_FAILURE);
    } catch (...) {
		cerr << "Unhandled error occurred" << endl;
		exit(EXIT_FAILURE);
  } 
}
