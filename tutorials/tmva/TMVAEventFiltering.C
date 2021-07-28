/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
///
/// This tutorial demonstrates the filtering of Event vectors by signal and background type,
/// as well as tree type (eg. kTesting and kTraining). In particular it demonstrates a
/// general pattern how testing and training data sets can be prepared in a ROOT-TMVA format
/// and then converted into some other format readable by other machine learning libraries.
///
/// - Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis
/// - Package   : TMVA
/// - Root Macro: TMVAEventFiltering
///
/// \macro_output
/// \macro_code
/// \author Xandru Mifsud

#include "TMVA/Event.h"
#include "TMVA/DataLoader.h"
#include "TMVA/DataSetInfo.h"
#include <ROOT/RDataFrame.hxx>

void TMVAEventFiltering(){
   // We first load some data to work on...

   const std::string filepath = "http://root.cern.ch/files/tmva_class_example.root";
   const std::vector<std::string> variables = {"var1", "var2", "var3", "var4"};

   // Open the root file and get the signal and background trees...
   auto data = TFile::Open(filepath.c_str());
   auto signal = (TTree*) data->Get("TreeS");
   auto background = (TTree*) data->Get("TreeB");

   // Add variables and register the trees with the dataloader
   auto dataloader = new TMVA::DataLoader("tmva003_BDT");
   for(const auto &var : variables){
      dataloader->AddVariable(var);
   }
   dataloader->AddSignalTree(signal, 1.0);
   dataloader->AddBackgroundTree(background, 1.0);

   // Split the signal and background data into training and test data, respectively
   dataloader->PrepareTrainingAndTestTree("", "");

   /* We wish to extract 4 datasets:
    * (1) signal training data
    * (2) background training data
    * (3) signal testing data
    * (4) background testing data
    */

   // The DataSetInfo instance maintained by the DataLoader instance maintains not only the events, but also the
   // necessary information to distinguish between events such that we can partition them as such. In particular these
   // are maintained in the default data set when used as above.
   const TMVA::DataSetInfo& datasetInfo = dataloader->GetDefaultDataSetInfo();

   auto sig_training_events = new std::vector<TMVA::Event*>;
   datasetInfo.GetSignalEventCollection(TMVA::Types::kTraining, sig_training_events);

   auto bgd_training_events = new std::vector<TMVA::Event*>;
   datasetInfo.GetBackgroundEventCollection(TMVA::Types::kTraining, bgd_training_events);

   auto sig_testing_events = new std::vector<TMVA::Event*>;
   datasetInfo.GetSignalEventCollection(TMVA::Types::kTesting, sig_testing_events);

   auto bgd_testing_events = new std::vector<TMVA::Event*>;
   datasetInfo.GetBackgroundEventCollection(TMVA::Types::kTesting, bgd_testing_events);

   // From these event vectors, we can for example extract event weights and represent them as a vector, etc...

   // Remember to carry out memory management when you are done...
   delete sig_training_events;
   delete bgd_training_events;
   delete sig_testing_events;
   delete bgd_testing_events;

   // Typically external machine learning libraries require data to be represented as a 2D float matrix. From the event
   // vectors extracted above this can be easily constructed, however for convenience a function is provided to do just
   // that. Suppose we want a 2D float matrix representation of the signal testing data...

   Long64_t* n_sig_vars;
   Long64_t* n_sig_test_events;
   float** sig_test_mat = datasetInfo.GetSignalMatrix(TMVA::Types::kTesting, n_sig_test_events, n_sig_vars);

   // Displaying (up to) the first 10 rows of the matrix...
   for(Int_t i = 0; i < *n_sig_test_events || i < 10; i++){
      printf("\n%d:\t", i+1);

      for(Int_t j = 0; j < *n_sig_vars; j++){
         printf("var %d = %f\t", j, sig_test_mat[i][j]);
      }
   }

   // Once again, remember to free memory as required...
   for(Int_t i = 0; i < *n_sig_test_events; i++)
      delete[] sig_test_mat[i];

   delete[] sig_test_mat;
}