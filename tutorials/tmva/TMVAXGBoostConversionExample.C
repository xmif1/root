//
// Created by Xandru Mifsud on 14/07/2021.
//

#include <xgboost/c_api.h>
#include <ROOT/RDataFrame.hxx>
#include <TMVA/DataLoader.h>
#include <TMVA/DataSetInfo.h>
#include <TMVA/DataSet.h>
#include <TMVA/Factory.h>
#include <TMVA/Types.h>

/* Simple macro to carry out error checking and handling around xgboost C-api calls
 * See also here: https://github.com/dmlc/xgboost/blob/master/demo/c-api/c-api-demo.c
 */
#define safe_xgboost(call){                                                                      \
   int err = (call);                                                                             \
   if(err != 0){                                                                                 \
      throw runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +               \
                            ": error in " + #call + ":" + XGBGetLastError());                    \
   }                                                                                             \
}

/* Struct-type which maintains the necessary data required for, in particular, training and making
 * predictions using xgboost's C-api.
 *
 * free() should be called to 'destruct' an xgboost_data instance; especially important since it
 * frees memory associated with DMatrix instances, which otherwise could lead to memory leaks.
 */
typedef struct xgboost_data{
   // meta-data
   DMatrixHandle sb_dmats[1];
   Float_t* weights;
   Float_t* labels;
   Long64_t n_sig, n_bgd;

   xgboost_data(Long64_t n_sig, Long64_t n_bgd){
      this->n_sig = n_sig;
      this->n_bgd = n_bgd;

      this->weights = new Float_t[n_sig + n_bgd];
      this->labels = new Float_t[n_sig + n_bgd];
   }

   // call for memory management
   void free() const{
      safe_xgboost(XGDMatrixFree(sb_dmats[0]))
   }
} xgboost_data;

typedef pair<char const*, char const*> kv_pair;
typedef std::vector<kv_pair> xgbooster_opts;

/* Utility function for converting from ROOT's TTree data representation, to xgboost's DMatrix representation.
 * Furthermore,
 * (i)  We also extract the number of signal and background events.
 * (ii) We calculate balanced weights between the signal and background datasets.
 *
 * All this data is wrapped in an xgboost_data instance, the internals of which can be used with xgboost's C-api.
 *
 * The passed vector<string> of variable names specifies which branches to extract from the respective TTree instances.
 */
xgboost_data* ROOTToXGBoost(TTree& signal_tree, TTree& background_tree, std::vector<string>& variables,
                            const Float_t* sig_weight, const Float_t* bgd_weight){

   // Represent signal TTree as a data frame
   ROOT::RDataFrame sig_dframe(signal_tree);
   auto n_sig = sig_dframe.Count(); // and extract the number of signal events

   // Represent background TTree as a data frame
   ROOT::RDataFrame bgd_dframe(background_tree);
   auto n_bgd = bgd_dframe.Count(); // and extract the number of background events

   const auto n_vars = variables.size(); // count the number of vars

   // Initialise Float_t array to hold a 2-dim representation of the signal and background trees
   Float_t sb_mat[*n_sig + *n_bgd][n_vars];

   // Loop across the variables and events, populating sb_mat resulting in a 2-dim representation of the signal and
   // background trees
   Long64_t i; Long64_t j = 0;
   for(auto& var: variables){
      i = 0;
      for(auto& sig_mat_ij: sig_dframe.Take<Float_t>(var)){ // first n_sig rows will be signal data
         sb_mat[i][j] = sig_mat_ij;
         i++;
      }
      for(auto& bgd_mat_ij: bgd_dframe.Take<Float_t>(var)){ // and the following n_bgd rows will be background data
         sb_mat[i][j] = bgd_mat_ij;
         i++;
      }

      j++;
   }

   auto data = new xgboost_data(*n_sig, *n_bgd); // maintains xgboost readable data

   Float_t sw; // unless given a weight for signal data, we calculate a balanced weight
   if(sig_weight == nullptr){
      sw = 1.0 + (*n_bgd/(*n_sig));
   }else{
      sw = *sig_weight;
   }

   // Set the weight for the signal data, and the labels to 0.0
   for(Long64_t k = 0; k < *n_sig; k++){ (data->labels)[k] = 0.0; (data->weights)[k] = sw; }

   Float_t bw; // unless given a weight for background data, we calculate a balanced weight
   if(bgd_weight == nullptr){
      bw = 1.0 + (*n_sig/(*n_bgd));
   }else{
      bw = *bgd_weight;
   }

   // Set the weight for the background data, and the labels to 1.0
   for(Long64_t k = *n_sig; k < (*n_sig + *n_bgd); k++){ (data->labels)[k] = 1.0; (data->weights)[k] = bw; }

   // Populate the DMatrix datastructure held in the xgboost_data instance...
   safe_xgboost(XGDMatrixCreateFromMat((Float_t*) sb_mat, *n_sig + *n_bgd, n_vars, 0, &((data->sb_dmats)[0])))
   safe_xgboost(XGDMatrixSetFloatInfo((data->sb_dmats)[0], "label", data->labels, *n_sig + *n_bgd))

   return data;
}

/* Utility function for converting from ROOT's DataSetInfo instance, to xgboost's DMatrix representation.
 * Furthermore,
 * (i)   We also extract the number of signal and background events.
 * (ii)  We calculate balanced weights between the signal and background datasets.
 * (iii) We are capable of carrying out additional filtering on the data by filtering out events whose type does not
 *       match the tree type specified, in particular either kTesting or kTraining are supported presently.
 *
 * All this data is wrapped in an xgboost_data instance, the internals of which can be used with xgboost's C-api.
 *
 * The passed vector<string> of variable names specifies which branches to extract from the respective TTree instances.
 */
xgboost_data* ROOTToXGBoost(const TMVA::DataSetInfo& dataset_info, TMVA::Types::ETreeType type){
   TMVA::DataSet* dataset = dataset_info.GetDataSet(); // reference to DataSet instance being filtered

   const auto n_vars = dataset->GetNVariables(); // get number of variables
   Long64_t n_sig, n_bgd; // maintain the number of signal and background events respectively

   // Depending on whether we are keeping testing or training events, set the number of such signal and background events
   if(type == TMVA::Types::kTesting){
      n_sig = dataset->GetNEvtSigTest();
      n_bgd = dataset->GetNEvtBkgdTest();
   }else if(type == TMVA::Types::kTraining){
      n_sig = dataset->GetNEvtSigTrain();
      n_bgd = dataset->GetNEvtBkgdTrain();
   }else{
      throw runtime_error("Unexpected treeType (must be either kTesting or kTraining).");
   }

   // Initialise Float_t array to hold a 2-dim representation of the signal and background trees
   Float_t sb_mat[n_sig + n_bgd][n_vars];

   auto data = new xgboost_data(n_sig, n_bgd); // maintains xgboost readable data

   Long64_t i = 0;
   for(auto& event: dataset->GetEventCollection(type)){ // Notice here that unlike the TTree variant of the function,
                                                      // the rows will be mixed signal and background events
      // Populate the 2d matrix...
      for(Long64_t j = 0; j < n_vars; j++){
         sb_mat[i][j] = event->GetValue(j);
      }


      if(dataset_info.IsSignal(event)){ // Set the weight for the signal data, and the labels to 0.0
         (data->labels)[i] = 0.0;
         (data->weights)[i] = event->GetOriginalWeight();
      }else{
         (data->labels)[i] = 1.0; // Set the weight for the background data, and the labels to 1.0
         (data->weights)[i] = event->GetOriginalWeight();
      }

      i++;
   }

   // Populate the DMatrix datastructure held in the xgboost_data instance...
   safe_xgboost(XGDMatrixCreateFromMat((Float_t*) sb_mat, n_sig + n_bgd, n_vars, 0, &((data->sb_dmats)[0])))
   safe_xgboost(XGDMatrixSetFloatInfo((data->sb_dmats)[0], "label", data->labels, n_sig + n_bgd))

   return data;
}

// Simple trainer using the XGBoost C-api, which returns a trained BoosterHandle instance.
BoosterHandle xgboost_train(xgboost_data* data, xgbooster_opts* opts, UInt_t n_iter){
   BoosterHandle booster;
   safe_xgboost(XGBoosterCreate(data->sb_dmats, 1, &booster)) // Initialise a BoosterHandle instance

   for(auto opt: *opts){ // For each specified option, set the booster parameters...
      safe_xgboost(XGBoosterSetParam(booster, opt.first, opt.second))
   }

   // Then train the booster for the specified number of iterations
   for(UInt_t iter = 0; iter < n_iter; iter++){
      safe_xgboost(XGBoosterUpdateOneIter(booster, iter, (data->sb_dmats)[0]))
   }

   return booster;
}

void ttree_xgboost_example(){
   // We first load some data to work on...

   const std::string filepath = "http://root.cern.ch/files/tmva_class_example.root";
   std::vector<std::string> variables = {"var1", "var2", "var3", "var4"};

   // Open the root file and get the signal and background trees...
   auto data = TFile::Open(filepath.c_str());
   auto signal = (TTree*) data->Get("TreeS");
   auto background = (TTree*) data->Get("TreeB");

   // ----------------------------------------------- XGBOOST TRAINING -------------------------------------------------
   // Extract the training data set and convert it to an XGBoost readable format
   xgboost_data* xg_train_data = ROOTToXGBoost(*signal, *background, variables);

   // Set the options for the BoosterHandle instance that will be trained...
   xgbooster_opts opts;
   opts.push_back(kv_pair("max_depth", "2"));

   // And finally call the simple trainer function we implemented based on XGBoost's C-api...
   BoosterHandle xgbooster = xgboost_train(xg_train_data, &opts, 300);

   // --------------------------------------------- XGBOOST PREDICTION -------------------------------------------------
   // Prepare small testing set...
   float test_data[3][4] = {{-1.14361, -0.822373, -0.495426, -0.629427},
                            {2.14344, -0.0189228, 0.26703, 1.26749},
                            {-0.443913, 0.486827, 0.139535, 0.611483}};

   DMatrixHandle xg_test_dmat;
   safe_xgboost(XGDMatrixCreateFromMat((Float_t*) test_data, 3, 4, 0, &xg_test_dmat))

   // Prepare the necessary data structures and carry out the predictions on the (converted) testing data set...
   bst_ulong output_length;
   const Float_t *output_result;
   safe_xgboost(XGBoosterPredict(xgbooster, xg_test_dmat, 0, 0, 0, &output_length, &output_result))

   // Print each class prediction...
   for(Long64_t j = 0; j < output_length; j++){
      printf("prediction[%lli] = %f \n", j, output_result[j]);
   }

   // Lastly, free any XGBoost related data structures to prevent memory leaks.
   xg_train_data->free();
   safe_xgboost(XGDMatrixFree(xg_test_dmat))
   safe_xgboost(XGBoosterFree(xgbooster))
}

/* This example carries out simple training and testing of BDT implementations provided by XGBoost.
 * We can prepare data using TMVA DataLoader, including splitting the data into signal and background data, and then
 * furthermore into training and testing data, resulting in 4 collections.
 *
 * As per usual, we can easily use the prepared datasets with a TMVA Factory instance. However, we can also use the
 * implemented conversion functions to transform the datasets into XGBoost readable datasets, and use them for training
 * and testing.
 *
 * A more general example of the general approach to conversion from ROOT to another machine learning readable format is
 * outlined in the event_filtering_tutorial.cpp file.
 */
void dataloader_xgboost_example(){
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

   // ----------------------------------------------- XGBOOST TRAINING -------------------------------------------------
   // Extract the training data set and convert it to an XGBoost readable format
   xgboost_data* xg_train_data = ROOTToXGBoost(dataloader->GetDefaultDataSetInfo(), TMVA::Types::kTraining);

   // Set the options for the BoosterHandle instance that will be trained...
   xgbooster_opts opts;
   opts.push_back(kv_pair("max_depth", "2"));

   // And finally call the simple trainer function we implemented baseed on XGBoost's C-api...
   BoosterHandle xgbooster = xgboost_train(xg_train_data, &opts, 300);

   // --------------------------------------------- XGBOOST PREDICTION -------------------------------------------------
   // Extract the testing data set and convert it to an XGBoost readable format
   xgboost_data* xg_test_data = ROOTToXGBoost(dataloader->GetDefaultDataSetInfo(), TMVA::Types::kTesting);

   // Prepare the necessary data structures and carry out the predictions on the (converted) testing data set...
   bst_ulong output_length;
   const Float_t *output_result;
   safe_xgboost(XGBoosterPredict(xgbooster, (xg_train_data->sb_dmats)[0], 0, 0, 0, &output_length, &output_result))

   // Print each prediction along with its label...
   for(Long64_t j = 0; j < output_length; j++){
      printf("label[%lli] = %f, prediction[%lli] = %f \n", j, (xg_train_data->labels)[j], j, output_result[j]);
   }

   // Lastly, free any XGBoost related data structures to prevent memory leaks.
   xg_train_data->free();
   xg_test_data->free();
   safe_xgboost(XGBoosterFree(xgbooster));
}

int main(){
   bool ttree_example = true;

   if(ttree_example){
      ttree_xgboost_example();
   }else{
      dataloader_xgboost_example();
   }

   return 0;
}