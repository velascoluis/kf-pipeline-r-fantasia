library("xgboost")
library("optparse")
library("jsonlite")
library("remotes")
remotes::install_github("cloudyr/googleCloudStorageR")
#@velascoluis - Needed to install the github version, otherwise we cannot upload objects to GCS buckets with uniform access policy
library("googleCloudStorageR")
#######################################
googleAuthR::gar_auth_service("/app/client_secret.json")
#googleAuthR::gar_auth_service("/Users/velascoluis/PycharmProjects/kubeflow-pipeline-r-fantasia/pipelines/image_code/00_r_train/src/secret/client_secret.json")
option_list = list(
              make_option(c("-f","--model_file_name"), type="character", default=NULL,help="model file name", metavar="character"),
              make_option(c("-b","--gcp_bucket"), type="character", default=NULL, help="GCP Bucket", metavar="character"));

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
#######################################
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
               eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic", save_period = NULL)
#@velascoluis - save period must be explictly set to NULL, otherwise the KFServing pod fails with xgboost.core.XGBoostError: b'[09:41:58] /workspace/dmlc-core/src/io/local_filesys.cc:209: Check failed: allow_null
xgb.save(bst, opt$model_file_name)
gcs_upload(opt$model_file_name,bucket =opt$gcp_bucket, name= paste("rmodel",opt$model_file_name, sep="/"), predefinedAcl = "bucketLevel")
