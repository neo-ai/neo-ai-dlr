#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 3rd-party
#include <nlohmann/json.hpp>
#include "npy.hpp"

// aws-sdk
#include <aws/core/Aws.h>
#include <aws/core/client/AWSError.h>
#include <aws/iam/IAMClient.h>
#include <aws/iam/model/AttachRolePolicyRequest.h>
#include <aws/iam/model/CreateRoleRequest.h>
#include <aws/iam/model/GetRoleRequest.h>
#include <aws/iam/model/ListRolesRequest.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/sagemaker/SageMakerClient.h>
#include <aws/sagemaker/model/CreateCompilationJobRequest.h>
#include <aws/sagemaker/model/DescribeCompilationJobRequest.h>
#include <aws/sagemaker/model/Framework.h>
#include <aws/sagemaker/model/InputConfig.h>

// internal
#include <dlr.h>

const std::string ROLE_NAME = "windows-demo-test-role";

Aws::S3::S3Client getS3Client() {
  // for tutorial, we set region to us-west-2
  Aws::String region_aws_s("us-west-2");

  // init client
  Aws::Client::ClientConfiguration client_config;
  client_config.region = region_aws_s;
  Aws::S3::S3Client s3_client(client_config);
  return s3_client;
}

Aws::IAM::IAMClient getIamClient() {
  // create IAM role, note that IAM only support global region.
  Aws::IAM::IAMClient iam_client;
  return iam_client;
}

Aws::SageMaker::SageMakerClient getSageMakerClient() {
  // for tutorial, we set region to us-west-2
  Aws::String region_aws_s("us-west-2");

  // init client
  Aws::Client::ClientConfiguration client_config;
  client_config.region = region_aws_s;
  Aws::SageMaker::SageMakerClient sm_client(client_config);
  return sm_client;
}

template <typename T>
std::string getErrorMessage(const Aws::Client::AWSError<T>& error) {
  auto exception_name = error.GetExceptionName();
  std::string expection_str = std::string(exception_name.c_str(), exception_name.size());

  auto error_msg = error.GetMessage();
  std::string error_msg_str = std::string(error_msg.c_str(), error_msg.size());

  return expection_str + ":" + error_msg_str;
}

void GetPretrainedModel(const std::string& bucket_name, const std::string& model_name,
                        const std::string& filename) {
  const Aws::String aws_bucket_name(bucket_name.c_str(),
                                    bucket_name.size());  // "neo-ai-dlr-test-artifacts";
  const Aws::String aws_object_name(model_name.c_str(), model_name.size());

  // download s3 object
  Aws::S3::S3Client s3_client = getS3Client();

  Aws::S3::Model::GetObjectRequest object_request;
  object_request.SetBucket(aws_bucket_name);
  object_request.SetKey(aws_object_name);

  auto get_object_outcome = s3_client.GetObject(object_request);
  if (!get_object_outcome.IsSuccess()) {
    auto error = get_object_outcome.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("GetModel error: " + error_str);
  } else {
    auto& model_file = get_object_outcome.GetResultWithOwnership().GetBody();

    // download the sample file
    const char* output_filename = filename.c_str();
    std::ofstream output_file(output_filename, std::ios::binary);
    output_file << model_file.rdbuf();
  }
}

void createBucket(const std::string& bucket_name) {
  // create bucket
  const Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
  Aws::S3::Model::CreateBucketRequest request;
  request.SetBucket(s3_bucket_name);

  // for demo purpose: we set that to us-west-2
  const Aws::S3::Model::BucketLocationConstraint& region =
      Aws::S3::Model::BucketLocationConstraint::us_west_2;
  Aws::S3::S3Client s3_client = getS3Client();
  Aws::S3::Model::CreateBucketConfiguration bucket_config;
  bucket_config.SetLocationConstraint(region);
  request.SetCreateBucketConfiguration(bucket_config);

  // making request call
  auto outcome = s3_client.CreateBucket(request);
  if (!outcome.IsSuccess()) {
    auto err = outcome.GetError();
    std::string error_str = getErrorMessage(err);
    throw std::runtime_error("CreateBucket error: " + error_str);
  }
}

void uploadModel(const std::string& bucket_name, const std::string& model_name,
                 const std::string& filename) {
  Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
  Aws::String s3_aws_object_name(model_name.c_str(), model_name.size());
  Aws::S3::S3Client s3_client = getS3Client();

  // upload model to s3 bucket
  Aws::S3::Model::PutObjectRequest object_request;
  object_request.SetBucket(s3_bucket_name);
  object_request.SetKey(s3_aws_object_name);

  // set binary to body
  const std::shared_ptr<Aws::IOStream> input_data = Aws::MakeShared<Aws::FStream>(
      "SampleAllocationTag", filename.c_str(), std::ios_base::in | std::ios_base::binary);
  object_request.SetBody(input_data);

  // put model to the s3 bucket
  auto put_object_outcome = s3_client.PutObject(object_request);
  if (!put_object_outcome.IsSuccess()) {
    auto error = put_object_outcome.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("UploadModel error: " + error_str);
  }
}

bool checkBucketExist(const std::string& bucket_name) {
  Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
  Aws::S3::S3Client s3_client = getS3Client();
  auto list_bucket_resp = s3_client.ListBuckets();
  if (list_bucket_resp.IsSuccess()) {
    Aws::Vector<Aws::S3::Model::Bucket> bucket_list = list_bucket_resp.GetResult().GetBuckets();
    for (auto const& bucket : bucket_list) {
      if (bucket.GetName().compare(s3_bucket_name) == 0) {
        return true;
      }
    }
  } else {
    auto error = list_bucket_resp.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("ListBucket error: " + error_str);
  }
  return false;
}

bool checkModelExist(const std::string& bucket_name, const std::string& model_name) {
  Aws::String aws_bucket_name(bucket_name.c_str(), bucket_name.size());
  Aws::String aws_model_name(model_name.c_str(), model_name.size());

  Aws::S3::S3Client s3_client = getS3Client();
  Aws::S3::Model::ListObjectsRequest objects_request;
  objects_request.WithBucket(aws_bucket_name);

  auto list_objects_outcome = s3_client.ListObjects(objects_request);
  if (list_objects_outcome.IsSuccess()) {
    Aws::Vector<Aws::S3::Model::Object> object_list =
        list_objects_outcome.GetResult().GetContents();
    for (auto const& s3_object : object_list) {
      if (s3_object.GetKey().compare(aws_model_name) == 0) {
        return true;
      }
    }
  } else {
    auto error = list_objects_outcome.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("ListObjects error: " + error_str);
  }

  return false;
}

void UploadModelToS3(const std::string& model_name, const std::string& filename,
                     const std::string& s3_bucket_name) {
  // first create bucket
  bool isBucketExist = checkBucketExist(s3_bucket_name);
  if (!isBucketExist) {
    createBucket(s3_bucket_name);
  }

  // then create/upload model
  if (!checkModelExist(s3_bucket_name, model_name)) {
    uploadModel(s3_bucket_name, model_name, filename);
  }
}

nlohmann::json getIamPolicy() {
  nlohmann::json statement;
  statement["Action"] = "sts:AssumeRole";
  statement["Effect"] = "Allow";
  statement["Principal"]["Service"] = "sagemaker.amazonaws.com";

  nlohmann::json policy;
  policy["Statement"] = nlohmann::json::array();
  policy["Statement"].push_back(statement);
  policy["Version"] = "2012-10-17";
  return policy;
}

bool checkIamRole(const std::string& iam_role) {
  Aws::IAM::IAMClient iam_client = getIamClient();
  Aws::String aws_iam_role(iam_role.c_str(), iam_role.size());
  Aws::IAM::Model::GetRoleRequest get_role_request;
  get_role_request.SetRoleName(aws_iam_role);

  auto get_role_response = iam_client.GetRole(get_role_request);
  if (get_role_response.IsSuccess()) {
    return true;
  } else {
    auto error = get_role_response.GetError();
    auto errorType = error.GetErrorType();
    if (errorType == Aws::IAM::IAMErrors::NO_SUCH_ENTITY) {
      std::cout << "Role doesn't exist and will create one" << std::endl;
    } else {
      std::string error_str = getErrorMessage(error);
      throw std::runtime_error("GetIamRole error: " + error_str);
    }
  }
  return false;
}

void createIamRole(const std::string& iam_role) {
  nlohmann::json policy = getIamPolicy();
  std::string policy_s = policy.dump();
  const Aws::String aws_role_name(iam_role.c_str(), iam_role.size());
  const Aws::String aws_policy_name(policy_s.c_str(), policy_s.size());

  Aws::IAM::IAMClient iam_client = getIamClient();
  Aws::IAM::Model::CreateRoleRequest create_role_request;
  create_role_request.SetRoleName(aws_role_name);
  create_role_request.SetPath("/");
  create_role_request.SetAssumeRolePolicyDocument(aws_policy_name);

  auto create_role_outcome = iam_client.CreateRole(create_role_request);
  if (!create_role_outcome.IsSuccess()) {
    auto error = create_role_outcome.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("CreateIam error: " + error_str);
  }
}

void attachIamPolicy(const std::string& iam_role) {
  const std::vector<Aws::String> policies = {"arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                                             "arn:aws:iam::aws:policy/AmazonS3FullAccess"};
  const Aws::String aws_role_name(iam_role.c_str(), iam_role.size());

  Aws::IAM::IAMClient iam_client = getIamClient();
  for (int i = 0; i < policies.size(); i++) {
    Aws::String aws_policy_arn = policies[i];
    Aws::IAM::Model::AttachRolePolicyRequest attach_policy_request;
    attach_policy_request.SetPolicyArn(aws_policy_arn);
    attach_policy_request.SetRoleName(aws_role_name);
    auto attach_policy_outcome = iam_client.AttachRolePolicy(attach_policy_request);
    if (!attach_policy_outcome.IsSuccess()) {
      auto error = attach_policy_outcome.GetError();
      std::string error_str = getErrorMessage(error);
      throw std::runtime_error("AttachIamPolicy error : " + error_str);
    }
  }
}

void CreateIamStep() {
  if (!checkIamRole(ROLE_NAME)) {
    createIamRole(ROLE_NAME);
    attachIamPolicy(ROLE_NAME);
  }
}

std::string getJobName() {
  std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch());

  std::stringstream ss;
  ss << "pi-demo-" << std::to_string(ms.count());
  return ss.str();
}

Aws::SageMaker::Model::CompilationJobStatus poll_job_status(const std::string& job_name) {
  Aws::SageMaker::SageMakerClient sm_client = getSageMakerClient();
  Aws::SageMaker::Model::DescribeCompilationJobRequest describe_job_request;

  Aws::String aws_job_name(job_name.c_str(), job_name.size());
  describe_job_request.SetCompilationJobName(aws_job_name);
  auto describe_job_resp = sm_client.DescribeCompilationJob(describe_job_request);

  if (!describe_job_resp.IsSuccess()) {
    auto error = describe_job_resp.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("DescribeCompilationJob error: " + error_str);
  }

  auto result = describe_job_resp.GetResult();
  return result.GetCompilationJobStatus();
}

void getIamRole(const std::string& role_name, Aws::IAM::Model::Role& role) {
  Aws::IAM::IAMClient iam_client = getIamClient();
  Aws::String aws_iam_role(role_name.c_str(), role_name.size());
  Aws::IAM::Model::GetRoleRequest get_role_request;
  get_role_request.SetRoleName(aws_iam_role);

  auto get_role_response = iam_client.GetRole(get_role_request);
  if (get_role_response.IsSuccess()) {
    role = get_role_response.GetResult().GetRole();
  } else {
    auto error = get_role_response.GetError();
    auto errorType = error.GetErrorType();
    if (errorType == Aws::IAM::IAMErrors::NO_SUCH_ENTITY) {
      throw std::runtime_error("Role doesn't exist and will create one");
    } else {
      std::string error_str = getErrorMessage(error);
      throw std::runtime_error("GetIamRole error: " + error_str);
    }
  }
}

void CompileNeoModel(const std::string& bucket_name, const std::string& model_name,
                     const std::string& target) {
  // set input parameters
  std::string input_s3 = "s3://" + bucket_name + "/" + model_name;

  const Aws::SageMaker::Model::Framework framework = Aws::SageMaker::Model::Framework::MXNET;
  const Aws::String aws_s3_uri(input_s3.c_str(), input_s3.size());
  const Aws::String data_shape = "{'data':[1,3,224,224]}";
  const Aws::String aws_role_name(ROLE_NAME.c_str(), ROLE_NAME.size());

  std::string job_name = getJobName();
  const Aws::String aws_job_name(job_name.c_str(), job_name.size());

  // set input config
  Aws::SageMaker::SageMakerClient sm_client = getSageMakerClient();
  Aws::SageMaker::Model::InputConfig input_config;
  input_config.SetS3Uri(aws_s3_uri);
  input_config.SetFramework(framework);
  input_config.SetDataInputConfig(data_shape);

  // set output config parameters
  std::string output_s3 = "s3://" + bucket_name + "/output";
  Aws::String aws_output_s3(output_s3.c_str(), output_s3.size());

  // set target device
  Aws::String aws_target_device(target.c_str(), target.size());
  Aws::SageMaker::Model::TargetDevice target_device =
      Aws::SageMaker::Model::TargetDeviceMapper::GetTargetDeviceForName(aws_target_device);

  // set output config
  Aws::SageMaker::Model::OutputConfig output_config;
  output_config.SetS3OutputLocation(aws_output_s3);
  output_config.SetTargetDevice(target_device);

  // set stopping condition
  int max_runtime_in_sec = 900;
  Aws::SageMaker::Model::StoppingCondition stopping_condition;
  stopping_condition.SetMaxRuntimeInSeconds(max_runtime_in_sec);

  // get iam role
  Aws::IAM::Model::Role role;
  getIamRole(ROLE_NAME, role);
  if (role.GetArn().empty()) {
    throw std::runtime_error("Role doesn't exist");
  }

  // create Neo compilation job
  Aws::SageMaker::Model::CreateCompilationJobRequest create_job_request;
  create_job_request.SetCompilationJobName(aws_job_name);
  create_job_request.SetRoleArn(role.GetArn());
  create_job_request.SetInputConfig(input_config);
  create_job_request.SetOutputConfig(output_config);
  create_job_request.SetStoppingCondition(stopping_condition);

  auto compilation_resp = sm_client.CreateCompilationJob(create_job_request);
  if (!compilation_resp.IsSuccess()) {
    auto error = compilation_resp.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("CreateCompilationJob error: " + error_str);
  }

  // poll job for validation
  bool is_success = false;
  int attempts = 10;
  for (int i = 0; i < attempts; i++) {
    std::cout << "Waiting for compilation..." << std::endl;
    auto status = poll_job_status(job_name);
    if (status == Aws::SageMaker::Model::CompilationJobStatus::COMPLETED) {
      std::cout << "Compile successfully" << std::endl;
      is_success = true;
      break;
    } else if (status == Aws::SageMaker::Model::CompilationJobStatus::FAILED) {
      throw std::runtime_error("Compilation fail");
    }
    std::this_thread::sleep_for(std::chrono::seconds(30));
  }

  if (!is_success) {
    throw std::runtime_error("Compilation timeout");
  }
  std::cout << "Done!" << std::endl;
}

void GetCompiledModelFromNeo(const std::string& bucket_name, const std::string& model_name,
                             const std::string& target, const std::string& compiled_filename) {
  std::string output_path = "output/" + model_name + "-" + target;

  Aws::String aws_bucket_name(bucket_name.c_str(), bucket_name.size());
  Aws::String aws_object_name(output_path.c_str(), output_path.size());

  // download s3 object
  Aws::S3::S3Client s3_client = getS3Client();

  Aws::S3::Model::GetObjectRequest object_request;
  object_request.SetBucket(aws_bucket_name);
  object_request.SetKey(aws_object_name);

  auto get_object_outcome = s3_client.GetObject(object_request);
  if (!get_object_outcome.IsSuccess()) {
    auto error = get_object_outcome.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("GetModel error: " + error_str);
  } else {
    auto& model_file = get_object_outcome.GetResultWithOwnership().GetBody();

    // download the sample file
    const char* output_filename = compiled_filename.c_str();
    std::ofstream output_file(output_filename, std::ios::binary);
    output_file << model_file.rdbuf();
  }
}

void DownloadNpyData() {
  Aws::S3::S3Client s3_client = getS3Client();

  // this is a preprocess image file in our sample directory
  // this can be changed to any s3 bucket you specified
  Aws::String aws_bucket_name = "neo-ai-dlr-test-artifacts";
  Aws::String aws_object_name = "test-data/dog.npy";
  std::string filename = "dog.npy";

  Aws::S3::Model::GetObjectRequest object_request;
  object_request.SetBucket(aws_bucket_name);
  object_request.SetKey(aws_object_name);

  auto get_object_outcome = s3_client.GetObject(object_request);
  if (!get_object_outcome.IsSuccess()) {
    auto error = get_object_outcome.GetError();
    std::string error_str = getErrorMessage(error);
    throw std::runtime_error("DownloadNpyData error: " + error_str);
  } else {
    auto& image_file = get_object_outcome.GetResultWithOwnership().GetBody();
    const char* output_filename = filename.c_str();
    std::ofstream output_file(output_filename, std::ios::binary);
    output_file << image_file.rdbuf();
  }
}

template <typename T>
int GetPreprocessNpyFile(const std::string& npy_filename, std::vector<unsigned long>& input_shape,
                         std::vector<T>& input_data) {
  bool fortran_order;
  npy::LoadArrayFromNumpy(npy_filename, input_shape, fortran_order, input_data);
  return 0;
}

void RunInference(std::string& compiled_model, std::string& npy_name) {
  DLRModelHandle handle;

  std::cout << "CreateDLRModel" << std::endl;
  int dev_type = 1;  // cpu == 1
  int dev_id = 0;
  CreateDLRModel(&handle, compiled_model.c_str(), dev_type, dev_id);

  std::cout << "GetDLRNumOutputs" << std::endl;
  int num_outputs;
  GetDLRNumOutputs(&handle, &num_outputs);
  std::vector<std::vector<float>> outputs;
  for (int i = 0; i < num_outputs; i++) {
    int64_t cur_size = 0;
    int cur_dim = 0;
    GetDLROutputSizeDim(&handle, i, &cur_size, &cur_dim);
    std::vector<float> output(cur_size, 0);
    outputs.push_back(output);
  }

  std::cout << "GetPreprocessNpyFile" << std::endl;
  std::vector<unsigned long> in_shape_ul;
  std::vector<float> input_data;
  GetPreprocessNpyFile<float>(npy_name, in_shape_ul, input_data);
  std::vector<int64_t> in_shape = std::vector<int64_t>(in_shape_ul.begin(), in_shape_ul.end());

  std::cout << "SetDLRInput" << std::endl;
  std::string input_name = "data";
  int64_t input_dimension = in_shape.size();
  SetDLRInput(&handle, input_name.c_str(), in_shape.data(), input_data.data(),
              static_cast<int>(input_dimension));

  std::cout << "RunDLRModel" << std::endl;
  RunDLRModel(&handle);

  std::cout << "GetDLROutput" << std::endl;
  for (int i = 0; i < num_outputs; i++) {
    GetDLROutput(&handle, i, outputs[i].data());
  }

  // print the output for examination
  int idx = 0;
  float max_val = 0;
  std::vector<float> data = outputs[0];
  for (int i = 0; i < data.size(); i++) {
    if (data[i] > max_val) {
      max_val = data[i];
      idx = i;
    }
  }
  std::cout << "max prob: " << max_val << " at " << idx << std::endl;
  std::cout << "Inference complete" << std::endl;
}

int main(int argc, char** argv) {
  // in this example, we're using gluon_imagenet_classifier resnet18
  std::string MODEL_NAME = "resnet18_v1";
  std::string MODEL = MODEL_NAME + ".tar.gz";
  std::string MODEL_ZOO = "gluon_imagenet_classifier";
  std::string FILENAME = "./" + MODEL;

  // this is where we set input bucket
  // this can be changed to your corresponding input
  std::string pretrained_bucket = "neo-ai-dlr-test-artifacts";
  std::string pretrained_key = "neo-ai-notebook/" + MODEL_ZOO + "/" + MODEL;
  std::string target_device = "ml_c4";

  std::string model_name = MODEL_NAME;
  std::string filename = "./" + model_name;

  // s3 bucket for compiled model output
  std::string s3_bucket_name = "windows-demo";

  // compiled model meta
  std::string compiled_filename = "./compiled_model.tar.gz";
  std::string compiled_folder = "./compiled_model";

  if (argc < 2) {
    std::cerr << "invalid argument count, need at least one command" << std::endl;
    return 1;
  }

  std::string cmd = argv[1];
  try {
    if (cmd == "compile") {
      std::string target = argv[2];
      Aws::SDKOptions options;
      Aws::InitAPI(options);

      // make your SDK calls here.
      GetPretrainedModel(pretrained_bucket, pretrained_key, filename);
      UploadModelToS3(model_name, filename, s3_bucket_name);
      CreateIamStep();
      CompileNeoModel(s3_bucket_name, model_name, target);
      GetCompiledModelFromNeo(s3_bucket_name, model_name, target, compiled_filename);

      Aws::ShutdownAPI(options);
    } else if (cmd == "inference") {
      std::string npy_name = "./dog.npy";
      Aws::SDKOptions options;
      Aws::InitAPI(options);

      // download sample npy file
      DownloadNpyData();

      // run inference
      RunInference(compiled_folder, npy_name);

      Aws::ShutdownAPI(options);
    } else {
      std::cerr << "no valid argument command" << std::endl;
    }
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
