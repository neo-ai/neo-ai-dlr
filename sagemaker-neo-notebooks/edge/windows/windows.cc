#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <ctime>

// 3rd-party
#include <nlohmann/json.hpp>

// aws-sdk
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/iam/IAMClient.h>
#include <aws/iam/model/CreateRoleRequest.h>
#include <aws/iam/model/AttachRolePolicyRequest.h>
#include <aws/sagemaker/SageMakerClient.h>
#include <aws/sagemaker/model/CreateCompilationJobRequest.h>
#include <aws/sagemaker/model/InputConfig.h>
#include <aws/sagemaker/model/Framework.h>

using namespace std;
using json = nlohmann::json;

const string model_name = "mobilenetv2_0.25";
const string model = model_name + ".tar.gz";
const string model_zoo = "gluon_imagenet_classifier";
const string filename = "./" + model;

const Aws::String role_name = "pi-demo-test-role";

void getPretrainedModel()
{
    const string object = "neo-ai-notebook/" + model_zoo + "/" + model;
    const Aws::String bucket_name = "neo-ai-dlr-test-artifacts";
    const Aws::String object_name(object.c_str(), object.size());

    // download s3 object
    Aws::S3::S3Client s3_client;

    Aws::S3::Model::GetObjectRequest object_request;
    object_request.SetBucket(bucket_name);
    object_request.SetKey(object_name);

    auto get_object_outcome = s3_client.GetObject(object_request);
    if (!get_object_outcome.IsSuccess())
    {
        auto error = get_object_outcome.GetError();
        cout << "ERROR: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }
}

void uploadModelToS3()
{
    const Aws::S3::Model::BucketLocationConstraint &region = Aws::S3::Model::BucketLocationConstraint::us_east_1;
    const Aws::String s3_bucket_name = "demo";
    const Aws::String s3_object_name(filename.c_str(), filename.size());

    // first create model
    Aws::S3::Model::CreateBucketRequest request;
    request.SetBucket(s3_bucket_name);

    // for demo purpose: we set that to us-west-2
    Aws::S3::S3Client s3_client;
    Aws::S3::Model::CreateBucketConfiguration bucket_config;
    bucket_config.SetLocationConstraint(region);
    request.SetCreateBucketConfiguration(bucket_config);
    auto outcome = s3_client.CreateBucket(request);
    if (!outcome.IsSuccess())
    {
        auto err = outcome.GetError();
        cout << "ERROR: CreateBucket: " << err.GetExceptionName() << ": " << err.GetMessage() << endl;
        throw 0;
    }

    // upload model to s3 bucket
    Aws::S3::Model::PutObjectRequest object_request;
    object_request.SetBucket(s3_bucket_name);
    object_request.SetKey(s3_object_name);
    // TODO verify this
    // const shared_ptr<Aws::IOStream> input_data =
    //     Aws::MakeShared<Aws::FStream>("SampleAllocationTag", filename.c_str(), ios_base::in | ios_base::binary);
    // object_request.SetBody(input_data);

    // put model to the s3 bucket
    auto put_object_outcome = s3_client.PutObject(object_request);
    if (!put_object_outcome.IsSuccess())
    {
        auto error = put_object_outcome.GetError();
        cout << "ERROR: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }

    return;
}

json getIamPolicy()
{
    json statement;
    statement["Action"] = "sts:AssumeRole";
    statement["Effect"] = "Allow";
    statement["Principal"] = {"Service", "sagemaker.amazonaws.com"};

    json policy;
    policy["Statement"] = json::array();
    policy["Statement"].push_back(statement);
    policy["Version"] = "2012-10-17";
    return policy;
}

void createIamRole()
{
    json policy = getIamPolicy();
    const string policy_s = policy.dump();
    const Aws::String policy_aws_s(policy_s.c_str(), policy_s.size());

    Aws::IAM::IAMClient iam_client;
    Aws::IAM::Model::CreateRoleRequest create_role_request;
    create_role_request.SetRoleName(role_name);
    create_role_request.SetPath("/");
    create_role_request.SetAssumeRolePolicyDocument(policy_aws_s);

    auto create_role_outcome = iam_client.CreateRole(create_role_request);
    if (!create_role_outcome.IsSuccess())
    {
        auto error = create_role_outcome.GetError();
        cout << "ERROR: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }
}

void attachIamPolicy()
{
    const vector<Aws::String> policies = {
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess"};

    Aws::IAM::IAMClient iam_client;
    for (int i = 0; i < policies.size(); i++)
    {
        Aws::String policy_arn = policies[i];
        Aws::IAM::Model::AttachRolePolicyRequest attach_policy_request;
        attach_policy_request.SetPolicyArn(policy_arn);
        attach_policy_request.SetRoleName(role_name);
        auto attach_policy_outcome = iam_client.AttachRolePolicy(attach_policy_request);
        if (!attach_policy_outcome.IsSuccess())
        {
            auto error = attach_policy_outcome.GetError();
            cout << "ERROR: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
            throw 0;
        }
    }
}

string getJobName()
{

    chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch());

    std::stringstream ss;
    ss << "pi-demo-" << std::to_string(ms.count()) << endl;
    return ss.str();
}

void compileModel()
{
    const string bucket = "";
    const string s3_location = "s3://" + bucket + "/output";

    const Aws::SageMaker::Model::Framework framework = Aws::SageMaker::Model::Framework::MXNET;
    const Aws::String s3_loc_aws_s(s3_location.c_str(), s3_location.size());
    const Aws::String data_shape = "{\"data\":[1,3,224,224]}";
    const string target_device = "rasp3b";

    const string job_name = getJobName();
    const Aws::String job_name_aws_s(job_name.c_str(), job_name.size());

    Aws::SageMaker::SageMakerClient sm_client;
    Aws::SageMaker::Model::InputConfig input_config;
    input_config.SetS3Uri(s3_loc_aws_s);
    input_config.SetFramework(framework);

    Aws::SageMaker::Model::CreateCompilationJobRequest create_job_request;
    create_job_request.SetCompilationJobName(job_name_aws_s);
    create_job_request.SetRoleArn(role_name);
    // create_job_request.SetInputConfig();
}

void getModelFromS3()
{
}

void inferenceModel()
{
}

int main(int argc, char **argv)
{
    Aws::SDKOptions options;
    Aws::InitAPI(options);
    {
        // make your SDK calls here.
        // getPretrainedModel();
        // uploadModelToS3()
        // createIamRole()
    }
    Aws::ShutdownAPI(options);
    return 0;
}