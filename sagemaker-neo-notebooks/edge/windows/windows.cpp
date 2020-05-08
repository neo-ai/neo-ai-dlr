#include <iostream>
#include <string>

// 3rd-party
#include <nlohmann/json.hpp>

// aws-sdk
#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/iam/model/CreateRoleRequest.h>

#include <aws/iam/IAMClient.h>

using namespace std;
using json = nlohmann::json;

Aws::S3::S3Client s3_client;
Aws::IAM::IAMClient iam_client;

const string model_name = "mobilenetv2_0.25";
const string model = model_name + ".tar.gz";
const string model_zoo = "gluon_imagenet_classifier";
const string filename = "./" + model;

int getPretrainedModel()
{
    const string object = "neo-ai-notebook/" + model_zoo + "/" + model;
    const Aws::String bucket_name = "neo-ai-dlr-test-artifacts";
    const Aws::String object_name(object.c_str(), object.size());

    // download s3 object
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

    return 0;
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
    const Aws::String role_name = "pi-demo-test-role";

    json policy = getIamPolicy();
    const string policy_s = policy.dump();
    const Aws::String policy_aws_s(policy_s.c_str(), policy_s.size());

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

void compileModel()
{
}

void getModelFromS3()
{
}

void inferenceModel()
{
}

int main(int argc, char **argv)
{
    // getPretrainedModel();
    // uploadModelToS3()
}