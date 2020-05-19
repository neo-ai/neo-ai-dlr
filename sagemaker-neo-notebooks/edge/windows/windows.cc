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
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/iam/IAMClient.h>
#include <aws/iam/model/CreateRoleRequest.h>
#include <aws/iam/model/AttachRolePolicyRequest.h>
#include <aws/sagemaker/SageMakerClient.h>
#include <aws/sagemaker/model/CreateCompilationJobRequest.h>
#include <aws/sagemaker/model/InputConfig.h>
#include <aws/sagemaker/model/Framework.h>
#include <aws/sagemaker/model/DescribeCompilationJobRequest.h>

using namespace std;
using json = nlohmann::json;

const string model_name = "mobilenetv2_0.25";
const string model = model_name + ".tar.gz";
const string model_zoo = "gluon_imagenet_classifier";
const string filename = "./" + model;

const Aws::String role_name = "pi-demo-test-role";

Aws::S3::S3Client getS3Client()
{
    // for tutorial, we set region to us-west-2
    const string region = "us-west-2";
    Aws::String region_aws_s(region.c_str(), region.size());

    // init client
    Aws::Client::ClientConfiguration client_config;
    client_config.region = region_aws_s;
    Aws::S3::S3Client s3_client(client_config);
    return s3_client;
}

void getPretrainedModel()
{
    const string object = "neo-ai-notebook/" + model_zoo + "/" + model;
    const Aws::String bucket_name = "neo-ai-dlr-test-artifacts";
    const Aws::String object_name(object.c_str(), object.size());

    // download s3 object
    Aws::S3::S3Client s3_client = getS3Client();

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

void createBucket(string bucket_name)
{
    // create bucket
    const Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
    Aws::S3::Model::CreateBucketRequest request;
    request.SetBucket(s3_bucket_name);

    // for demo purpose: we set that to us-west-2
    const Aws::S3::Model::BucketLocationConstraint &region = Aws::S3::Model::BucketLocationConstraint::us_west_2;
    Aws::S3::S3Client s3_client = getS3Client();
    Aws::S3::Model::CreateBucketConfiguration bucket_config;
    bucket_config.SetLocationConstraint(region);
    request.SetCreateBucketConfiguration(bucket_config);

    // making request call
    auto outcome = s3_client.CreateBucket(request);
    if (!outcome.IsSuccess())
    {
        auto err = outcome.GetError();
        cout << "ERROR: CreateBucket: " << err.GetExceptionName() << ": " << err.GetMessage() << endl;
        throw 0;
    }
}

void uploadModel(string bucket_name, string model_name)
{
    Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
    Aws::String s3_object_name(model_name.c_str(), model_name.size());
    Aws::S3::S3Client s3_client = getS3Client();

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
}

bool checkBucketExist(string bucket_name)
{
    Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
    Aws::S3::S3Client s3_client = getS3Client();
    auto list_bucket_resp = s3_client.ListBuckets();
    if (list_bucket_resp.IsSuccess())
    {
        Aws::Vector<Aws::S3::Model::Bucket> bucket_list = list_bucket_resp.GetResult().GetBuckets();
        for (auto const &bucket : bucket_list)
        {
            if (bucket.GetName().compare(bucket_name) == 0)
            {
                return true;
            }
        }
    }
    else
    {
        auto error = list_bucket_resp.GetError();
        cout << "ListBucket error: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }
    return false;
}

bool checkModelExist(string bucket_name, string model_name)
{
    Aws::String aws_bucket_name(bucket_name.c_str(), bucket_name.size());
    Aws::String aws_model_name(model_name.c_str(), model_name.size());

    Aws::S3::S3Client s3_client = getS3Client();
    Aws::S3::Model::ListObjectsRequest objects_request;
    objects_request.WithBucket(aws_bucket_name);

    auto list_objects_outcome = s3_client.ListObjects(objects_request);
    if (list_objects_outcome.IsSuccess())
    {
        Aws::Vector<Aws::S3::Model::Object> object_list = list_objects_outcome.GetResult().GetContents();
        for (auto const &s3_object : object_list)
        {
            if (s3_object.GetKey().compare(aws_model_name) == 0)
            {
                return true;
            }
        }
    }
    else
    {
        std::cout << "ListObjects error: " << list_objects_outcome.GetError().GetExceptionName()
                  << " " << list_objects_outcome.GetError().GetMessage() << std::endl;
        throw 0;
    }

    return false;
}

void uploadModelToS3()
{
    const string s3_bucket_name = "windows-demo";

    // first create bucket
    bool isBucketExist = checkBucketExist(s3_bucket_name);
    if (!isBucketExist)
    {
        createBucket(s3_bucket_name);
    }

    // then create/upload model
    if (!checkModelExist(s3_bucket_name, model_name))
    {
        uploadModel(s3_bucket_name, model_name);
    }
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

Aws::SageMaker::Model::CompilationJobStatus poll_job_status(string job_name)
{
    Aws::SageMaker::SageMakerClient sm_client;
    Aws::SageMaker::Model::DescribeCompilationJobRequest describe_job_request;

    Aws::String job_name_aws_s(job_name.c_str(), job_name.size());
    describe_job_request.SetCompilationJobName(job_name_aws_s);
    auto describe_job_resp = sm_client.DescribeCompilationJob(describe_job_request);

    if (!describe_job_resp.IsSuccess())
    {
        auto error = describe_job_resp.GetError();
        cout << "Describe Job Error: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }

    auto result = describe_job_resp.GetResult();
    return result.GetCompilationJobStatus();
}

void compileNeoModel()
{
    // set input parameters
    const string bucket = "";
    const string s3_location = "s3://" + bucket + "/output";

    const Aws::SageMaker::Model::Framework framework = Aws::SageMaker::Model::Framework::MXNET;
    const Aws::String s3_loc_aws_s(s3_location.c_str(), s3_location.size());
    const Aws::String data_shape = "{\"data\":[1,3,224,224]}";

    const string job_name = getJobName();
    const Aws::String job_name_aws_s(job_name.c_str(), job_name.size());

    // set input config
    Aws::SageMaker::SageMakerClient sm_client;
    Aws::SageMaker::Model::InputConfig input_config;
    input_config.SetS3Uri(s3_loc_aws_s);
    input_config.SetFramework(framework);
    input_config.SetDataInputConfig(data_shape);

    // set output config parameters
    const string s3_output_location = "s3://" + bucket + "";
    Aws::String s3_output_location_aws_s(s3_output_location.c_str(), s3_output_location.size());
    Aws::SageMaker::Model::TargetDevice target_device = Aws::SageMaker::Model::TargetDevice::ml_c4;

    // set output config
    Aws::SageMaker::Model::OutputConfig output_config;
    output_config.SetS3OutputLocation(s3_output_location_aws_s);
    output_config.SetTargetDevice(target_device);

    // set stopping condition
    int max_runtime_in_sec = 900;
    Aws::SageMaker::Model::StoppingCondition stopping_condition;
    stopping_condition.SetMaxRuntimeInSeconds(max_runtime_in_sec);

    // create Neo compilation job
    Aws::SageMaker::Model::CreateCompilationJobRequest create_job_request;
    create_job_request.SetCompilationJobName(job_name_aws_s);
    create_job_request.SetRoleArn(role_name);
    create_job_request.SetInputConfig(input_config);
    create_job_request.SetOutputConfig(output_config);
    create_job_request.SetStoppingCondition(stopping_condition);

    auto compilation_resp = sm_client.CreateCompilationJob(create_job_request);
    if (!compilation_resp.IsSuccess())
    {
        auto error = compilation_resp.GetError();
        cout << "Create job error: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }

    // poll job for validation
    bool isSuccess = false;
    int attempt = 0;
    while (attempt < 10)
    {
        auto status = poll_job_status(job_name);
        if (status == Aws::SageMaker::Model::CompilationJobStatus::COMPLETED)
        {
            std::cout << "Compile successfully!";
            isSuccess = true;
            break;
        }
        else if (status == Aws::SageMaker::Model::CompilationJobStatus::FAILED)
        {
            std::cout << "Compile fail!";
            throw 0;
        }
        std::this_thread::sleep_for(std::chrono::seconds(30));
        attempt++;
    }

    if (isSuccess == false)
    {
        std::cout << "Compilation timeout";
        throw 0;
    }
    std::cout << "Done!";
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
        getPretrainedModel();
        uploadModelToS3();
        // createIamRole()
    }
    Aws::ShutdownAPI(options);

    return 0;
}