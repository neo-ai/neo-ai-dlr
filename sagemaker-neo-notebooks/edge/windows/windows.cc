#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <stdexcept>

// 3rd-party
#include <nlohmann/json.hpp>
// #include <xtensor/xarray.hpp>
#include "npy.hpp"

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
#include <aws/iam/model/ListRolesRequest.h>
#include <aws/iam/model/GetRoleRequest.h>
#include <aws/sagemaker/SageMakerClient.h>
#include <aws/sagemaker/model/CreateCompilationJobRequest.h>
#include <aws/sagemaker/model/InputConfig.h>
#include <aws/sagemaker/model/Framework.h>
#include <aws/sagemaker/model/DescribeCompilationJobRequest.h>

// internal
#include <dlr.h>

using namespace std;
using json = nlohmann::json;

const string ROLE_NAME = "windows-demo-test-role";

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

Aws::IAM::IAMClient getIamClient()
{
    // create IAM role, note that IAM only support global region.
    Aws::IAM::IAMClient iam_client;
    return iam_client;
}

Aws::SageMaker::SageMakerClient getSageMakerClient()
{
    // for tutorial, we set region to us-west-2
    const string region = "us-west-2";
    Aws::String region_aws_s(region.c_str(), region.size());

    // init client
    Aws::Client::ClientConfiguration client_config;
    client_config.region = region_aws_s;
    Aws::SageMaker::SageMakerClient sm_client(client_config);
    return sm_client;
}

void GetPretrainedModel(string bucket_name, string model_name, string filename)
{
    const Aws::String aws_bucket_name(bucket_name.c_str(), bucket_name.size()); // "neo-ai-dlr-test-artifacts";
    const Aws::String aws_object_name(model_name.c_str(), model_name.size());

    // download s3 object
    Aws::S3::S3Client s3_client = getS3Client();

    Aws::S3::Model::GetObjectRequest object_request;
    object_request.SetBucket(aws_bucket_name);
    object_request.SetKey(aws_object_name);

    auto get_object_outcome = s3_client.GetObject(object_request);
    if (!get_object_outcome.IsSuccess())
    {
        auto error = get_object_outcome.GetError();
        cout << "GetModel error: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }
    else
    {
        auto &model_file = get_object_outcome.GetResultWithOwnership().GetBody();

        // download the sample file
        const char *output_filename = filename.c_str();
        std::ofstream output_file(output_filename, std::ios::binary);
        output_file << model_file.rdbuf();
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

void uploadModel(string bucket_name, string model_name, string filename)
{
    Aws::String s3_bucket_name(bucket_name.c_str(), bucket_name.size());
    Aws::String s3_aws_object_name(model_name.c_str(), model_name.size());
    Aws::S3::S3Client s3_client = getS3Client();

    // upload model to s3 bucket
    Aws::S3::Model::PutObjectRequest object_request;
    object_request.SetBucket(s3_bucket_name);
    object_request.SetKey(s3_aws_object_name);

    // TODO verify this
    const shared_ptr<Aws::IOStream> input_data =
        Aws::MakeShared<Aws::FStream>(
            "SampleAllocationTag",
            filename.c_str(), ios_base::in | ios_base::binary);
    object_request.SetBody(input_data);

    // put model to the s3 bucket
    auto put_object_outcome = s3_client.PutObject(object_request);
    if (!put_object_outcome.IsSuccess())
    {
        auto error = put_object_outcome.GetError();
        cout << "UploadModel error: " << error.GetExceptionName() << ": "
             << error.GetMessage() << endl;
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
            if (bucket.GetName().compare(s3_bucket_name) == 0)
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

void uploadModelToS3(string model_name, string filename)
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
        uploadModel(s3_bucket_name, model_name, filename);
    }
}

json getIamPolicy()
{
    json statement;
    statement["Action"] = "sts:AssumeRole";
    statement["Effect"] = "Allow";
    statement["Principal"]["Service"] = "sagemaker.amazonaws.com";

    json policy;
    policy["Statement"] = json::array();
    policy["Statement"].push_back(statement);
    policy["Version"] = "2012-10-17";
    return policy;
}

bool checkIamRole(string iam_role)
{
    Aws::IAM::IAMClient iam_client = getIamClient();
    Aws::String aws_iam_role(iam_role.c_str(), iam_role.size());
    Aws::IAM::Model::GetRoleRequest get_role_request;
    get_role_request.SetRoleName(aws_iam_role);

    auto get_role_response = iam_client.GetRole(get_role_request);
    if (get_role_response.IsSuccess())
    {
        return true;
    }
    else
    {
        auto error = get_role_response.GetError();
        auto errorType = error.GetErrorType();
        if (errorType == Aws::IAM::IAMErrors::NO_SUCH_ENTITY)
        {
            cout << "Role doesn't exist and will create one" << endl;
        }
        else
        {
            cout << "GetIamRole error: " << error.GetExceptionName()
                 << ":" << error.GetMessage() << endl;
            throw 0;
        }
    }
    return false;
}

void createIamRole(string iam_role)
{
    json policy = getIamPolicy();
    const string policy_s = policy.dump();
    const Aws::String aws_role_name(iam_role.c_str(), iam_role.size());
    const Aws::String aws_policy_name(policy_s.c_str(), policy_s.size());

    Aws::IAM::IAMClient iam_client = getIamClient();
    Aws::IAM::Model::CreateRoleRequest create_role_request;
    create_role_request.SetRoleName(aws_role_name);
    create_role_request.SetPath("/");
    create_role_request.SetAssumeRolePolicyDocument(aws_policy_name);

    auto create_role_outcome = iam_client.CreateRole(create_role_request);
    if (!create_role_outcome.IsSuccess())
    {
        auto error = create_role_outcome.GetError();
        cout << "CreateIam error: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }
}

void attachIamPolicy(string iam_role)
{
    const vector<Aws::String> policies = {
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3FullAccess"};
    const Aws::String aws_role_name(iam_role.c_str(), iam_role.size());

    Aws::IAM::IAMClient iam_client = getIamClient();
    for (int i = 0; i < policies.size(); i++)
    {
        Aws::String aws_policy_arn = policies[i];
        Aws::IAM::Model::AttachRolePolicyRequest attach_policy_request;
        attach_policy_request.SetPolicyArn(aws_policy_arn);
        attach_policy_request.SetRoleName(aws_role_name);
        auto attach_policy_outcome = iam_client.AttachRolePolicy(attach_policy_request);
        if (!attach_policy_outcome.IsSuccess())
        {
            auto error = attach_policy_outcome.GetError();
            cout << "ERROR: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
            throw 0;
        }
    }
}

void createIamStep()
{
    if (!checkIamRole(ROLE_NAME))
    {
        createIamRole(ROLE_NAME);
        attachIamPolicy(ROLE_NAME);
    }
}

string getJobName()
{
    chrono::milliseconds ms = chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch());

    std::stringstream ss;
    ss << "pi-demo-" << std::to_string(ms.count());
    return ss.str();
}

Aws::SageMaker::Model::CompilationJobStatus poll_job_status(string job_name)
{
    Aws::SageMaker::SageMakerClient sm_client = getSageMakerClient();
    Aws::SageMaker::Model::DescribeCompilationJobRequest describe_job_request;

    Aws::String aws_job_name(job_name.c_str(), job_name.size());
    describe_job_request.SetCompilationJobName(aws_job_name);
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

void getIamRole(string role_name, Aws::IAM::Model::Role &role)
{
    Aws::IAM::IAMClient iam_client = getIamClient();
    Aws::String aws_iam_role(role_name.c_str(), role_name.size());
    Aws::IAM::Model::GetRoleRequest get_role_request;
    get_role_request.SetRoleName(aws_iam_role);

    auto get_role_response = iam_client.GetRole(get_role_request);
    if (get_role_response.IsSuccess())
    {
        role = get_role_response.GetResult().GetRole();
    }
    else
    {
        auto error = get_role_response.GetError();
        auto errorType = error.GetErrorType();
        if (errorType == Aws::IAM::IAMErrors::NO_SUCH_ENTITY)
        {
            cout << "Role doesn't exist and will create one" << endl;
            throw 0;
        }
        else
        {
            cout << "GetIamRole error: " << error.GetExceptionName()
                 << ":" << error.GetMessage() << endl;
            throw 0;
        }
    }
}

void compileNeoModel(string bucket_name, string model_name)
{
    // set input parameters
    const string input_s3 = "s3://" + bucket_name + "/" + model_name;

    const Aws::SageMaker::Model::Framework framework = Aws::SageMaker::Model::Framework::MXNET;
    const Aws::String aws_s3_uri(input_s3.c_str(), input_s3.size());
    const Aws::String data_shape = "{\"data\":[1,3,224,224]}";
    const Aws::String aws_role_name(ROLE_NAME.c_str(), ROLE_NAME.size());

    const string job_name = getJobName();
    const Aws::String aws_job_name(job_name.c_str(), job_name.size());

    // set input config
    Aws::SageMaker::SageMakerClient sm_client = getSageMakerClient();
    Aws::SageMaker::Model::InputConfig input_config;
    input_config.SetS3Uri(aws_s3_uri);
    input_config.SetFramework(framework);
    input_config.SetDataInputConfig(data_shape);

    // set output config parameters
    const string output_s3 = "s3://" + bucket_name + "/output";
    Aws::String aws_output_s3(output_s3.c_str(), output_s3.size());
    Aws::SageMaker::Model::TargetDevice target_device = Aws::SageMaker::Model::TargetDevice::ml_c4;

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
    if (role.GetArn().empty())
    {
        cout << "role doesn't exist" << endl;
        throw 0;
    }

    // create Neo compilation job
    Aws::SageMaker::Model::CreateCompilationJobRequest create_job_request;
    create_job_request.SetCompilationJobName(aws_job_name);
    create_job_request.SetRoleArn(role.GetArn());
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
        cout << "Waiting for compilation..." << endl;
        auto status = poll_job_status(job_name);
        if (status == Aws::SageMaker::Model::CompilationJobStatus::COMPLETED)
        {
            cout << "Compile successfully" << endl;
            isSuccess = true;
            break;
        }
        else if (status == Aws::SageMaker::Model::CompilationJobStatus::FAILED)
        {
            cout << "Compile fail" << endl;
            throw 0;
        }
        std::this_thread::sleep_for(std::chrono::seconds(30));
        attempt++;
    }

    if (isSuccess == false)
    {
        cout << "Compilation timeout" << endl;
        throw 0;
    }
    cout << "Done!" << endl;
}

void GetCompiledModelFromNeo(string bucket_name, string model_name, string target, string compiled_filename)
{
    const string output_path = "output/" + model_name + "-" + target;

    Aws::String aws_bucket_name(bucket_name.c_str(), bucket_name.size());
    Aws::String aws_object_name(output_path.c_str(), output_path.size());

    // download s3 object
    Aws::S3::S3Client s3_client = getS3Client();

    Aws::S3::Model::GetObjectRequest object_request;
    object_request.SetBucket(aws_bucket_name);
    object_request.SetKey(aws_object_name);

    auto get_object_outcome = s3_client.GetObject(object_request);
    if (!get_object_outcome.IsSuccess())
    {
        auto error = get_object_outcome.GetError();
        cout << "GetModel error: " << error.GetExceptionName() << ": " << error.GetMessage() << endl;
        throw 0;
    }
    else
    {
        auto &model_file = get_object_outcome.GetResultWithOwnership().GetBody();

        // download the sample file
        const char *output_filename = compiled_filename.c_str();
        std::ofstream output_file(output_filename, std::ios::binary);
        output_file << model_file.rdbuf();
    }
}

template <typename T>
int GetPreprocessNpyFile(string npy_filename,
                         std::vector<unsigned long> &input_shape, std::vector<T> &input_data)
{
    npy::LoadArrayFromNumpy(npy_filename, input_shape, input_data);
    return 0;
}

void RunInference(const std::string &compiled_model, const std::string &npy_name)
{
    DLRModelHandle handle;

    int dev_type = 1; // cpu == 1
    int dev_id = 0;
    char *model_path = const_cast<char *>(compiled_model.c_str());
    CreateDLRModel(&handle, model_path, dev_type, dev_id);

    int num_outputs;
    GetDLRNumOutputs(&handle, &num_outputs);
    std::vector<std::vector<double>> outputs;
    for (int i = 0; i < num_outputs; i++)
    {
        int64_t cur_size = 0;
        int cur_dim = 0;
        GetDLROutputSizeDim(&handle, i, &cur_size, &cur_dim);
        std::vector<double> output(cur_size, 0);
        outputs.push_back(output);
    }

    std::vector<unsigned long> in_shape_ul;
    std::vector<double> input_data;
    GetPreprocessNpyFile<double>(npy_name, in_shape_ul, input_data);
    std::vector<int64_t> in_shape =
        std::vector<int64_t>(in_shape_ul.begin(), in_shape_ul.end());

    string input_name = "data";
    int64_t input_dimension = in_shape.size();
    SetDLRInput(&handle, input_name.c_str(), in_shape.data(),
                (float *)input_data.data(), static_cast<int>(input_dimension));


    RunDLRModel(&handle);

    for (int i = 0; i < num_outputs; i++)
    {
        GetDLROutput(&handle, i, (float *)outputs[i].data());
    }

    for (int i = 0; i < outputs.size(); i++)
    {
        std::vector<double> output = outputs.data()[i];
        for (int j = 0; j < output.size(); j++)
        {
            std::cout << std::setprecision(10) << output.data()[j] << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    const string MODEL_NAME = "mobilenetv2_0.25";
    const string MODEL = MODEL_NAME + ".tar.gz";
    const string MODEL_ZOO = "gluon_imagenet_classifier";
    const string FILENAME = "./" + MODEL;

    const string pretrained_bucket = "neo-ai-dlr-test-artifacts";
    const string pretrained_key = "neo-ai-notebook/" + MODEL_ZOO + "/" + MODEL;

    // const string pretrained_bucket = "dlc-nightly-benchmark";
    // const string pretrained_key = "gluon_cv_object_detection/ssd_512_mobilenet1.0_voc.tar.gz";

    // const string pretrained_bucket = "dlc-nightly-benchmark";
    // const string pretrained_key = "gluon_cv_object_detection/ssd_512_mobilenet1.0_voc.tar.gz";

    const string model_name = MODEL_NAME;
    // const string model_name = "ssd_512_mobilenet1.0_voc";
    const string filename = "./" + model_name;
    const string target = "ml_c4";

    const string s3_bucket_name = "windows-demo";
    
    const std::string compiled_filename = "./compiled_model.tar.gz";
    const std::string compiled_folder = "./compiled_model";
    
    const std::string npy_name = "../data/dog.npy";
    
    
    if (argc != 2) {
        std::cerr << "invalid argument count, need at least one command\n";
        return 1;
    }
    
    const string cmd = argv[1];
    if (cmd == "compile") {
        Aws::SDKOptions options;
        Aws::InitAPI(options);
        {
            // make your SDK calls here.
            GetPretrainedModel(pretrained_bucket, pretrained_key, filename);
            uploadModelToS3(model_name, filename);
            createIamStep();
            compileNeoModel(s3_bucket_name, model_name);

            GetCompiledModelFromNeo(s3_bucket_name, model_name, target, compiled_filename);
        }
        Aws::ShutdownAPI(options);
    } else if (cmd == "inference") {
        RunInference(compiled_folder, npy_name);
    } else {
        std::cerr << "no valid argument command\n";
    }

    return 0;
}