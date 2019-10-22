package com.amazon.neo.dlr;

public class DLR {

    static {
        System.loadLibrary("dlr");
    }

    public static native int GetDLRNumInputs(long handle);

    public static native int GetDLRNumWeights(long handle);

    public static native String GetDLRInputName(long handle, int index);

    public static native String GetDLRWeightName(long handle, int index);

    public static native int SetDLRInput(long jhandle, String jname,
                                         long[] shape, float[] input, int dim);

    public static native int GetDLRInput(long handle, String jname, float[] input);

    public static native int GetDLROutputShape(long jhandle, int index, long[] shape);

    public static native int GetDLROutput(long handle, int index, float[] output);

    public static native int GetDLROutputDim(long handle, int index);

    public static native long GetDLROutputSize(long handle, int index);

    public static native int GetDLRNumOutputs(long handle);

    public static native long CreateDLRModelFromTFLite(String model_path, int threads, int use_nnapi);

    public static native long CreateDLRModel(String model_path, int dev_type, int dev_id);

    public static native int DeleteDLRModel(long handle);

    public static native int RunDLRModel(long handle);

    public static native String DLRGetLastError();

    public static native String GetDLRBackend(long handle);

    public static native int SetDLRNumThreads(long jhandle, int threads);

    public static native int UseDLRCPUAffinity(long jhandle, boolean use);
}
