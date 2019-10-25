# Neo AI DLR image classification Android example application

## Overview

This is an example application for [neo-ai-dlr](https://github.com/neo-ai/neo-ai-dlr)
on Android. It uses
Image classification models from different frameworks
to continuously classify whatever it sees from the device's back camera.
Inference is performed using the DLR Java API. The demo app
classifies frames in real-time, displaying the top most probable
classifications. It allows the user to choose between MobileNet or ResNet
models from different frameworks

These instructions walk you through building and
running the demo on an Android device. For an explanation of the source, see
[DLR Android image classification example](https://github.com/neo-ai/neo-ai-dlr/tree/master/examples/android/image_classification).

<!-- TODO(b/124116863): Add app screenshot. -->

### Models

App Uses the following Neo pre-compiled models:
* gluoncv_mobilenet_v2_075
* gluoncv_mobilenet_v2_100
* gluoncv_resnet18_v2
* gluoncv_resnet50_v2
* keras_mobilenet_v2
* tf_mobilenet_v1_100


## Requirements

*   Android Studio 3.2 (installed on a Linux, Mac or Windows machine)

*   Android device in
    [developer mode](https://developer.android.com/studio/debug/dev-options)
    with USB debugging enabled

*   USB cable (to connect Android device to your computer)

## Build and run

### Step 1. Clone Neo AI DLR source code

Clone the neo-ai-dlr GitHub repository to your computer to get the demo
application.

```
git clone --recursive https://github.com/neo-ai/neo-ai-dlr.git

cd neo-ai-dlr/examples/android/image_classification
```

### Step 2. Download dlr-release.aar
Download `dlr-release.aar` file by running
```
./download-dependencies.sh
```
dlr-release.aar will be downloaded to `dlr-release` folder.

### Step 3. Download Neo pre-compiled models
Download pre-compiled models by running gradle task downloadModels.
Set device arch in `app/download.gradle` (`arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`) before running the command
```
./gradlew downloadModels
```
The models will be downloaded and extracted to app assets folder.

### Step 4. Open the image_classification project in Android Studio.
To do this, open Android Studio and select `Open an existing project`, setting the folder to
`neo-ai-dlr/examples/android/image_classification`

<img src="images/classifydemo_img1.png?raw=true" />


### Step 5. Build the Android Studio project

Select `Build -> Make Project` and check that the project builds successfully.
You will need Android SDK configured in the settings. You'll need at least SDK
version 23. The `build.gradle` file will prompt you to download any missing
libraries.

<img src="images/classifydemo_img4.png?raw=true" style="width: 40%" />

<img src="images/classifydemo_img2.png?raw=true" style="width: 60%" />

<aside class="note"><b>Note:</b><p>`download-dependencies.sh` downloads the latest
dlr-release.aar library.</p><p>If you see a build error related to
compatibility with DLR Java API (for example, `method X is
undefined for type DLR`), there has likely been a backwards compatible
change to the API. You will need to run `git pull` in the neo-ai-dlr repo to
obtain a version that is compatible with the latest dlr-release.aar.</p></aside>

### Step 6. Install and run the app

Connect the Android device to the computer and be sure to approve any ADB
permission prompts that appear on your phone. Select `Run -> Run app.` Select
the deployment target in the connected devices to the device on which the app
will be installed. This will install the app on the device.

<img src="images/classifydemo_img5.png?raw=true" style="width: 60%" />

<img src="images/classifydemo_img6.png?raw=true" style="width: 70%" />

<img src="images/classifydemo_img7.png?raw=true" style="width: 40%" />

<img src="images/classifydemo_img8.png?raw=true" style="width: 80%" />

To test the app, open the app called `Neo Classify` on your device. When you run
the app the first time, the app will request permission to access the camera.
Re-installing the app may require you to uninstall the previous installations.

## Assets folder
_Do not delete the assets folder content_. If you explicitly deleted the
folders/files, download Neo pre-compiled models as described in Step 3.
