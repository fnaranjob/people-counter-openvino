# Project Write-Up

This project's solution used the faster_rcnn_inception_v2 tensorflow model, trained with the coco dataset

[model github repository here](https://github.com/opencv/open_model_zoo/blob/master/models/public/faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.md) 

## Explaining Custom Layers

OpenVino's model optimizer and Inference Engine support a variety of layer types, frameworks and devices, however, some models can have layers for which there isn't any available implementation, these models can still be used and the unsupported layers can be treated as "custom layers"

To be able to work with custom layers, extensions need to be created both for the Model Optimizer and the Inference Engine

#### Model Optimizer extensions

The model optimizer needs 2 extensions, a custom layer extractor that identifies and extracts the layer parameters, and a custom layer operation that specifies the layer's attributes and computes its output shape for every instance of the custom layer in the model

#### Inference engine extensions

Extensions to work with custom layers are only supported for CPU and GPU devices

* For CPU, a compiled shared library (.dll or .so) needs to be provided to the Inference Engine

* For GPU, openCl source code (.cl) and layer description file (.xml) need to be provided to the Inference Engine

These extensions can be created with help of the Model Extension Generator which will generate templates for the specified framework and device, these must be edited to include the source code, detailed instructions can be found in the [OpenVino Docs](https://docs.openvinotoolkit.org/latest/_docs_HOWTO_Custom_Layers_Guide.html) 

## Comparing Model Performance

To be able to compare the original model with the IR representation that was ultimately used for the project, inference was made on a few sample images using both Tensorflow and OpenVino, the results were:

* Model accuracy loss was imperceptible, even though the original model's precision was FP32 and it was converted to IR with FP16 precision

* The size of the model was lowered significantly (roughly 50%) from 53 MB for the original Tensorflow to around 26 MB for the IR

* Inference time didn't change with the device used (Intel core I5, 4th gen for both tests), performance was poor here as inference time was around 350 msec, testing with GPU or VPU is recommended to improve performance

## Model Use Cases

Some of the potential use cases of the people counter app are:

#### Smart staff deployment at crowded shops

Large crowded businesses like supermarkets and banks could benefit from an AI enabled people counting app that can alert the staff if there is a sudden spike in the amount of people entering the shop, stats could be gathered and presented timely so that the store staff can be deployed fast and large queues can be reduced.

#### Customer behaviour stats for marketing purposes

AI enabled people couting apps can be used at retail stores to gather statistics about customer behaviour that later can help the marketing team optimize placement of products, store layout, etc.

#### Social distancing alarm

COVID-19 made it very important for people to keep some distance, the people counting app can effectively rise alarms in case too many people are too close, this can be deployed at normally crowded places like airports, stores, banks, parks, etc.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

* Poor lighting conditions will very likely lower the model's detection accuracy, this can cause false detections, people on frame not detected or intermittent detection of the same person (this has been filtered to an extent on the solution but if it is too bad it can damage the statistics gathered by the app)

* Camera focal length is very important for the used model's accuracy, the model was tested on several pictures of people, and for the ones on which some individuals were out of focus, they couldn't be detected reliably, detection probability threshold could be lowered to mitigate this effect but it would only work to an extent, it's preferrable if the whole frame stays sharp and focused, different models may show better results with out of focus subjects

* Image size might also affect model accuracy, the smallest the image the least amount of information the model has available to work with, this was not tested though