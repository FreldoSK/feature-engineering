#include "fill_op.h"

#include <string.h>

#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace {

constexpr int kValueInput = 1;      // input 0 (dims) is ignored
constexpr int kOutput     = 0;

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node)
{
    tflite::MicroContext *mc = tflite::GetMicroContext(context);
    TfLiteTensor *value  = mc->AllocateTempInputTensor(node, kValueInput);
    TfLiteTensor *output = mc->AllocateTempOutputTensor(node, kOutput);

    TF_LITE_ENSURE(context, value != nullptr && output != nullptr);
    TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(value), 0);   // scalar fill value
    TF_LITE_ENSURE_EQ(context, value->type, output->type);
    for (int i = 0; i < output->dims->size; i++)
        TF_LITE_ENSURE(context, output->dims->data[i] > 0);        // static shape

    mc->DeallocateTempTfLiteTensor(value);
    mc->DeallocateTempTfLiteTensor(output);
    return kTfLiteOk;
}

size_t element_size(TfLiteType type)
{
    switch (type) {
        case kTfLiteFloat32:
        case kTfLiteInt32:   return 4;
        case kTfLiteInt16:   return 2;
        case kTfLiteInt8:    return 1;
        default:             return 0;
    }
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node)
{
    const TfLiteEvalTensor *value  = tflite::micro::GetEvalInput(context, node, kValueInput);
    TfLiteEvalTensor       *output = tflite::micro::GetEvalOutput(context, node, kOutput);

    size_t esz = element_size(value->type);
    if (esz == 0) {
        MicroPrintf("FILL: unsupported type %d", value->type);
        return kTfLiteError;
    }

    int count = tflite::micro::ElementCount(*output->dims);
    const char *src = value->data.raw;
    char       *dst = output->data.raw;
    for (int i = 0; i < count; i++)
        memcpy(dst + i * esz, src, esz);
    return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_STATIC_SHAPE_FILL()
{
    return tflite::micro::RegisterOp(nullptr, Prepare, Eval);
}
