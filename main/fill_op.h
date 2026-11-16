// =============================================================================
// FILL operator that tolerates a non-constant dims tensor.
//
// A Keras LSTM exported with a dynamic batch builds its zero initial state as
// SHAPE -> STRIDED_SLICE -> PACK -> FILL. The stock TFLM FILL rejects that
// ("Non-constant >dims< tensor is not supported"). The converter still stores
// the static output shape ([1, units]) in the model, so this FILL ignores the
// dims tensor and fills the output tensor as declared in the model.
// =============================================================================
#pragma once

#include "tensorflow/lite/micro/micro_common.h"

TFLMRegistration Register_STATIC_SHAPE_FILL();
