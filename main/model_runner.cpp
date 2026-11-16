#include "model_runner.h"

#include <math.h>
#include <stdio.h>
#include <inttypes.h>

#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "fill_op.h"
#include "pv_features.h"

namespace {

// 120 kB fails: AllocateTensors needs planner scratch on top of ~101 kB.
constexpr size_t ARENA_SIZE = 128 * 1024;
alignas(16) uint8_t tensor_arena[ARENA_SIZE];

using OpResolver = tflite::MicroMutableOpResolver<16>;

// Operators used by the models (+ a few for a fixed-batch export).
OpResolver &op_resolver()
{
    static OpResolver resolver;
    static bool registered = false;
    if (!registered) {
        resolver.AddAdd();
        resolver.AddFill(Register_STATIC_SHAPE_FILL());
        resolver.AddFullyConnected();
        resolver.AddLogistic();
        resolver.AddMul();
        resolver.AddPack();
        resolver.AddShape();
        resolver.AddSplit();
        resolver.AddStridedSlice();
        resolver.AddTanh();
        resolver.AddTranspose();
        resolver.AddUnpack();
        // needed by a model exported with a fixed batch size (fused LSTM op)
        resolver.AddUnidirectionalSequenceLSTM();
        resolver.AddReshape();
        resolver.AddQuantize();
        resolver.AddDequantize();
        registered = true;
    }
    return resolver;
}

int8_t quantize(float v, QuantParams q)
{
    int32_t x = (int32_t)lrintf(v / q.scale) + q.zero_point;
    if (x < -128) x = -128;
    if (x >  127) x =  127;
    return (int8_t)x;
}

float dequantize(int8_t x, QuantParams q)
{
    return ((float)x - (float)q.zero_point) * q.scale;
}

// normalize -> quantize, one window [WIN_ROWS][n_features]
void write_input(const ModelSpec &spec, const float *features, int8_t *dst)
{
    for (int t = 0; t < WIN_ROWS; t++) {
        for (int j = 0; j < spec.n_features; j++) {
            float scaled = *features++ * spec.feat_scale[j] + spec.feat_min[j];
            *dst++ = quantize(scaled, spec.input);
        }
    }
}

// dequantize -> de-normalize, clipped at 0 kW
void read_output(const ModelSpec &spec, const int8_t *src, float *kW)
{
    for (int h = 0; h < spec.n_outputs; h++) {
        float y = dequantize(src[h], spec.output);
        float p = (y - spec.target_min) / spec.target_scale;
        kW[h] = p < 0.0f ? 0.0f : p;
    }
}

bool check_tensors(const ModelSpec &spec, const TfLiteTensor *in, const TfLiteTensor *out)
{
    if (in->type != kTfLiteInt8 || out->type != kTfLiteInt8) {
        printf("%s: expected int8 tensors\n", spec.name);
        return false;
    }
    const size_t in_expected = WIN_ROWS * spec.n_features;
    if (in->bytes != in_expected) {
        printf("%s: input tensor is %u B, expected %u\n",
               spec.name, (unsigned)in->bytes, (unsigned)in_expected);
        return false;
    }
    if (out->bytes != (size_t)spec.n_outputs) {
        printf("%s: output tensor has %u values, expected %d -- fix MODEL_N_OUTPUTS\n",
               spec.name, (unsigned)out->bytes, spec.n_outputs);
        return false;
    }
    return true;
}

}  // namespace

bool run_model(const ModelSpec &spec, const float *features, int n_runs,
               Forecast &result)
{
    result = Forecast{};
    if (n_runs * spec.n_outputs > N_HORIZONS) {
        printf("%s: %d runs x %d outputs do not fit %d horizons\n",
               spec.name, n_runs, spec.n_outputs, N_HORIZONS);
        return false;
    }

    const tflite::Model *model = tflite::GetModel(spec.flatbuffer);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("%s: schema mismatch\n", spec.name);
        return false;
    }

    tflite::MicroInterpreter interpreter(model, op_resolver(), tensor_arena, ARENA_SIZE);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printf("%s: AllocateTensors failed -- raise ARENA_SIZE\n", spec.name);
        return false;
    }

    TfLiteTensor *in  = interpreter.input(0);
    TfLiteTensor *out = interpreter.output(0);
    if (!check_tensors(spec, in, out))
        return false;

    const int window_size = WIN_ROWS * spec.n_features;
    int64_t total_us = 0;

    for (int r = 0; r < n_runs; r++) {
        write_input(spec, features + r * window_size, in->data.int8);

        int64_t t0 = esp_timer_get_time();
        TfLiteStatus status = interpreter.Invoke();
        total_us += esp_timer_get_time() - t0;

        if (status != kTfLiteOk) {
            printf("%s: Invoke failed\n", spec.name);
            return false;
        }
        read_output(spec, out->data.int8, result.kW + r * spec.n_outputs);
    }

    result.ok = true;
    result.us_per_run = total_us / n_runs;
    printf("%s: arena used %u B, %d run(s), %" PRId64 " us per run\n",
           spec.name, (unsigned)interpreter.arena_used_bytes(), n_runs, result.us_per_run);
    return true;
}
