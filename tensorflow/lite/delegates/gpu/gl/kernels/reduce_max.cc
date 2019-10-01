/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/gl/kernels/reduce_max.h"

#include <memory>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ReduceMax : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto attr = absl::any_cast<const ReduceMaxAttributes&>(
        ctx.node->operation.attributes);

    auto axis_count = 0;

    if (attr.axis.data.back() == 1) {
      axis_count = input->tensor.shape.w;
    }
    else if (attr.axis.data.back() == 2) {
      axis_count = input->tensor.shape.h;
    }
    else if (attr.axis.data.back() == 3) {
      axis_count = input->tensor.shape.c;
    }

    std::vector<std::string> indexers_output {
      "[0, gid.y, gid.z]", "[gid.x, 0, gid.z]", "[gid.x, gid.y, 0]"
    };

    std::vector<std::string> indexers_input {
      "[i, gid.y, gid.z]", "[gid.x, i, gid.z]", "[gid.x, gid.y, i]"
    };

    auto idx_from_axis = attr.axis.data.back() - 1;

    auto code =
    "for (int i = 0; i < $axis_count$; i++) {"
    "  if ($input_data_0" + indexers_input[idx_from_axis] + "$[0] > $output_data_0" + indexers_output[idx_from_axis] + "$[0]) {"
    "    $output_data_0" + indexers_output[idx_from_axis] + "$ = $input_data_0" + indexers_input[idx_from_axis] + "$;"
    "  }"
    "}"
    "$output_data_0" + indexers_output[idx_from_axis] + "$ = $input_data_0" + indexers_output[idx_from_axis] + "$;";

    std::vector<UniformParameter> parameters {
        UniformParameter{"axis_count", UniformParameter::ValueType(axis_count)},
    };

    *generated_code = GeneratedCode{
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };

    return OkStatus();
  }
};
}

int SelectMultiplier(int32_t input_width,
                     const NodeShader::GenerationContext& ctx) {
  std::vector<int> multipliers = {4, 2};
  if (!ctx.compiler_options.allow_precision_loss &&
      ctx.gpu_info->type == GpuType::MALI) {
    multipliers = {2};
  }
  for (int i : multipliers) {
    if (input_width % i == 0) {
      return i;
    }
  }
  return 1;
}

std::unique_ptr<NodeShader> NewReduceMaxShader() {
  return absl::make_unique<ReduceMax>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
