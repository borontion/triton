/* Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef DIALECT_NVWS_IR_DIALECT_H_
#define DIALECT_NVWS_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

#include "nvidia/include/Dialect/NVWS/IR/NVWSAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "nvidia/include/Dialect/NVWS/IR/NVWSAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "nvidia/include/Dialect/NVWS/IR/Types.h.inc"

#define GET_OP_CLASSES
#include "nvidia/include/Dialect/NVWS/IR/NVWSOpInterfaces.h.inc"
#include "nvidia/include/Dialect/NVWS/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace nvws {} // namespace nvws
} // namespace triton
} // namespace mlir

#endif // DIALECT_NVWS_IR_DIALECT_H_
