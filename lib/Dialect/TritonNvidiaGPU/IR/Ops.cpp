/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.cpp.inc"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

// -- WarpGroupDotOp --
LogicalResult WarpGroupDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc = cast<TensorOrMemDesc>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult WarpGroupDotOp::verify() {
  auto resTy = getD().getType();
  auto nvmmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding());
  if (!nvmmaEnc || !nvmmaEnc.isHopper())
    return emitOpError("WGMMA result layout must be Hopper NVMMA");

  if (!isa<NVMMASharedEncodingAttr, DotOperandEncodingAttr>(
          getA().getType().getEncoding()))
    return emitOpError("WGMMA A operand must have NVMMA shared or dot layout");
  if (!isa<NVMMASharedEncodingAttr>(getB().getType().getEncoding()))
    return emitOpError("WGMMA B operand must have NVMMA shared layout");

  auto numWarps = gpu::lookupNumWarps(getOperation());
  if (numWarps % 4)
    return emitOpError("WGMMA requires num_warps to be divisible by 4");

  auto retShapePerCTA = getShapePerCTA(resTy);
  int rank = retShapePerCTA.size();
  if (rank != 2)
    return emitOpError("WGMMA result shape must be 2D");
  if (retShapePerCTA[0] % 64 != 0)
    return emitOpError("WGMMA result M dimension must be divisible by 64");
  if (retShapePerCTA[1] % 8 != 0)
    return emitOpError("WGMMA result N dimension must be divisible by 8");

  auto aElemTy = getA().getType().getElementType();
  if (!(llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy) ||
        aElemTy.isInteger(8) || aElemTy.isF16() || aElemTy.isBF16() ||
        aElemTy.isF32()))
    return emitOpError("WGMMA result element type must be F16, BF16, F32, "
                       "F8E5M2, F8E4M3FN, or integer type");

  if (getMaxNumImpreciseAcc() < 32 &&
      (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy)) &&
      resTy.getElementType().isF32()) {
    return emitOpError("Cannot use F32 as the accumulator element type when "
                       "the max_num_imprecise_acc is less than 32");
  }
  return success();
}

void WarpGroupDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a, SharedMemory::get());
  if (isa<MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b, SharedMemory::get());
}

bool WarpGroupDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto aElTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(aElTy);
  bool accFP32 =
      cast<triton::gpu::TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

bool WarpGroupDotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

// -- WarpGroupDotWaitOp --
LogicalResult WarpGroupDotWaitOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return mlir::success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- InvalBarrierOp --
LogicalResult InvalBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- BarrierExpectOp --
LogicalResult BarrierExpectOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

// -- AsyncTMACopyGlobalToLocalOp --
LogicalResult AsyncTMACopyGlobalToLocalOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();
  if (getCoord().size() < 1 || getCoord().size() > 5)
    return emitOpError("TMA copies must have between 1 and 5 coordinates");
  if (!getResult().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

// -- AsyncTMAGatherOp --
LogicalResult AsyncTMAGatherOp::verify() {
  if (failed(verifyBarrierType(*this, getBarrier().getType())))
    return failure();

  triton::gpu::MemDescType resultType = getResult().getType();
  if (!resultType.getMutableMemory())
    return emitOpError("cannot store into immutable memory");
  return DescriptorGatherOp::verifyResultType(*this, resultType,
                                              getXOffsets().getType());
}

// -- AsyncTMAScatter --
LogicalResult AsyncTMAScatterOp::verify() {
  return DescriptorGatherOp::verifyResultType(*this, getSrc().getType(),
                                              getXOffsets().getType());
}

// -- TCGen5MMAOp --

// barrier-and-pred := `,` ssa-value `[` ssa-value `]`
// barriers-and-preds := (barrier-and-pred)*
static ParseResult
parseBarriersAndPreds(OpAsmParser &p,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &barriers,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preds) {
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseOperand(barriers.emplace_back()) || p.parseLSquare() ||
        p.parseOperand(preds.emplace_back()) || p.parseRSquare())
      return failure();
  }
  return success();
}
static void printBarriersAndPreds(OpAsmPrinter &p, Operation *op,
                                  OperandRange barriers, OperandRange preds) {
  assert(barriers.size() == preds.size());
  for (auto [barrier, pred] : llvm::zip(barriers, preds)) {
    p << ", " << barrier << '[' << pred << ']';
  }
}

// token := `[` (ssa-value (`,` ssa-value)*)? `]`
// dep-operand := token?
static ParseResult
parseToken(OpAsmParser &p, std::optional<OpAsmParser::UnresolvedOperand> &dep,
           Type &token) {
  if (failed(p.parseOptionalLSquare()))
    return success();
  token = p.getBuilder().getType<AsyncTokenType>();
  if (succeeded(p.parseOptionalRSquare()))
    return success();
  if (p.parseOperand(dep.emplace()) || p.parseRSquare())
    return failure();
  return success();
}
static void printToken(OpAsmPrinter &p, Operation *op, Value dep, Type token) {
  if (!token)
    return;
  p << '[';
  if (dep)
    p << dep;
  p << ']';
}

LogicalResult TCGen5MMAOp::verify() {
  if (!getIsAsync() && !getBarriers().empty()) {
    return emitOpError("The op is synchronous but a barrier is present.");
  }
  return success();
}

void TCGen5MMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The op reads the accumulator if `useD` is not known to be false.
  APInt useD;
  if (!matchPattern(getUseD(), m_ConstantInt(&useD)) || !useD.isZero()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());

  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
}

bool TCGen5MMAOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

Value TCGen5MMAOp::useAccumulator() { return getUseD(); }

void TCGen5MMAOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAOp::getAccumulator() { return getD(); }

void TCGen5MMAOp::setAccumulator(Value accum) { getDMutable().assign(accum); }

Value TCGen5MMAOp::getPredicate() { return getPred(); }

void TCGen5MMAOp::setPredicate(Value pred) { getPredMutable().assign(pred); }

void TCGen5MMAOp::build(OpBuilder &builder, OperationState &state, Type token,
                        Value a, Value b, Value d, Value accDep, Value useD,
                        Value pred, bool useTwoCTAs, ValueRange barriers,
                        ValueRange barrierPreds, bool isAsync) {
  if (!barriers.empty()) {
    isAsync = true;
  }
  build(builder, state, token, a, b, d, accDep, useD, pred, barriers,
        barrierPreds, isAsync ? builder.getUnitAttr() : UnitAttr(),
        useTwoCTAs ? builder.getUnitAttr() : UnitAttr());
}

// -- TCGen5MMAScaledOp --
LogicalResult TCGen5MMAScaledOp::verify() {
  if (!getIsAsync() && !getBarriers().empty()) {
    return emitOpError("The op is synchronous but a barrier is present.");
  }
  return success();
}

void TCGen5MMAScaledOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The op reads the accumulator if `useD` is not known to be false.
  APInt useD;
  if (!matchPattern(getUseD(), m_ConstantInt(&useD)) || !useD.isZero()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());

  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getAScaleMutable(),
                       TensorMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBScaleMutable(),
                       TensorMemory::get());
}

bool TCGen5MMAScaledOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  auto aKdim = aShape[aShape.size() - 1];
  auto bKdim = bShape[aShape.size() - 2];
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    aKdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && !transB)
    bKdim *= 2;

  return aKdim == bKdim;
}

bool TCGen5MMAScaledOp::verifyOutputDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  auto cShape = this->getD().getType().getShape();
  auto oMdim = cShape[cShape.size() - 2];
  auto oNdim = cShape[cShape.size() - 1];

  int aMdim = aShape[aShape.size() - 2];
  int bNdim = bShape[bShape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    aMdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    bNdim *= 2;

  if (aMdim != oMdim || bNdim != oNdim)
    return false;
  return true;
}

Value TCGen5MMAScaledOp::useAccumulator() { return getUseD(); }

void TCGen5MMAScaledOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

void TCGen5MMAScaledOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAScaledOp::getAccumulator() { return getD(); }

void TCGen5MMAScaledOp::setAccumulator(Value accum) {
  getDMutable().assign(accum);
}

Value TCGen5MMAScaledOp::getPredicate() { return getPred(); }

void TCGen5MMAScaledOp::setPredicate(Value pred) {
  getPredMutable().assign(pred);
}

int64_t TCGen5MMAScaledOp::getBlockM() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockM = shape[shape.size() - 2];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    blockM *= 2;
  return blockM;
}

int64_t TCGen5MMAScaledOp::getBlockN() {
  ArrayRef<int64_t> shape = getB().getType().getShape();
  int64_t blockN = shape[shape.size() - 1];
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    blockN *= 2;
  return blockN;
}

int64_t TCGen5MMAScaledOp::getBlockK() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockK = shape[shape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    blockK *= 2;
  return blockK;
}

void TCGen5MMAScaledOp::build(OpBuilder &builder, OperationState &state,
                              Type token, Value a, Value b, Value d,
                              Value accDep, Value aScale, Value bScale,
                              ScaleDotElemType aType, ScaleDotElemType bType,
                              Value useD, Value pred, ValueRange barriers,
                              ValueRange barrierPreds, bool isAsync) {
  MLIRContext *ctx = builder.getContext();
  if (!barriers.empty()) {
    isAsync = true;
  }
  build(builder, state, token, a, b, d, accDep, aScale, bScale,
        ScaleDotElemTypeAttr::get(ctx, aType),
        ScaleDotElemTypeAttr::get(ctx, bType), useD, pred, barriers,
        barrierPreds, isAsync ? builder.getUnitAttr() : UnitAttr());
}

// -- TMEMStoreOp --
static LogicalResult verifyTMEMOperand(Operation *op, RankedTensorType type,
                                       MemDescType memdesc, StringRef regName) {
  if (type.getRank() != 2)
    return op->emitOpError(regName) << " must be a 2D tensor";
  if (type.getEncoding()) {
    auto enc = dyn_cast<DistributedEncodingTrait>(type.getEncoding());
    if (!enc) {
      return op->emitOpError(regName)
             << " does not have an distributed encoding";
    }
    SmallVector<DistributedEncodingTrait> layouts =
        getTmemCompatibleLayouts(op, type, memdesc);
    if (layouts.empty()) {
      return op->emitOpError(regName)
             << " does not have any TMEM compatible layouts";
    }
    if (llvm::none_of(layouts, [&](DistributedEncodingTrait layout) {
          return areLayoutsEquivalent(type.getShape(), layout, enc);
        })) {
      InFlightDiagnostic diag = op->emitOpError(regName)
                                << " layout is not TMEM compatible";
      for (Attribute layout : layouts)
        diag.attachNote() << "potential TMEM layout: " << layout;
      return diag;
    }
  }
  return success();
}

LogicalResult TMEMStoreOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getDst().getType().getMemorySpace()))
    return emitOpError("destination must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr,
           TensorMemoryScalesEncodingAttr>(getDst().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot store into an immutable alloc");
  }
  if (failed(verifyTMEMOperand(*this, getSrc().getType(), getDst().getType(),
                               "source")))
    return failure();
  return triton::gpu::verifyMemoryOpTypes(*this, getSrc().getType(),
                                          getDst().getType());
}

// -- TMEMLoadOp --
LogicalResult TMEMLoadOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("source must be a tensor memory buffer.");
  if (!isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
          getSrc().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (failed(verifyTMEMOperand(*this, getType(), getSrc().getType(), "result")))
    return failure();
  return triton::gpu::verifyMemoryOpTypes(*this, getSrc().getType(), getType());
}

// -- TMEMAllocOp --
LogicalResult TMEMAllocOp::verify() {
  if (!isa<TensorMemorySpaceAttr>(getType().getMemorySpace()))
    return emitOpError("should create a buffer of tensor memory");
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr>(
          getType().getEncoding()))
    return emitOpError("should use tensor memory encoding");
  if (getSrc() &&
      failed(verifyTMEMOperand(*this, getSrc().getType(), getType(), "source")))
    return failure();
  return triton::gpu::verifyAllocOp(*this, getSrc(), getType());
}

void TMEMAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("tensor_memory_col_offset"))
    return;
  OpResult alloc = getOperation()->getOpResult(0);
  effects.emplace_back(MemoryEffects::Allocate::get(), alloc,
                       TensorMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(), alloc,
                         TensorMemory::get());
}

// -- TMEMCopyOp --
LogicalResult TMEMCopyOp::verify() {
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("The source must be a shared memory buffer");
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr>(
          getDst().getType().getEncoding()))
    return emitOpError("The destination must be a tensor memory buffer.");

  if (getBarrier() && !isa<triton::gpu::SharedMemorySpaceAttr>(
                          getBarrier().getType().getMemorySpace())) {
    return emitOpError("The optional barrier should be a shared memory buffer");
  }
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot copy into an immutable alloc");
  }

  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto sharedEnc =
      cast<triton::gpu::NVMMASharedEncodingAttr>(srcTy.getEncoding());

  if (!sharedEnc || sharedEnc.getTransposed() || sharedEnc.getFp4Padded() ||
      sharedEnc.getSwizzlingByteWidth() != 0)
    return emitOpError("The source should not have swizzling applied for now");

  if (!triton::gpu::isInnermostContiguous(srcTy, 512)) {
    return emitOpError("The source must be in a row-major order.");
  }

  // Given that we want to support flexible input SMEM shapes, kinds of shape
  // checking we can do here are limited. For simplicity, shape checking is
  // omitted.
  return success();
}

// -- TMEMSubSliceOp --
LogicalResult TMEMSubSliceOp::verify() {
  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto encoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      srcTy.getEncoding());
  if (!encoding)
    return emitOpError("The source must be a tensor memory buffer.");
  if (!llvm::is_contained({64, 128}, encoding.getBlockM())) {
    return emitOpError("The source tensor memory descriptor must have a 128xN "
                       "or 64xN layout, got block_m=")
           << encoding.getBlockM();
  }
  auto dstTy = cast<triton::gpu::MemDescType>(getResult().getType());
  auto dstEncoding = dyn_cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(
      dstTy.getEncoding());
  if (!dstEncoding)
    return emitOpError("The destination must be a tensor memory buffer.");
  if (dstEncoding.getBlockM() != encoding.getBlockM() ||
      dstEncoding.getCTASplitM() != encoding.getCTASplitM() ||
      dstEncoding.getCTASplitN() != encoding.getCTASplitN() ||
      dstEncoding.getUnpacked() != encoding.getUnpacked())
    return emitOpError("The destination must have the same block size and "
                       "CTASplit size as the source.");
  return mlir::success();
}

void TMEMSubSliceOp::build(OpBuilder &builder, OperationState &state,
                           Value alloc, int offset, int size) {
  auto allocTy = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocTy.getShape());
  shape.back() = size;
  auto encoding =
      cast<triton::nvidia_gpu::TensorMemoryEncodingAttr>(allocTy.getEncoding());
  unsigned newBlockN = std::min<unsigned>(encoding.getBlockN(), size);
  auto newEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
      builder.getContext(), encoding.getBlockM(), newBlockN,
      encoding.getUnpacked(), encoding.getCTASplitM(), encoding.getCTASplitN());
  auto subsliceType = gpu::MemDescType::get(
      shape, allocTy.getElementType(), newEncoding, allocTy.getMemorySpace(),
      allocTy.getMutableMemory());
  build(builder, state, subsliceType, alloc, offset);
}

// -- TensormapCreateOp --
LogicalResult TensormapCreateOp::verify() {
  auto rank = getBoxDim().size();
  if (getGlobalDim().size() != rank) {
    return emitError("Rank mismatch for global dim. Got ")
           << getGlobalDim().size() << " but expected " << rank;
  }
  if (getGlobalStride().size() + 1 != rank) {
    return emitError("Rank mismatch for global stride. Got ")
           << getGlobalStride().size() << " but expected " << rank - 1;
  }
  if (getElementStride().size() != rank) {
    return emitError("Rank mismatch for element stride. Got ")
           << getElementStride().size() << " but expected " << rank;
  }
  return success();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"
