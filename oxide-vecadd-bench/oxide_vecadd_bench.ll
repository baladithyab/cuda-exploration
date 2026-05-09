; ModuleID = 'builtin.module'
source_filename = "oxide_vecadd_bench"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define ptx_kernel void @vecadd_safe(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5) {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v5, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v7, %entry ]
  %v13 = phi { ptr, i64 } [ %v9, %entry ]
  %v14 = phi { ptr, i64 } [ %v11, %entry ]
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb5
bb1:
  %v16 = extractvalue { i8, ptr } %v41, 1
  %v17 = extractvalue { ptr, i64 } %v12, 1
  %v18 = icmp ult i64 %v32, %v17
  br i1 %v18, label %bb2, label %bb13
bb2:
  %v19 = extractvalue { ptr, i64 } %v12, 0
  %v20 = getelementptr inbounds float, ptr %v19, i64 %v32
  %v21 = load float, ptr %v20
  %v22 = extractvalue { ptr, i64 } %v13, 1
  %v23 = icmp ult i64 %v32, %v22
  br i1 %v23, label %bb3, label %bb14
bb3:
  %v24 = extractvalue { ptr, i64 } %v13, 0
  %v25 = getelementptr inbounds float, ptr %v24, i64 %v32
  %v26 = load float, ptr %v25
  %v27 = fadd float %v21, %v26
  store float %v27, ptr %v16
  br label %bb4
bb4:
  ret void
bb5:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb6
bb6:
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb7
bb7:
  %v30 = mul i32 %v28, %v29
  %v31 = add i32 %v30, %v15
  %v32 = zext i32 %v31 to i64
  %v33 = extractvalue { ptr, i64 } %v14, 1
  %v34 = icmp ult i64 %v32, %v33
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb9, label %bb8
bb8:
  %v36 = extractvalue { ptr, i64 } %v14, 0
  %v37 = getelementptr inbounds float, ptr %v36, i64 %v32
  %v38 = insertvalue { i8, ptr } undef, i8 1, 0
  %v39 = insertvalue { i8, ptr } %v38, ptr %v37, 1
  br label %bb10
bb9:
  %v40 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb10
bb10:
  %v41 = phi { i8, ptr } [ %v39, %bb8 ], [ %v40, %bb9 ]
  %v42 = extractvalue { i8, ptr } %v41, 0
  %v43 = zext i8 %v42 to i64
  %v44 = icmp eq i64 %v43, 1
  br i1 %v44, label %bb1, label %bb11
bb11:
  %v45 = icmp eq i64 %v43, 0
  br i1 %v45, label %bb4, label %bb12
bb12:
  unreachable
bb13:
  unreachable
bb14:
  unreachable
}

define ptx_kernel void @vecadd_unchecked(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb4
bb1:
  br label %bb3
bb2:
  %v18 = extractvalue { ptr, i64 } %v13, 0
  %v19 = getelementptr inbounds float, ptr %v18, i64 %v31
  %v20 = load float, ptr %v19
  %v21 = extractvalue { ptr, i64 } %v14, 0
  %v22 = getelementptr inbounds float, ptr %v21, i64 %v31
  %v23 = load float, ptr %v22
  %v24 = extractvalue { ptr, i64 } %v15, 0
  %v25 = getelementptr inbounds float, ptr %v24, i64 %v31
  %v26 = fadd float %v20, %v23
  store float %v26, ptr %v25
  br label %bb3
bb3:
  ret void
bb4:
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb5
bb5:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb6
bb6:
  %v29 = mul i32 %v27, %v28
  %v30 = add i32 %v29, %v17
  %v31 = zext i32 %v30 to i64
  %v32 = zext i32 %v16 to i64
  %v33 = icmp uge i64 %v31, %v32
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb2, label %bb1
}

