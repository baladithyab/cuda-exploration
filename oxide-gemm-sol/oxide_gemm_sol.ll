; ModuleID = 'builtin.module'
source_filename = "oxide_gemm_sol"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_111 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_110 = addrspace(3) global [2 x i64] zeroinitializer, align 16
@__shared_mem_109 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_108 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_107 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_106 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_105 = addrspace(3) global [4 x i32] zeroinitializer, align 4
@__shared_mem_104 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_103 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_102 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_101 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_100 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_99 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_98 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_97 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_96 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_95 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_94 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_93 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_92 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_91 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_90 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_89 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_88 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_87 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_86 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_85 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_84 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_83 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_82 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_81 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_80 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_79 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_78 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_77 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_76 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_75 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_74 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_73 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_72 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_71 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_70 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_69 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_68 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_67 = addrspace(3) global [2 x i64] zeroinitializer, align 16
@__shared_mem_66 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_65 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_64 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_63 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_62 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_61 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_60 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_59 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_58 = addrspace(3) global [4 x i32] zeroinitializer, align 4
@__shared_mem_57 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_56 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_55 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_54 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_53 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_52 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_51 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_50 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_49 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_48 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_47 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_46 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_45 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_44 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_43 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_42 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_41 = addrspace(3) global [2 x i64] zeroinitializer, align 16
@__shared_mem_40 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_39 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_38 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_37 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_36 = addrspace(3) global [4 x i32] zeroinitializer, align 4
@__shared_mem_35 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_34 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_33 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_32 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_31 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_30 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_29 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_28 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_27 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_26 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_25 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_24 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_23 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_22 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_21 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_20 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_19 = addrspace(3) global [4 x i32] zeroinitializer, align 4
@__shared_mem_18 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_17 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_16 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_15 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_14 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_13 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_12 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_11 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_10 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_9 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_8 = addrspace(3) global [8192 x i32] zeroinitializer, align 128
@__shared_mem_7 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_6 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_5 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_4 = addrspace(3) global [16384 x i8] zeroinitializer, align 128
@__shared_mem_3 = addrspace(3) global [1 x i32] zeroinitializer, align 4
@__shared_mem_2 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_1 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_0 = addrspace(3) global [1 x i64] zeroinitializer, align 8
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3), i32) #0
declare void @llvm.nvvm.barrier0() #0
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7), ptr addrspace(3), ptr, i32, i32, i16, i64, i1, i1, i32) #0
declare void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3)) #0

define ptx_kernel void @gemm_sol_pipelined(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi ptr [ %v0, %entry ]
  %v9 = phi ptr [ %v1, %entry ]
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = phi i32 [ %v5, %entry ]
  %v13 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  %v14 = bitcast i32 %v11 to i32
  %v15 = bitcast i32 %v12 to i32
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb100
bb2:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb3
bb3:
  %v19 = xor i1 %v324, 1
  br i1 %v19, label %bb8, label %bb4
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_0, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_1, i32 1) #0
  br label %bb6
bb6:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_2, i32 1) #0
  br label %bb7
bb7:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb8
bb8:
  call void @llvm.nvvm.barrier0() #0
  br label %bb9
bb9:
  %v27 = icmp eq i32 %v322, 0
  %v28 = icmp eq i32 %v322, 0
  br i1 %v28, label %bb10, label %bb12
bb10:
  %v30 = addrspacecast ptr addrspace(3) @__shared_mem_3 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v30, i32 512) #0
  br label %bb11
bb11:
  br label %bb12
bb12:
  call void @llvm.nvvm.barrier0() #0
  br label %bb13
bb13:
  %v33 = bitcast ptr addrspace(3) @__shared_mem_3 to ptr addrspace(3)
  %v34 = addrspacecast ptr addrspace(3) %v33 to ptr
  %v35 = load i32, ptr %v34
  %v36 = insertvalue { i8 } undef, i8 1, 0
  %v37 = insertvalue { i8 } undef, i8 0, 0
  %v38 = insertvalue { i8 } undef, i8 0, 0
  %v39 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v40 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v39, i32 128, 1
  %v41 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v40, i8 0, 2
  %v42 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v41, { i8 } %v37, 3
  %v43 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v42, { i8 } %v38, 4
  %v44 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v43, i1 0, 5
  %v45 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v44, { i8 } %v36, 6
  %v46 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v45, i1 0, 7
  %v47 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v46, i1 0, 8
  %v48 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v47, i1 0, 9
  %v49 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v48, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v49, ptr %v13
  %v50 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 1
  store i32 128, ptr %v50
  %v51 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 0
  store i32 128, ptr %v51
  %v52 = insertvalue { i8 } undef, i8 0, 0
  %v53 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 3
  store { i8 } %v52, ptr %v53
  %v54 = insertvalue { i8 } undef, i8 0, 0
  %v55 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 4
  store { i8 } %v54, ptr %v55
  %v56 = insertvalue { i8 } undef, i8 1, 0
  %v57 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 6
  store { i8 } %v56, ptr %v57
  %v58 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13
  %v59 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 0
  %v60 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 1
  %v61 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 2
  %v62 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 3
  %v63 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 4
  %v64 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 5
  %v65 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 6
  %v66 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 7
  %v67 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 8
  %v68 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 9
  %v69 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, 10
  %v70 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v59, i32 %v60, i8 %v61, { i8 } %v62, { i8 } %v63, i1 %v64, { i8 } %v65, i1 %v66, i1 %v67, i1 %v68, i1 %v69)
  br label %bb14
bb14:
  %v71 = extractvalue { i32 } %v70, 0
  %v72 = udiv i32 %v15, 64
  %v73 = mul i32 %v325, 128
  %v74 = bitcast i32 %v73 to i32
  %v75 = mul i32 %v18, 128
  %v76 = bitcast i32 %v75 to i32
  %v77 = xor i1 %v324, 1
  br i1 %v77, label %bb19, label %bb15
bb15:
  %v79 = addrspacecast ptr addrspace(3) @__shared_mem_4 to ptr
  %v82 = addrspacecast ptr %v79 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v82, ptr addrspace(3) @__shared_mem_0, ptr %v8, i32 0, i32 %v74, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb16
bb16:
  %v85 = addrspacecast ptr addrspace(3) @__shared_mem_5 to ptr
  %v87 = addrspacecast ptr %v85 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v87, ptr addrspace(3) @__shared_mem_0, ptr %v9, i32 0, i32 %v76, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb17
bb17:
  %v89 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v90 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v89, i32 32768) #0
  br label %bb18
bb18:
  br label %bb19
bb19:
  br label %bb20
bb20:
  %v91 = phi i32 [ 0, %bb19 ], [ %v199, %bb64 ]
  %v92 = icmp ult i32 %v91, %v72
  %v93 = xor i1 %v92, 1
  br i1 %v93, label %bb65, label %bb21
bb21:
  %v94 = and i32 %v91, 1
  %v95 = and i32 1, 31
  %v96 = lshr i32 %v91, %v95
  %v97 = and i32 %v96, 1
  %v98 = and i32 %v91, 1
  %v99 = icmp eq i32 %v94, 0
  %v100 = icmp eq i32 %v94, 0
  br i1 %v100, label %bb22, label %bb26
bb22:
  %v102 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v103 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v102, i32 %v97) #0
  %v104 = trunc i32 %v103 to i1
  br label %bb23
bb23:
  %v105 = xor i1 %v104, 1
  br i1 %v105, label %bb25, label %bb24
bb24:
  br label %bb30
bb25:
  br label %bb22
bb26:
  %v107 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v108 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v107, i32 %v97) #0
  %v109 = trunc i32 %v108 to i1
  br label %bb27
bb27:
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb29, label %bb28
bb28:
  br label %bb30
bb29:
  br label %bb26
bb30:
  call void @llvm.nvvm.barrier0() #0
  br label %bb31
bb31:
  %v112 = xor i1 %v324, 1
  br i1 %v112, label %bb47, label %bb32
bb32:
  %v113 = xor i1 %v99, 1
  br i1 %v113, label %bb34, label %bb33
bb33:
  %v115 = bitcast ptr addrspace(3) @__shared_mem_4 to ptr addrspace(3)
  %v116 = ptrtoint ptr addrspace(3) %v115 to i64
  br label %bb35
bb34:
  %v118 = bitcast ptr addrspace(3) @__shared_mem_6 to ptr addrspace(3)
  %v119 = ptrtoint ptr addrspace(3) %v118 to i64
  br label %bb35
bb35:
  %v120 = phi i64 [ %v116, %bb33 ], [ %v119, %bb34 ]
  %v121 = xor i1 %v99, 1
  br i1 %v121, label %bb37, label %bb36
bb36:
  %v123 = bitcast ptr addrspace(3) @__shared_mem_5 to ptr addrspace(3)
  %v124 = ptrtoint ptr addrspace(3) %v123 to i64
  br label %bb38
bb37:
  %v126 = bitcast ptr addrspace(3) @__shared_mem_7 to ptr addrspace(3)
  %v127 = ptrtoint ptr addrspace(3) %v126 to i64
  br label %bb38
bb38:
  %v128 = phi i64 [ %v124, %bb36 ], [ %v127, %bb37 ]
  br label %bb39
bb39:
  %v129 = phi i32 [ 0, %bb38 ], [ %v156, %bb44 ]
  %v130 = icmp ult i32 %v129, 4
  %v131 = xor i1 %v130, 1
  br i1 %v131, label %bb45, label %bb40
bb40:
  %v132 = mul i32 %v129, 32
  %v133 = zext i32 %v132 to i64
  %v134 = add i64 %v120, %v133
  %v135 = zext i32 4 to i64
  %v136 = and i64 %v135, 63
  %v137 = lshr i64 %v134, %v136
  %v138 = and i64 %v137, 16383
  %v139 = or i64 %v138, 65536
  %v140 = or i64 %v139, 274877906944
  %v141 = or i64 %v140, 70368744177664
  %v142 = or i64 %v141, 4611686018427387904
  %v143 = add i64 %v128, %v133
  %v144 = zext i32 4 to i64
  %v145 = and i64 %v144, 63
  %v146 = lshr i64 %v143, %v145
  %v147 = and i64 %v146, 16383
  %v148 = or i64 %v147, 65536
  %v149 = or i64 %v148, 274877906944
  %v150 = or i64 %v149, 70368744177664
  %v151 = or i64 %v150, 4611686018427387904
  %v152 = icmp ugt i32 %v91, 0
  %v153 = xor i1 %v152, 1
  br i1 %v153, label %bb42, label %bb41
bb41:
  br label %bb43
bb42:
  %v154 = icmp ugt i32 %v129, 0
  br label %bb43
bb43:
  %v155 = phi i1 [ 1, %bb41 ], [ %v154, %bb42 ]
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v35, i64 %v142, i64 %v151, i32 %v71, i1 %v155) #0
  br label %bb44
bb44:
  %v156 = add i32 %v129, 1
  br label %bb39
bb45:
  %v158 = addrspacecast ptr addrspace(3) @__shared_mem_2 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v158) #0
  br label %bb46
bb46:
  br label %bb47
bb47:
  %v159 = xor i1 %v324, 1
  br i1 %v159, label %bb59, label %bb48
bb48:
  %v160 = add i32 %v91, 1
  %v161 = icmp ult i32 %v160, %v72
  %v162 = xor i1 %v161, 1
  br i1 %v162, label %bb58, label %bb49
bb49:
  %v163 = add i32 %v91, 1
  %v164 = mul i32 %v163, 64
  %v165 = bitcast i32 %v164 to i32
  %v166 = xor i1 %v99, 1
  br i1 %v166, label %bb54, label %bb50
bb50:
  %v168 = addrspacecast ptr addrspace(3) @__shared_mem_6 to ptr
  %v171 = addrspacecast ptr %v168 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v171, ptr addrspace(3) @__shared_mem_1, ptr %v8, i32 %v165, i32 %v74, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb51
bb51:
  %v174 = addrspacecast ptr addrspace(3) @__shared_mem_7 to ptr
  %v176 = addrspacecast ptr %v174 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v176, ptr addrspace(3) @__shared_mem_1, ptr %v9, i32 %v165, i32 %v76, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb52
bb52:
  %v178 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v179 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v178, i32 32768) #0
  br label %bb53
bb53:
  br label %bb59
bb54:
  %v181 = addrspacecast ptr addrspace(3) @__shared_mem_4 to ptr
  %v184 = addrspacecast ptr %v181 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v184, ptr addrspace(3) @__shared_mem_0, ptr %v8, i32 %v165, i32 %v74, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb55
bb55:
  %v187 = addrspacecast ptr addrspace(3) @__shared_mem_5 to ptr
  %v189 = addrspacecast ptr %v187 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v189, ptr addrspace(3) @__shared_mem_0, ptr %v9, i32 %v165, i32 %v76, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb56
bb56:
  %v191 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v192 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v191, i32 32768) #0
  br label %bb57
bb57:
  br label %bb59
bb58:
  br label %bb59
bb59:
  br label %bb60
bb60:
  %v194 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v195 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v194, i32 %v98) #0
  %v196 = trunc i32 %v195 to i1
  br label %bb61
bb61:
  %v197 = xor i1 %v196, 1
  br i1 %v197, label %bb63, label %bb62
bb62:
  call void @llvm.nvvm.barrier0() #0
  br label %bb64
bb63:
  br label %bb60
bb64:
  %v199 = add i32 %v91, 1
  br label %bb20
bb65:
  %v200 = mul i32 %v322, 32
  %v201 = zext i32 %v200 to i64
  %v202 = urem i32 %v323, 8
  %v203 = zext i32 %v202 to i64
  %v204 = icmp uge i32 %v323, 8
  %v205 = xor i1 %v204, 1
  br i1 %v205, label %bb67, label %bb66
bb66:
  %v206 = icmp ult i32 %v323, 16
  br label %bb68
bb67:
  br label %bb68
bb68:
  %v207 = phi i1 [ %v206, %bb66 ], [ 0, %bb67 ]
  %v208 = xor i1 %v207, 1
  br i1 %v208, label %bb70, label %bb69
bb69:
  br label %bb71
bb70:
  br label %bb71
bb71:
  %v209 = phi i64 [ 16, %bb69 ], [ 0, %bb70 ]
  br label %bb72
bb72:
  %v210 = phi i32 [ 0, %bb71 ], [ %v287, %bb86 ]
  %v211 = icmp ult i32 %v210, 2
  %v212 = xor i1 %v211, 1
  br i1 %v212, label %bb87, label %bb73
bb73:
  %v213 = mul i32 %v210, 16
  %v214 = add i32 %v200, %v213
  br label %bb74
bb74:
  %v215 = phi i32 [ 0, %bb73 ], [ %v286, %bb85 ]
  %v216 = icmp ult i32 %v215, 8
  %v217 = xor i1 %v216, 1
  br i1 %v217, label %bb86, label %bb75
bb75:
  %v218 = mul i32 %v215, 16
  %v219 = zext i32 %v218 to i64
  %v220 = and i32 16, 31
  %v221 = shl i32 %v214, %v220
  %v222 = add i32 %v35, %v221
  %v223 = trunc i64 %v219 to i32
  %v224 = add i32 %v222, %v223
  %v225 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v224) #0
  %v226 = extractvalue { float, float, float, float } %v225, 0
  %v227 = extractvalue { float, float, float, float } %v225, 1
  %v228 = extractvalue { float, float, float, float } %v225, 2
  %v229 = extractvalue { float, float, float, float } %v225, 3
  %v230 = insertvalue [4 x float] undef, float %v226, 0
  %v231 = insertvalue [4 x float] %v230, float %v227, 1
  %v232 = insertvalue [4 x float] %v231, float %v228, 2
  %v233 = insertvalue [4 x float] %v232, float %v229, 3
  %v234 = insertvalue { [4 x float] } undef, [4 x float] %v233, 0
  br label %bb76
bb76:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb77
bb77:
  %v235 = add i32 %v224, 8
  %v236 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v235) #0
  %v237 = extractvalue { float, float, float, float } %v236, 0
  %v238 = extractvalue { float, float, float, float } %v236, 1
  %v239 = extractvalue { float, float, float, float } %v236, 2
  %v240 = extractvalue { float, float, float, float } %v236, 3
  %v241 = insertvalue [4 x float] undef, float %v237, 0
  %v242 = insertvalue [4 x float] %v241, float %v238, 1
  %v243 = insertvalue [4 x float] %v242, float %v239, 2
  %v244 = insertvalue [4 x float] %v243, float %v240, 3
  %v245 = insertvalue { [4 x float] } undef, [4 x float] %v244, 0
  br label %bb78
bb78:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb79
bb79:
  %v246 = extractvalue { [4 x float] } %v234, 0
  %v247 = extractvalue [4 x float] %v246, 0
  %v248 = extractvalue { [4 x float] } %v234, 0
  %v249 = extractvalue [4 x float] %v248, 1
  %v250 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v247, float %v249)
  br label %bb80
bb80:
  %v251 = extractvalue { [4 x float] } %v245, 0
  %v252 = extractvalue [4 x float] %v251, 0
  %v253 = extractvalue { [4 x float] } %v245, 0
  %v254 = extractvalue [4 x float] %v253, 1
  %v255 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v252, float %v254)
  br label %bb81
bb81:
  %v256 = zext i32 %v210 to i64
  %v257 = mul i64 %v256, 16
  %v258 = add i64 %v201, %v257
  %v259 = add i64 %v258, %v203
  %v261 = addrspacecast ptr addrspace(3) @__shared_mem_8 to ptr
  %v262 = mul i64 %v259, 256
  %v263 = mul i64 %v219, 2
  %v264 = add i64 %v262, %v263
  %v265 = add i64 %v264, %v209
  %v266 = getelementptr inbounds i8, ptr %v261, i64 %v265
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v266, i32 %v250, i32 %v255) #0
  br label %bb82
bb82:
  %v267 = extractvalue { [4 x float] } %v234, 0
  %v268 = extractvalue [4 x float] %v267, 2
  %v269 = extractvalue { [4 x float] } %v234, 0
  %v270 = extractvalue [4 x float] %v269, 3
  %v271 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v268, float %v270)
  br label %bb83
bb83:
  %v272 = extractvalue { [4 x float] } %v245, 0
  %v273 = extractvalue [4 x float] %v272, 2
  %v274 = extractvalue { [4 x float] } %v245, 0
  %v275 = extractvalue [4 x float] %v274, 3
  %v276 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v273, float %v275)
  br label %bb84
bb84:
  %v277 = zext i32 %v210 to i64
  %v278 = mul i64 %v277, 16
  %v279 = add i64 %v201, %v278
  %v280 = add i64 %v279, 8
  %v281 = add i64 %v280, %v203
  %v282 = mul i64 %v281, 256
  %v283 = add i64 %v282, %v263
  %v284 = add i64 %v283, %v209
  %v285 = getelementptr inbounds i8, ptr %v261, i64 %v284
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v285, i32 %v271, i32 %v276) #0
  br label %bb85
bb85:
  %v286 = add i32 %v215, 1
  br label %bb74
bb86:
  %v287 = add i32 %v210, 1
  br label %bb72
bb87:
  call void @llvm.nvvm.barrier0() #0
  br label %bb88
bb88:
  %v289 = udiv i32 %v14, 2
  %v290 = zext i32 %v289 to i64
  %v291 = zext i32 %v73 to i64
  %v292 = mul i32 %v18, 64
  %v293 = zext i32 %v292 to i64
  %v294 = zext i32 %v16 to i64
  br label %bb89
bb89:
  %v295 = phi i64 [ %v294, %bb88 ], [ %v312, %bb91 ]
  %v296 = icmp ult i64 %v295, 8192
  %v297 = xor i1 %v296, 1
  br i1 %v297, label %bb92, label %bb90
bb90:
  %v298 = zext i32 6 to i64
  %v299 = and i64 %v298, 63
  %v300 = lshr i64 %v295, %v299
  %v301 = and i64 %v295, 63
  %v302 = add i64 %v291, %v300
  %v303 = add i64 %v293, %v301
  %v304 = mul i64 %v302, %v290
  %v305 = add i64 %v304, %v303
  %v307 = bitcast ptr addrspace(3) @__shared_mem_8 to ptr addrspace(3)
  %v308 = getelementptr inbounds i32, ptr addrspace(3) %v307, i64 %v295
  br label %bb91
bb91:
  %v309 = load i32, ptr addrspace(3) %v308
  %v310 = extractvalue { ptr, i64 } %v10, 0
  %v311 = getelementptr inbounds i32, ptr %v310, i64 %v305
  store i32 %v309, ptr %v311
  %v312 = add i64 %v295, 128
  br label %bb89
bb92:
  call void @llvm.nvvm.barrier0() #0
  br label %bb93
bb93:
  %v314 = xor i1 %v27, 1
  br i1 %v314, label %bb95, label %bb94
bb94:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v35, i32 512) #0
  br label %bb95
bb95:
  %v315 = xor i1 %v324, 1
  br i1 %v315, label %bb99, label %bb96
bb96:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_0) #0
  br label %bb97
bb97:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_1) #0
  br label %bb98
bb98:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_2) #0
  br label %bb99
bb99:
  ret void
bb100:
  %v322 = udiv i32 %v17, 32
  %v323 = urem i32 %v16, 32
  %v324 = icmp eq i32 %v16, 0
  %v325 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
}

declare i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3)) #0

define ptx_kernel void @gemm_sol_persistent(ptr %v0, ptr %v1, ptr %v2, i64 %v3, ptr %v4, i32 %v5, i32 %v6, i32 %v7, i32 %v8) {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  br label %bb0
bb0:
  %v11 = phi ptr [ %v0, %entry ]
  %v12 = phi ptr [ %v1, %entry ]
  %v13 = phi { ptr, i64 } [ %v10, %entry ]
  %v14 = phi ptr [ %v4, %entry ]
  %v15 = phi i32 [ %v5, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = phi i32 [ %v7, %entry ]
  %v18 = phi i32 [ %v8, %entry ]
  %v19 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  br label %bb1
bb1:
  %v20 = bitcast i32 %v15 to i32
  %v21 = bitcast i32 %v16 to i32
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb2
bb2:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb181
bb3:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_9, i32 1) #0
  br label %bb4
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_10, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_11, i32 1) #0
  br label %bb6
bb6:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_12, i32 1) #0
  br label %bb7
bb7:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_13, i32 1) #0
  br label %bb8
bb8:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_14, i32 1) #0
  br label %bb9
bb9:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_15, i32 128) #0
  br label %bb10
bb10:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_16, i32 128) #0
  br label %bb11
bb11:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_17, i32 1) #0
  br label %bb12
bb12:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb13
bb13:
  call void @llvm.nvvm.barrier0() #0
  br label %bb14
bb14:
  %v43 = xor i1 %v492, 1
  br i1 %v43, label %bb18, label %bb15
bb15:
  %v45 = bitcast ptr addrspace(3) @__shared_mem_11 to ptr addrspace(3)
  %v46 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v45) #0
  br label %bb16
bb16:
  %v48 = bitcast ptr addrspace(3) @__shared_mem_12 to ptr addrspace(3)
  %v49 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v48) #0
  br label %bb17
bb17:
  br label %bb18
bb18:
  call void @llvm.nvvm.barrier0() #0
  br label %bb19
bb19:
  %v51 = icmp eq i32 %v490, 0
  %v52 = icmp eq i32 %v490, 0
  br i1 %v52, label %bb20, label %bb22
bb20:
  %v54 = addrspacecast ptr addrspace(3) @__shared_mem_18 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v54, i32 512) #0
  br label %bb21
bb21:
  br label %bb22
bb22:
  call void @llvm.nvvm.barrier0() #0
  br label %bb23
bb23:
  %v57 = bitcast ptr addrspace(3) @__shared_mem_18 to ptr addrspace(3)
  %v58 = addrspacecast ptr addrspace(3) %v57 to ptr
  %v59 = load i32, ptr %v58
  %v60 = insertvalue { i8 } undef, i8 1, 0
  %v61 = insertvalue { i8 } undef, i8 0, 0
  %v62 = insertvalue { i8 } undef, i8 0, 0
  %v63 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v64 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v63, i32 128, 1
  %v65 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v64, i8 0, 2
  %v66 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v65, { i8 } %v61, 3
  %v67 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v66, { i8 } %v62, 4
  %v68 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v67, i1 0, 5
  %v69 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v68, { i8 } %v60, 6
  %v70 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v69, i1 0, 7
  %v71 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v70, i1 0, 8
  %v72 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, i1 0, 9
  %v73 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v72, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v73, ptr %v19
  %v74 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v19, i32 0, i32 1
  store i32 128, ptr %v74
  %v75 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v19, i32 0, i32 0
  store i32 128, ptr %v75
  %v76 = insertvalue { i8 } undef, i8 0, 0
  %v77 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v19, i32 0, i32 3
  store { i8 } %v76, ptr %v77
  %v78 = insertvalue { i8 } undef, i8 0, 0
  %v79 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v19, i32 0, i32 4
  store { i8 } %v78, ptr %v79
  %v80 = insertvalue { i8 } undef, i8 1, 0
  %v81 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v19, i32 0, i32 6
  store { i8 } %v80, ptr %v81
  %v82 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v19
  %v83 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 0
  %v84 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 1
  %v85 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 2
  %v86 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 3
  %v87 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 4
  %v88 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 5
  %v89 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 6
  %v90 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 7
  %v91 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 8
  %v92 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 9
  %v93 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 10
  %v94 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v83, i32 %v84, i8 %v85, { i8 } %v86, { i8 } %v87, i1 %v88, { i8 } %v89, i1 %v90, i1 %v91, i1 %v92, i1 %v93)
  br label %bb24
bb24:
  %v95 = extractvalue { i32 } %v94, 0
  %v96 = udiv i32 %v21, 64
  %v97 = mul i32 %v17, %v18
  call void asm sideeffect "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}"() #0
  br label %bb25
bb25:
  %v98 = icmp eq i32 %v490, 4
  br i1 %v98, label %bb26, label %bb60
bb26:
  %v99 = icmp eq i32 %v491, 0
  %v100 = bitcast ptr %v14 to ptr
  br label %bb27
bb27:
  %v101 = phi i32 [ 0, %bb26 ], [ %v137, %bb59 ]
  %v102 = xor i1 %v99, 1
  br i1 %v102, label %bb35, label %bb28
bb28:
  %v103 = atomicrmw add ptr %v100, i32 1 syncscope("device") monotonic
  br label %bb29
bb29:
  %v104 = icmp ult i32 %v103, %v97
  %v105 = xor i1 %v104, 1
  br i1 %v105, label %bb32, label %bb30
bb30:
  %v106 = icmp eq i32 %v18, 0
  %v107 = xor i1 %v106, 1
  br i1 %v107, label %bb31, label %bb182
bb31:
  %v109 = addrspacecast ptr addrspace(3) @__shared_mem_19 to ptr
  %v110 = udiv i32 %v103, %v18
  store i32 %v110, ptr %v109
  %v111 = getelementptr inbounds i32, ptr %v109, i64 1
  %v112 = urem i32 %v103, %v18
  store i32 %v112, ptr %v111
  %v113 = getelementptr inbounds i32, ptr %v109, i64 2
  store i32 1, ptr %v113
  br label %bb33
bb32:
  %v115 = addrspacecast ptr addrspace(3) @__shared_mem_19 to ptr
  %v116 = getelementptr inbounds i32, ptr %v115, i64 2
  store i32 0, ptr %v116
  br label %bb33
bb33:
  %v118 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v119 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v118) #0
  br label %bb34
bb34:
  br label %bb35
bb35:
  %v121 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v122 = addrspacecast ptr addrspace(3) %v121 to ptr
  %v123 = getelementptr inbounds i32, ptr %v122, i64 2
  %v124 = load i32, ptr %v123
  %v125 = icmp eq i32 %v124, 0
  br i1 %v125, label %bb36, label %bb37
bb36:
  br label %bb60
bb37:
  %v126 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v127 = addrspacecast ptr addrspace(3) %v126 to ptr
  %v128 = load i32, ptr %v127
  %v129 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v130 = addrspacecast ptr addrspace(3) %v129 to ptr
  %v131 = getelementptr inbounds i32, ptr %v130, i64 1
  %v132 = load i32, ptr %v131
  %v133 = mul i32 %v128, 128
  %v134 = bitcast i32 %v133 to i32
  %v135 = mul i32 %v132, 128
  %v136 = bitcast i32 %v135 to i32
  br label %bb38
bb38:
  %v137 = phi i32 [ %v101, %bb37 ], [ %v188, %bb58 ]
  %v138 = phi i32 [ 0, %bb37 ], [ %v187, %bb58 ]
  %v139 = icmp ult i32 %v138, %v96
  %v140 = xor i1 %v139, 1
  br i1 %v140, label %bb59, label %bb39
bb39:
  %v141 = and i32 %v137, 1
  %v142 = and i32 1, 31
  %v143 = lshr i32 %v137, %v142
  %v144 = and i32 %v143, 1
  %v145 = icmp eq i32 %v141, 0
  %v146 = icmp eq i32 %v141, 0
  br i1 %v146, label %bb40, label %bb44
bb40:
  %v148 = bitcast ptr addrspace(3) @__shared_mem_11 to ptr addrspace(3)
  %v149 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v148, i32 %v144) #0
  %v150 = trunc i32 %v149 to i1
  br label %bb41
bb41:
  %v151 = xor i1 %v150, 1
  br i1 %v151, label %bb43, label %bb42
bb42:
  br label %bb48
bb43:
  br label %bb40
bb44:
  %v153 = bitcast ptr addrspace(3) @__shared_mem_12 to ptr addrspace(3)
  %v154 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v153, i32 %v144) #0
  %v155 = trunc i32 %v154 to i1
  br label %bb45
bb45:
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb47, label %bb46
bb46:
  br label %bb48
bb47:
  br label %bb44
bb48:
  %v157 = xor i1 %v99, 1
  br i1 %v157, label %bb58, label %bb49
bb49:
  %v158 = mul i32 %v138, 64
  %v159 = bitcast i32 %v158 to i32
  %v160 = xor i1 %v145, 1
  br i1 %v160, label %bb54, label %bb50
bb50:
  %v162 = addrspacecast ptr addrspace(3) @__shared_mem_20 to ptr
  %v165 = addrspacecast ptr %v162 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v165, ptr addrspace(3) @__shared_mem_9, ptr %v11, i32 %v159, i32 %v134, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb51
bb51:
  %v168 = addrspacecast ptr addrspace(3) @__shared_mem_21 to ptr
  %v170 = addrspacecast ptr %v168 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v170, ptr addrspace(3) @__shared_mem_9, ptr %v12, i32 %v159, i32 %v136, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb52
bb52:
  %v172 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v173 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v172, i32 32768) #0
  br label %bb53
bb53:
  br label %bb58
bb54:
  %v175 = addrspacecast ptr addrspace(3) @__shared_mem_22 to ptr
  %v178 = addrspacecast ptr %v175 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v178, ptr addrspace(3) @__shared_mem_10, ptr %v11, i32 %v159, i32 %v134, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb55
bb55:
  %v181 = addrspacecast ptr addrspace(3) @__shared_mem_23 to ptr
  %v183 = addrspacecast ptr %v181 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v183, ptr addrspace(3) @__shared_mem_10, ptr %v12, i32 %v159, i32 %v136, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb56
bb56:
  %v185 = bitcast ptr addrspace(3) @__shared_mem_10 to ptr addrspace(3)
  %v186 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v185, i32 32768) #0
  br label %bb57
bb57:
  br label %bb58
bb58:
  %v187 = add i32 %v138, 1
  %v188 = add i32 %v137, 1
  br label %bb38
bb59:
  br label %bb27
bb60:
  %v189 = icmp eq i32 %v490, 5
  br i1 %v189, label %bb61, label %bb119
bb61:
  %v190 = icmp eq i32 %v491, 0
  br label %bb62
bb62:
  %v191 = phi i32 [ 0, %bb61 ], [ %v191, %bb65 ], [ %v303, %bb118 ]
  %v192 = phi i32 [ 0, %bb61 ], [ %v192, %bb65 ], [ %v199, %bb118 ]
  %v193 = phi i32 [ 0, %bb61 ], [ %v193, %bb65 ], [ %v224, %bb118 ]
  %v195 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v196 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v195, i32 %v192) #0
  %v197 = trunc i32 %v196 to i1
  br label %bb63
bb63:
  %v198 = xor i1 %v197, 1
  br i1 %v198, label %bb65, label %bb64
bb64:
  %v199 = xor i32 %v192, 1
  %v201 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v202 = addrspacecast ptr addrspace(3) %v201 to ptr
  %v203 = getelementptr inbounds i32, ptr %v202, i64 2
  %v204 = load i32, ptr %v203
  %v205 = icmp eq i32 %v204, 0
  br i1 %v205, label %bb66, label %bb67
bb65:
  br label %bb62
bb66:
  br label %bb119
bb67:
  %v206 = urem i32 %v191, 2
  %v207 = mul i32 %v206, 128
  %v208 = icmp uge i32 %v191, 2
  %v209 = xor i1 %v208, 1
  br i1 %v209, label %bb78, label %bb68
bb68:
  %v210 = sub i32 %v191, 2
  %v211 = udiv i32 %v210, 2
  %v212 = and i32 %v211, 1
  %v213 = icmp eq i32 %v206, 0
  br i1 %v213, label %bb69, label %bb73
bb69:
  %v215 = bitcast ptr addrspace(3) @__shared_mem_15 to ptr addrspace(3)
  %v216 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v215, i32 %v212) #0
  %v217 = trunc i32 %v216 to i1
  br label %bb70
bb70:
  %v218 = xor i1 %v217, 1
  br i1 %v218, label %bb72, label %bb71
bb71:
  br label %bb77
bb72:
  br label %bb69
bb73:
  %v220 = bitcast ptr addrspace(3) @__shared_mem_16 to ptr addrspace(3)
  %v221 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v220, i32 %v212) #0
  %v222 = trunc i32 %v221 to i1
  br label %bb74
bb74:
  %v223 = xor i1 %v222, 1
  br i1 %v223, label %bb76, label %bb75
bb75:
  br label %bb77
bb76:
  br label %bb73
bb77:
  br label %bb79
bb78:
  br label %bb79
bb79:
  br label %bb80
bb80:
  %v224 = phi i32 [ %v193, %bb79 ], [ %v296, %bb110 ]
  %v225 = phi i32 [ 0, %bb79 ], [ %v295, %bb110 ]
  %v226 = icmp ult i32 %v225, %v96
  %v227 = xor i1 %v226, 1
  br i1 %v227, label %bb111, label %bb81
bb81:
  %v228 = and i32 %v224, 1
  %v229 = and i32 1, 31
  %v230 = lshr i32 %v224, %v229
  %v231 = and i32 %v230, 1
  %v232 = icmp eq i32 %v228, 0
  %v233 = icmp eq i32 %v228, 0
  br i1 %v233, label %bb82, label %bb86
bb82:
  %v235 = bitcast ptr addrspace(3) @__shared_mem_9 to ptr addrspace(3)
  %v236 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v235, i32 %v231) #0
  %v237 = trunc i32 %v236 to i1
  br label %bb83
bb83:
  %v238 = xor i1 %v237, 1
  br i1 %v238, label %bb85, label %bb84
bb84:
  br label %bb90
bb85:
  br label %bb82
bb86:
  %v240 = bitcast ptr addrspace(3) @__shared_mem_10 to ptr addrspace(3)
  %v241 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v240, i32 %v231) #0
  %v242 = trunc i32 %v241 to i1
  br label %bb87
bb87:
  %v243 = xor i1 %v242, 1
  br i1 %v243, label %bb89, label %bb88
bb88:
  br label %bb90
bb89:
  br label %bb86
bb90:
  %v244 = xor i1 %v190, 1
  br i1 %v244, label %bb110, label %bb91
bb91:
  %v245 = xor i1 %v232, 1
  br i1 %v245, label %bb93, label %bb92
bb92:
  %v247 = bitcast ptr addrspace(3) @__shared_mem_20 to ptr addrspace(3)
  %v248 = ptrtoint ptr addrspace(3) %v247 to i64
  br label %bb94
bb93:
  %v250 = bitcast ptr addrspace(3) @__shared_mem_22 to ptr addrspace(3)
  %v251 = ptrtoint ptr addrspace(3) %v250 to i64
  br label %bb94
bb94:
  %v252 = phi i64 [ %v248, %bb92 ], [ %v251, %bb93 ]
  %v253 = xor i1 %v232, 1
  br i1 %v253, label %bb96, label %bb95
bb95:
  %v255 = bitcast ptr addrspace(3) @__shared_mem_21 to ptr addrspace(3)
  %v256 = ptrtoint ptr addrspace(3) %v255 to i64
  br label %bb97
bb96:
  %v258 = bitcast ptr addrspace(3) @__shared_mem_23 to ptr addrspace(3)
  %v259 = ptrtoint ptr addrspace(3) %v258 to i64
  br label %bb97
bb97:
  %v260 = phi i64 [ %v256, %bb95 ], [ %v259, %bb96 ]
  br label %bb98
bb98:
  %v261 = phi i32 [ 0, %bb97 ], [ %v289, %bb103 ]
  %v262 = icmp ult i32 %v261, 4
  %v263 = xor i1 %v262, 1
  br i1 %v263, label %bb104, label %bb99
bb99:
  %v264 = mul i32 %v261, 32
  %v265 = zext i32 %v264 to i64
  %v266 = add i64 %v252, %v265
  %v267 = zext i32 4 to i64
  %v268 = and i64 %v267, 63
  %v269 = lshr i64 %v266, %v268
  %v270 = and i64 %v269, 16383
  %v271 = or i64 %v270, 65536
  %v272 = or i64 %v271, 274877906944
  %v273 = or i64 %v272, 70368744177664
  %v274 = or i64 %v273, 4611686018427387904
  %v275 = add i64 %v260, %v265
  %v276 = zext i32 4 to i64
  %v277 = and i64 %v276, 63
  %v278 = lshr i64 %v275, %v277
  %v279 = and i64 %v278, 16383
  %v280 = or i64 %v279, 65536
  %v281 = or i64 %v280, 274877906944
  %v282 = or i64 %v281, 70368744177664
  %v283 = or i64 %v282, 4611686018427387904
  %v284 = icmp ugt i32 %v225, 0
  %v285 = xor i1 %v284, 1
  br i1 %v285, label %bb101, label %bb100
bb100:
  br label %bb102
bb101:
  %v286 = icmp ugt i32 %v261, 0
  br label %bb102
bb102:
  %v287 = phi i1 [ 1, %bb100 ], [ %v286, %bb101 ]
  %v288 = add i32 %v59, %v207
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v288, i64 %v274, i64 %v283, i32 %v95, i1 %v287) #0
  br label %bb103
bb103:
  %v289 = add i32 %v261, 1
  br label %bb98
bb104:
  %v290 = xor i1 %v232, 1
  br i1 %v290, label %bb106, label %bb105
bb105:
  %v292 = addrspacecast ptr addrspace(3) @__shared_mem_11 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v292) #0
  br label %bb107
bb106:
  %v294 = addrspacecast ptr addrspace(3) @__shared_mem_12 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v294) #0
  br label %bb108
bb107:
  br label %bb109
bb108:
  br label %bb109
bb109:
  br label %bb110
bb110:
  %v295 = add i32 %v225, 1
  %v296 = add i32 %v224, 1
  br label %bb80
bb111:
  %v297 = xor i1 %v190, 1
  br i1 %v297, label %bb118, label %bb112
bb112:
  %v298 = icmp eq i32 %v206, 0
  br i1 %v298, label %bb113, label %bb115
bb113:
  %v300 = addrspacecast ptr addrspace(3) @__shared_mem_13 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v300) #0
  br label %bb114
bb114:
  br label %bb117
bb115:
  %v302 = addrspacecast ptr addrspace(3) @__shared_mem_14 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v302) #0
  br label %bb116
bb116:
  br label %bb117
bb117:
  br label %bb118
bb118:
  %v303 = add i32 %v191, 1
  br label %bb62
bb119:
  %v304 = icmp ult i32 %v490, 4
  %v305 = xor i1 %v304, 1
  br i1 %v305, label %bb167, label %bb120
bb120:
  %v306 = mul i32 %v490, 32
  %v307 = zext i32 %v306 to i64
  %v308 = urem i32 %v491, 8
  %v309 = zext i32 %v308 to i64
  %v310 = icmp uge i32 %v491, 8
  %v311 = xor i1 %v310, 1
  br i1 %v311, label %bb122, label %bb121
bb121:
  %v312 = icmp ult i32 %v491, 16
  br label %bb123
bb122:
  br label %bb123
bb123:
  %v313 = phi i1 [ %v312, %bb121 ], [ 0, %bb122 ]
  %v314 = xor i1 %v313, 1
  br i1 %v314, label %bb125, label %bb124
bb124:
  br label %bb126
bb125:
  br label %bb126
bb126:
  %v315 = phi i64 [ 16, %bb124 ], [ 0, %bb125 ]
  br label %bb127
bb127:
  %v316 = phi i32 [ 0, %bb126 ], [ %v316, %bb130 ], [ %v468, %bb166 ]
  %v317 = phi i32 [ 0, %bb126 ], [ %v317, %bb130 ], [ %v323, %bb166 ]
  %v319 = bitcast ptr addrspace(3) @__shared_mem_17 to ptr addrspace(3)
  %v320 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v319, i32 %v317) #0
  %v321 = trunc i32 %v320 to i1
  br label %bb128
bb128:
  %v322 = xor i1 %v321, 1
  br i1 %v322, label %bb130, label %bb129
bb129:
  %v323 = xor i32 %v317, 1
  %v325 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v326 = addrspacecast ptr addrspace(3) %v325 to ptr
  %v327 = getelementptr inbounds i32, ptr %v326, i64 2
  %v328 = load i32, ptr %v327
  %v329 = icmp eq i32 %v328, 0
  br i1 %v329, label %bb131, label %bb132
bb130:
  br label %bb127
bb131:
  br label %bb167
bb132:
  %v330 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v331 = addrspacecast ptr addrspace(3) %v330 to ptr
  %v332 = load i32, ptr %v331
  %v333 = bitcast ptr addrspace(3) @__shared_mem_19 to ptr addrspace(3)
  %v334 = addrspacecast ptr addrspace(3) %v333 to ptr
  %v335 = getelementptr inbounds i32, ptr %v334, i64 1
  %v336 = load i32, ptr %v335
  %v337 = urem i32 %v316, 2
  %v338 = mul i32 %v337, 128
  %v339 = udiv i32 %v316, 2
  %v340 = and i32 %v339, 1
  %v341 = icmp eq i32 %v337, 0
  %v342 = icmp eq i32 %v337, 0
  br i1 %v342, label %bb133, label %bb137
bb133:
  %v344 = bitcast ptr addrspace(3) @__shared_mem_13 to ptr addrspace(3)
  %v345 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v344, i32 %v340) #0
  %v346 = trunc i32 %v345 to i1
  br label %bb134
bb134:
  %v347 = xor i1 %v346, 1
  br i1 %v347, label %bb136, label %bb135
bb135:
  br label %bb141
bb136:
  br label %bb133
bb137:
  %v349 = bitcast ptr addrspace(3) @__shared_mem_14 to ptr addrspace(3)
  %v350 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v349, i32 %v340) #0
  %v351 = trunc i32 %v350 to i1
  br label %bb138
bb138:
  %v352 = xor i1 %v351, 1
  br i1 %v352, label %bb140, label %bb139
bb139:
  br label %bb141
bb140:
  br label %bb137
bb141:
  br label %bb142
bb142:
  %v353 = phi i32 [ 0, %bb141 ], [ %v431, %bb156 ]
  %v354 = icmp ult i32 %v353, 2
  %v355 = xor i1 %v354, 1
  br i1 %v355, label %bb157, label %bb143
bb143:
  %v356 = mul i32 %v353, 16
  %v357 = add i32 %v306, %v356
  br label %bb144
bb144:
  %v358 = phi i32 [ 0, %bb143 ], [ %v430, %bb155 ]
  %v359 = icmp ult i32 %v358, 8
  %v360 = xor i1 %v359, 1
  br i1 %v360, label %bb156, label %bb145
bb145:
  %v361 = mul i32 %v358, 16
  %v362 = zext i32 %v361 to i64
  %v363 = add i32 %v59, %v338
  %v364 = and i32 16, 31
  %v365 = shl i32 %v357, %v364
  %v366 = add i32 %v363, %v365
  %v367 = trunc i64 %v362 to i32
  %v368 = add i32 %v366, %v367
  %v369 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v368) #0
  %v370 = extractvalue { float, float, float, float } %v369, 0
  %v371 = extractvalue { float, float, float, float } %v369, 1
  %v372 = extractvalue { float, float, float, float } %v369, 2
  %v373 = extractvalue { float, float, float, float } %v369, 3
  %v374 = insertvalue [4 x float] undef, float %v370, 0
  %v375 = insertvalue [4 x float] %v374, float %v371, 1
  %v376 = insertvalue [4 x float] %v375, float %v372, 2
  %v377 = insertvalue [4 x float] %v376, float %v373, 3
  %v378 = insertvalue { [4 x float] } undef, [4 x float] %v377, 0
  br label %bb146
bb146:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb147
bb147:
  %v379 = add i32 %v368, 8
  %v380 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v379) #0
  %v381 = extractvalue { float, float, float, float } %v380, 0
  %v382 = extractvalue { float, float, float, float } %v380, 1
  %v383 = extractvalue { float, float, float, float } %v380, 2
  %v384 = extractvalue { float, float, float, float } %v380, 3
  %v385 = insertvalue [4 x float] undef, float %v381, 0
  %v386 = insertvalue [4 x float] %v385, float %v382, 1
  %v387 = insertvalue [4 x float] %v386, float %v383, 2
  %v388 = insertvalue [4 x float] %v387, float %v384, 3
  %v389 = insertvalue { [4 x float] } undef, [4 x float] %v388, 0
  br label %bb148
bb148:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb149
bb149:
  %v390 = extractvalue { [4 x float] } %v378, 0
  %v391 = extractvalue [4 x float] %v390, 0
  %v392 = extractvalue { [4 x float] } %v378, 0
  %v393 = extractvalue [4 x float] %v392, 1
  %v394 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v391, float %v393)
  br label %bb150
bb150:
  %v395 = extractvalue { [4 x float] } %v389, 0
  %v396 = extractvalue [4 x float] %v395, 0
  %v397 = extractvalue { [4 x float] } %v389, 0
  %v398 = extractvalue [4 x float] %v397, 1
  %v399 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v396, float %v398)
  br label %bb151
bb151:
  %v400 = zext i32 %v353 to i64
  %v401 = mul i64 %v400, 16
  %v402 = add i64 %v307, %v401
  %v403 = add i64 %v402, %v309
  %v405 = addrspacecast ptr addrspace(3) @__shared_mem_24 to ptr
  %v406 = mul i64 %v403, 256
  %v407 = mul i64 %v362, 2
  %v408 = add i64 %v406, %v407
  %v409 = add i64 %v408, %v315
  %v410 = getelementptr inbounds i8, ptr %v405, i64 %v409
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v410, i32 %v394, i32 %v399) #0
  br label %bb152
bb152:
  %v411 = extractvalue { [4 x float] } %v378, 0
  %v412 = extractvalue [4 x float] %v411, 2
  %v413 = extractvalue { [4 x float] } %v378, 0
  %v414 = extractvalue [4 x float] %v413, 3
  %v415 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v412, float %v414)
  br label %bb153
bb153:
  %v416 = extractvalue { [4 x float] } %v389, 0
  %v417 = extractvalue [4 x float] %v416, 2
  %v418 = extractvalue { [4 x float] } %v389, 0
  %v419 = extractvalue [4 x float] %v418, 3
  %v420 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v417, float %v419)
  br label %bb154
bb154:
  %v421 = zext i32 %v353 to i64
  %v422 = mul i64 %v421, 16
  %v423 = add i64 %v307, %v422
  %v424 = add i64 %v423, 8
  %v425 = add i64 %v424, %v309
  %v426 = mul i64 %v425, 256
  %v427 = add i64 %v426, %v407
  %v428 = add i64 %v427, %v315
  %v429 = getelementptr inbounds i8, ptr %v405, i64 %v428
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v429, i32 %v415, i32 %v420) #0
  br label %bb155
bb155:
  %v430 = add i32 %v358, 1
  br label %bb144
bb156:
  %v431 = add i32 %v353, 1
  br label %bb142
bb157:
  %v432 = udiv i32 %v20, 2
  %v433 = zext i32 %v432 to i64
  %v434 = mul i32 %v332, 128
  %v435 = zext i32 %v434 to i64
  %v436 = mul i32 %v336, 64
  %v437 = zext i32 %v436 to i64
  %v438 = zext i32 %v490 to i64
  %v439 = mul i64 %v438, 32
  %v440 = zext i32 %v491 to i64
  br label %bb158
bb158:
  %v441 = phi i64 [ %v440, %bb157 ], [ %v460, %bb160 ]
  %v442 = icmp ult i64 %v441, 2048
  %v443 = xor i1 %v442, 1
  br i1 %v443, label %bb161, label %bb159
bb159:
  %v444 = udiv i64 %v441, 64
  %v445 = urem i64 %v441, 64
  %v446 = add i64 %v439, %v444
  %v447 = mul i64 %v446, 64
  %v448 = add i64 %v447, %v445
  %v449 = add i64 %v435, %v439
  %v450 = add i64 %v449, %v444
  %v451 = add i64 %v437, %v445
  %v452 = mul i64 %v450, %v433
  %v453 = add i64 %v452, %v451
  %v455 = bitcast ptr addrspace(3) @__shared_mem_24 to ptr addrspace(3)
  %v456 = getelementptr inbounds i32, ptr addrspace(3) %v455, i64 %v448
  br label %bb160
bb160:
  %v457 = load i32, ptr addrspace(3) %v456
  %v458 = extractvalue { ptr, i64 } %v13, 0
  %v459 = getelementptr inbounds i32, ptr %v458, i64 %v453
  store i32 %v457, ptr %v459
  %v460 = add i64 %v441, 32
  br label %bb158
bb161:
  %v461 = xor i1 %v341, 1
  br i1 %v461, label %bb163, label %bb162
bb162:
  %v463 = bitcast ptr addrspace(3) @__shared_mem_15 to ptr addrspace(3)
  %v464 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v463) #0
  br label %bb164
bb163:
  %v466 = bitcast ptr addrspace(3) @__shared_mem_16 to ptr addrspace(3)
  %v467 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v466) #0
  br label %bb165
bb164:
  br label %bb166
bb165:
  br label %bb166
bb166:
  %v468 = add i32 %v316, 1
  br label %bb127
bb167:
  call void @llvm.nvvm.barrier0() #0
  br label %bb168
bb168:
  %v470 = xor i1 %v51, 1
  br i1 %v470, label %bb170, label %bb169
bb169:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v59, i32 512) #0
  br label %bb170
bb170:
  %v471 = xor i1 %v492, 1
  br i1 %v471, label %bb180, label %bb171
bb171:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_9) #0
  br label %bb172
bb172:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_10) #0
  br label %bb173
bb173:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_11) #0
  br label %bb174
bb174:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_12) #0
  br label %bb175
bb175:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_13) #0
  br label %bb176
bb176:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_14) #0
  br label %bb177
bb177:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_15) #0
  br label %bb178
bb178:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_16) #0
  br label %bb179
bb179:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_17) #0
  br label %bb180
bb180:
  ret void
bb181:
  %v490 = udiv i32 %v23, 32
  %v491 = urem i32 %v22, 32
  %v492 = icmp eq i32 %v22, 0
  %v493 = icmp eq i32 %v22, 0
  br i1 %v493, label %bb3, label %bb13
bb182:
  unreachable
}

define ptx_kernel void @gemm_sol_clc(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7) {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi ptr [ %v0, %entry ]
  %v11 = phi ptr [ %v1, %entry ]
  %v12 = phi { ptr, i64 } [ %v9, %entry ]
  %v13 = phi i32 [ %v4, %entry ]
  %v14 = phi i32 [ %v5, %entry ]
  %v15 = phi i32 [ %v6, %entry ]
  %v16 = phi i32 [ %v7, %entry ]
  %v17 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  br label %bb1
bb1:
  %v18 = bitcast i32 %v13 to i32
  %v19 = bitcast i32 %v14 to i32
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb2
bb2:
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb222
bb3:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_25, i32 1) #0
  br label %bb4
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_26, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_27, i32 1) #0
  br label %bb6
bb6:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_28, i32 1) #0
  br label %bb7
bb7:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_29, i32 1) #0
  br label %bb8
bb8:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_30, i32 1) #0
  br label %bb9
bb9:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_31, i32 128) #0
  br label %bb10
bb10:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_32, i32 128) #0
  br label %bb11
bb11:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_33, i32 1) #0
  br label %bb12
bb12:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_34, i32 1) #0
  br label %bb13
bb13:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb14
bb14:
  call void @llvm.nvvm.barrier0() #0
  br label %bb15
bb15:
  %v43 = xor i1 %v577, 1
  br i1 %v43, label %bb19, label %bb16
bb16:
  %v45 = bitcast ptr addrspace(3) @__shared_mem_27 to ptr addrspace(3)
  %v46 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v45) #0
  br label %bb17
bb17:
  %v48 = bitcast ptr addrspace(3) @__shared_mem_28 to ptr addrspace(3)
  %v49 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v48) #0
  br label %bb18
bb18:
  br label %bb19
bb19:
  call void @llvm.nvvm.barrier0() #0
  br label %bb20
bb20:
  %v51 = icmp eq i32 %v575, 0
  %v52 = icmp eq i32 %v575, 0
  br i1 %v52, label %bb21, label %bb23
bb21:
  %v54 = addrspacecast ptr addrspace(3) @__shared_mem_35 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v54, i32 512) #0
  br label %bb22
bb22:
  br label %bb23
bb23:
  call void @llvm.nvvm.barrier0() #0
  br label %bb24
bb24:
  %v57 = bitcast ptr addrspace(3) @__shared_mem_35 to ptr addrspace(3)
  %v58 = addrspacecast ptr addrspace(3) %v57 to ptr
  %v59 = load i32, ptr %v58
  %v60 = insertvalue { i8 } undef, i8 1, 0
  %v61 = insertvalue { i8 } undef, i8 0, 0
  %v62 = insertvalue { i8 } undef, i8 0, 0
  %v63 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v64 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v63, i32 128, 1
  %v65 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v64, i8 0, 2
  %v66 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v65, { i8 } %v61, 3
  %v67 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v66, { i8 } %v62, 4
  %v68 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v67, i1 0, 5
  %v69 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v68, { i8 } %v60, 6
  %v70 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v69, i1 0, 7
  %v71 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v70, i1 0, 8
  %v72 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, i1 0, 9
  %v73 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v72, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v73, ptr %v17
  %v74 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 1
  store i32 128, ptr %v74
  %v75 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 0
  store i32 128, ptr %v75
  %v76 = insertvalue { i8 } undef, i8 0, 0
  %v77 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 3
  store { i8 } %v76, ptr %v77
  %v78 = insertvalue { i8 } undef, i8 0, 0
  %v79 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 4
  store { i8 } %v78, ptr %v79
  %v80 = insertvalue { i8 } undef, i8 1, 0
  %v81 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 6
  store { i8 } %v80, ptr %v81
  %v82 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17
  %v83 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 0
  %v84 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 1
  %v85 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 2
  %v86 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 3
  %v87 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 4
  %v88 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 5
  %v89 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 6
  %v90 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 7
  %v91 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 8
  %v92 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 9
  %v93 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, 10
  %v94 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v83, i32 %v84, i8 %v85, { i8 } %v86, { i8 } %v87, i1 %v88, { i8 } %v89, i1 %v90, i1 %v91, i1 %v92, i1 %v93)
  br label %bb25
bb25:
  %v95 = extractvalue { i32 } %v94, 0
  %v96 = udiv i32 %v19, 64
  call void asm sideeffect "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}"() #0
  br label %bb26
bb26:
  %v97 = icmp eq i32 %v575, 4
  br i1 %v97, label %bb27, label %bb100
bb27:
  %v98 = icmp eq i32 %v576, 0
  %v99 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb28
bb28:
  %v100 = icmp eq i32 %v15, 0
  %v101 = xor i1 %v100, 1
  br i1 %v101, label %bb29, label %bb223
bb29:
  %v102 = urem i32 %v99, %v15
  %v103 = udiv i32 %v99, %v15
  %v104 = xor i1 %v98, 1
  br i1 %v104, label %bb32, label %bb30
bb30:
  %v106 = addrspacecast ptr addrspace(3) @__shared_mem_36 to ptr
  store i32 %v102, ptr %v106
  %v107 = getelementptr inbounds i32, ptr %v106, i64 1
  store i32 %v103, ptr %v107
  %v108 = getelementptr inbounds i32, ptr %v106, i64 2
  store i32 1, ptr %v108
  %v110 = bitcast ptr addrspace(3) @__shared_mem_33 to ptr addrspace(3)
  %v111 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v110) #0
  br label %bb31
bb31:
  br label %bb32
bb32:
  %v112 = mul i32 %v102, 128
  %v113 = bitcast i32 %v112 to i32
  %v114 = mul i32 %v103, 128
  %v115 = bitcast i32 %v114 to i32
  br label %bb33
bb33:
  %v116 = phi i32 [ 0, %bb32 ], [ %v167, %bb53 ]
  %v117 = phi i32 [ 0, %bb32 ], [ %v166, %bb53 ]
  %v118 = icmp ult i32 %v117, %v96
  %v119 = xor i1 %v118, 1
  br i1 %v119, label %bb54, label %bb34
bb34:
  %v120 = and i32 %v116, 1
  %v121 = and i32 1, 31
  %v122 = lshr i32 %v116, %v121
  %v123 = and i32 %v122, 1
  %v124 = icmp eq i32 %v120, 0
  %v125 = icmp eq i32 %v120, 0
  br i1 %v125, label %bb35, label %bb39
bb35:
  %v127 = bitcast ptr addrspace(3) @__shared_mem_27 to ptr addrspace(3)
  %v128 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v127, i32 %v123) #0
  %v129 = trunc i32 %v128 to i1
  br label %bb36
bb36:
  %v130 = xor i1 %v129, 1
  br i1 %v130, label %bb38, label %bb37
bb37:
  br label %bb43
bb38:
  br label %bb35
bb39:
  %v132 = bitcast ptr addrspace(3) @__shared_mem_28 to ptr addrspace(3)
  %v133 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v132, i32 %v123) #0
  %v134 = trunc i32 %v133 to i1
  br label %bb40
bb40:
  %v135 = xor i1 %v134, 1
  br i1 %v135, label %bb42, label %bb41
bb41:
  br label %bb43
bb42:
  br label %bb39
bb43:
  %v136 = xor i1 %v98, 1
  br i1 %v136, label %bb53, label %bb44
bb44:
  %v137 = mul i32 %v117, 64
  %v138 = bitcast i32 %v137 to i32
  %v139 = xor i1 %v124, 1
  br i1 %v139, label %bb49, label %bb45
bb45:
  %v141 = addrspacecast ptr addrspace(3) @__shared_mem_37 to ptr
  %v144 = addrspacecast ptr %v141 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v144, ptr addrspace(3) @__shared_mem_25, ptr %v10, i32 %v138, i32 %v113, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb46
bb46:
  %v147 = addrspacecast ptr addrspace(3) @__shared_mem_38 to ptr
  %v149 = addrspacecast ptr %v147 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v149, ptr addrspace(3) @__shared_mem_25, ptr %v11, i32 %v138, i32 %v115, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb47
bb47:
  %v151 = bitcast ptr addrspace(3) @__shared_mem_25 to ptr addrspace(3)
  %v152 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v151, i32 32768) #0
  br label %bb48
bb48:
  br label %bb53
bb49:
  %v154 = addrspacecast ptr addrspace(3) @__shared_mem_39 to ptr
  %v157 = addrspacecast ptr %v154 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v157, ptr addrspace(3) @__shared_mem_26, ptr %v10, i32 %v138, i32 %v113, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb50
bb50:
  %v160 = addrspacecast ptr addrspace(3) @__shared_mem_40 to ptr
  %v162 = addrspacecast ptr %v160 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v162, ptr addrspace(3) @__shared_mem_26, ptr %v11, i32 %v138, i32 %v115, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb51
bb51:
  %v164 = bitcast ptr addrspace(3) @__shared_mem_26 to ptr addrspace(3)
  %v165 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v164, i32 32768) #0
  br label %bb52
bb52:
  br label %bb53
bb53:
  %v166 = add i32 %v117, 1
  %v167 = add i32 %v116, 1
  br label %bb33
bb54:
  %v169 = addrspacecast ptr addrspace(3) @__shared_mem_41 to ptr
  br label %bb55
bb55:
  %v170 = phi i32 [ %v116, %bb54 ], [ %v199, %bb99 ]
  %v171 = phi i32 [ 0, %bb54 ], [ %v271, %bb99 ]
  %v172 = and i32 %v171, 1
  %v173 = xor i1 %v98, 1
  br i1 %v173, label %bb59, label %bb56
bb56:
  %v175 = bitcast ptr addrspace(3) @__shared_mem_34 to ptr addrspace(3)
  %v176 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v175, i32 16) #0
  br label %bb57
bb57:
  %v178 = addrspacecast ptr addrspace(3) @__shared_mem_41 to ptr
  call void asm sideeffect "{ .reg .u64 %resp_shared64; .reg .u32 %resp_shared32; cvta.to.shared.u64 %resp_shared64, $0; cvt.u32.u64 %resp_shared32, %resp_shared64; .reg .u64 %mbar_shared64; .reg .u32 %mbar_shared32; cvta.to.shared.u64 %mbar_shared64, $1; cvt.u32.u64 %mbar_shared32, %mbar_shared64; clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%resp_shared32], [%mbar_shared32]; }", "l,l,~{memory}"(ptr %v178, ptr addrspace(3) @__shared_mem_34) #0
  br label %bb58
bb58:
  br label %bb59
bb59:
  %v180 = xor i1 %v98, 1
  br i1 %v180, label %bb64, label %bb60
bb60:
  %v182 = bitcast ptr addrspace(3) @__shared_mem_34 to ptr addrspace(3)
  %v183 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v182, i32 %v172) #0
  %v184 = trunc i32 %v183 to i1
  br label %bb61
bb61:
  %v185 = xor i1 %v184, 1
  br i1 %v185, label %bb63, label %bb62
bb62:
  br label %bb64
bb63:
  br label %bb60
bb64:
  %v186 = load i64, ptr %v169
  %v187 = getelementptr inbounds i64, ptr %v169, i64 1
  %v188 = load i64, ptr %v187
  %v189 = call i32 asm sideeffect "{ .reg .b128 %resp; mov.b128 %resp, {$1, $2}; .reg .pred %p; clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %resp; selp.b32 $0, 1, 0, %p; }", "=r,l,l"(i64 %v186, i64 %v188) #0
  br label %bb65
bb65:
  %v190 = icmp eq i32 %v189, 0
  br i1 %v190, label %bb66, label %bb70
bb66:
  %v191 = xor i1 %v98, 1
  br i1 %v191, label %bb69, label %bb67
bb67:
  %v193 = addrspacecast ptr addrspace(3) @__shared_mem_36 to ptr
  %v194 = getelementptr inbounds i32, ptr %v193, i64 2
  store i32 0, ptr %v194
  %v196 = bitcast ptr addrspace(3) @__shared_mem_33 to ptr addrspace(3)
  %v197 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v196) #0
  br label %bb68
bb68:
  br label %bb69
bb69:
  br label %bb100
bb70:
  %v198 = call i32 asm sideeffect "{ .reg .b128 %resp; mov.b128 %resp, {$1, $2}; clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 $0, %resp; }", "=r,l,l"(i64 %v186, i64 %v188) #0
  br label %bb71
bb71:
  br label %bb72
bb72:
  %v199 = phi i32 [ %v170, %bb71 ], [ %v218, %bb98 ]
  %v200 = phi i32 [ 0, %bb71 ], [ %v270, %bb98 ]
  %v201 = icmp ult i32 %v200, 4
  %v202 = xor i1 %v201, 1
  br i1 %v202, label %bb99, label %bb73
bb73:
  %v203 = add i32 %v198, %v200
  %v204 = urem i32 %v203, %v15
  %v205 = udiv i32 %v203, %v15
  %v206 = xor i1 %v98, 1
  br i1 %v206, label %bb76, label %bb74
bb74:
  %v208 = addrspacecast ptr addrspace(3) @__shared_mem_36 to ptr
  store i32 %v204, ptr %v208
  %v209 = getelementptr inbounds i32, ptr %v208, i64 1
  store i32 %v205, ptr %v209
  %v210 = getelementptr inbounds i32, ptr %v208, i64 2
  store i32 1, ptr %v210
  %v212 = bitcast ptr addrspace(3) @__shared_mem_33 to ptr addrspace(3)
  %v213 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v212) #0
  br label %bb75
bb75:
  br label %bb76
bb76:
  %v214 = mul i32 %v204, 128
  %v215 = bitcast i32 %v214 to i32
  %v216 = mul i32 %v205, 128
  %v217 = bitcast i32 %v216 to i32
  br label %bb77
bb77:
  %v218 = phi i32 [ %v199, %bb76 ], [ %v269, %bb97 ]
  %v219 = phi i32 [ 0, %bb76 ], [ %v268, %bb97 ]
  %v220 = icmp ult i32 %v219, %v96
  %v221 = xor i1 %v220, 1
  br i1 %v221, label %bb98, label %bb78
bb78:
  %v222 = and i32 %v218, 1
  %v223 = and i32 1, 31
  %v224 = lshr i32 %v218, %v223
  %v225 = and i32 %v224, 1
  %v226 = icmp eq i32 %v222, 0
  %v227 = icmp eq i32 %v222, 0
  br i1 %v227, label %bb79, label %bb83
bb79:
  %v229 = bitcast ptr addrspace(3) @__shared_mem_27 to ptr addrspace(3)
  %v230 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v229, i32 %v225) #0
  %v231 = trunc i32 %v230 to i1
  br label %bb80
bb80:
  %v232 = xor i1 %v231, 1
  br i1 %v232, label %bb82, label %bb81
bb81:
  br label %bb87
bb82:
  br label %bb79
bb83:
  %v234 = bitcast ptr addrspace(3) @__shared_mem_28 to ptr addrspace(3)
  %v235 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v234, i32 %v225) #0
  %v236 = trunc i32 %v235 to i1
  br label %bb84
bb84:
  %v237 = xor i1 %v236, 1
  br i1 %v237, label %bb86, label %bb85
bb85:
  br label %bb87
bb86:
  br label %bb83
bb87:
  %v238 = xor i1 %v98, 1
  br i1 %v238, label %bb97, label %bb88
bb88:
  %v239 = mul i32 %v219, 64
  %v240 = bitcast i32 %v239 to i32
  %v241 = xor i1 %v226, 1
  br i1 %v241, label %bb93, label %bb89
bb89:
  %v243 = addrspacecast ptr addrspace(3) @__shared_mem_37 to ptr
  %v246 = addrspacecast ptr %v243 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v246, ptr addrspace(3) @__shared_mem_25, ptr %v10, i32 %v240, i32 %v215, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb90
bb90:
  %v249 = addrspacecast ptr addrspace(3) @__shared_mem_38 to ptr
  %v251 = addrspacecast ptr %v249 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v251, ptr addrspace(3) @__shared_mem_25, ptr %v11, i32 %v240, i32 %v217, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb91
bb91:
  %v253 = bitcast ptr addrspace(3) @__shared_mem_25 to ptr addrspace(3)
  %v254 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v253, i32 32768) #0
  br label %bb92
bb92:
  br label %bb97
bb93:
  %v256 = addrspacecast ptr addrspace(3) @__shared_mem_39 to ptr
  %v259 = addrspacecast ptr %v256 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v259, ptr addrspace(3) @__shared_mem_26, ptr %v10, i32 %v240, i32 %v215, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb94
bb94:
  %v262 = addrspacecast ptr addrspace(3) @__shared_mem_40 to ptr
  %v264 = addrspacecast ptr %v262 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v264, ptr addrspace(3) @__shared_mem_26, ptr %v11, i32 %v240, i32 %v217, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb95
bb95:
  %v266 = bitcast ptr addrspace(3) @__shared_mem_26 to ptr addrspace(3)
  %v267 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v266, i32 32768) #0
  br label %bb96
bb96:
  br label %bb97
bb97:
  %v268 = add i32 %v219, 1
  %v269 = add i32 %v218, 1
  br label %bb77
bb98:
  %v270 = add i32 %v200, 1
  br label %bb72
bb99:
  %v271 = add i32 %v171, 1
  br label %bb55
bb100:
  %v272 = icmp eq i32 %v575, 5
  br i1 %v272, label %bb101, label %bb159
bb101:
  %v273 = icmp eq i32 %v576, 0
  br label %bb102
bb102:
  %v274 = phi i32 [ 0, %bb101 ], [ %v274, %bb105 ], [ %v386, %bb158 ]
  %v275 = phi i32 [ 0, %bb101 ], [ %v275, %bb105 ], [ %v282, %bb158 ]
  %v276 = phi i32 [ 0, %bb101 ], [ %v276, %bb105 ], [ %v307, %bb158 ]
  %v278 = bitcast ptr addrspace(3) @__shared_mem_33 to ptr addrspace(3)
  %v279 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v278, i32 %v275) #0
  %v280 = trunc i32 %v279 to i1
  br label %bb103
bb103:
  %v281 = xor i1 %v280, 1
  br i1 %v281, label %bb105, label %bb104
bb104:
  %v282 = xor i32 %v275, 1
  %v284 = bitcast ptr addrspace(3) @__shared_mem_36 to ptr addrspace(3)
  %v285 = addrspacecast ptr addrspace(3) %v284 to ptr
  %v286 = getelementptr inbounds i32, ptr %v285, i64 2
  %v287 = load i32, ptr %v286
  %v288 = icmp eq i32 %v287, 0
  br i1 %v288, label %bb106, label %bb107
bb105:
  br label %bb102
bb106:
  br label %bb159
bb107:
  %v289 = urem i32 %v274, 2
  %v290 = mul i32 %v289, 128
  %v291 = icmp uge i32 %v274, 2
  %v292 = xor i1 %v291, 1
  br i1 %v292, label %bb118, label %bb108
bb108:
  %v293 = sub i32 %v274, 2
  %v294 = udiv i32 %v293, 2
  %v295 = and i32 %v294, 1
  %v296 = icmp eq i32 %v289, 0
  br i1 %v296, label %bb109, label %bb113
bb109:
  %v298 = bitcast ptr addrspace(3) @__shared_mem_31 to ptr addrspace(3)
  %v299 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v298, i32 %v295) #0
  %v300 = trunc i32 %v299 to i1
  br label %bb110
bb110:
  %v301 = xor i1 %v300, 1
  br i1 %v301, label %bb112, label %bb111
bb111:
  br label %bb117
bb112:
  br label %bb109
bb113:
  %v303 = bitcast ptr addrspace(3) @__shared_mem_32 to ptr addrspace(3)
  %v304 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v303, i32 %v295) #0
  %v305 = trunc i32 %v304 to i1
  br label %bb114
bb114:
  %v306 = xor i1 %v305, 1
  br i1 %v306, label %bb116, label %bb115
bb115:
  br label %bb117
bb116:
  br label %bb113
bb117:
  br label %bb119
bb118:
  br label %bb119
bb119:
  br label %bb120
bb120:
  %v307 = phi i32 [ %v276, %bb119 ], [ %v379, %bb150 ]
  %v308 = phi i32 [ 0, %bb119 ], [ %v378, %bb150 ]
  %v309 = icmp ult i32 %v308, %v96
  %v310 = xor i1 %v309, 1
  br i1 %v310, label %bb151, label %bb121
bb121:
  %v311 = and i32 %v307, 1
  %v312 = and i32 1, 31
  %v313 = lshr i32 %v307, %v312
  %v314 = and i32 %v313, 1
  %v315 = icmp eq i32 %v311, 0
  %v316 = icmp eq i32 %v311, 0
  br i1 %v316, label %bb122, label %bb126
bb122:
  %v318 = bitcast ptr addrspace(3) @__shared_mem_25 to ptr addrspace(3)
  %v319 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v318, i32 %v314) #0
  %v320 = trunc i32 %v319 to i1
  br label %bb123
bb123:
  %v321 = xor i1 %v320, 1
  br i1 %v321, label %bb125, label %bb124
bb124:
  br label %bb130
bb125:
  br label %bb122
bb126:
  %v323 = bitcast ptr addrspace(3) @__shared_mem_26 to ptr addrspace(3)
  %v324 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v323, i32 %v314) #0
  %v325 = trunc i32 %v324 to i1
  br label %bb127
bb127:
  %v326 = xor i1 %v325, 1
  br i1 %v326, label %bb129, label %bb128
bb128:
  br label %bb130
bb129:
  br label %bb126
bb130:
  %v327 = xor i1 %v273, 1
  br i1 %v327, label %bb150, label %bb131
bb131:
  %v328 = xor i1 %v315, 1
  br i1 %v328, label %bb133, label %bb132
bb132:
  %v330 = bitcast ptr addrspace(3) @__shared_mem_37 to ptr addrspace(3)
  %v331 = ptrtoint ptr addrspace(3) %v330 to i64
  br label %bb134
bb133:
  %v333 = bitcast ptr addrspace(3) @__shared_mem_39 to ptr addrspace(3)
  %v334 = ptrtoint ptr addrspace(3) %v333 to i64
  br label %bb134
bb134:
  %v335 = phi i64 [ %v331, %bb132 ], [ %v334, %bb133 ]
  %v336 = xor i1 %v315, 1
  br i1 %v336, label %bb136, label %bb135
bb135:
  %v338 = bitcast ptr addrspace(3) @__shared_mem_38 to ptr addrspace(3)
  %v339 = ptrtoint ptr addrspace(3) %v338 to i64
  br label %bb137
bb136:
  %v341 = bitcast ptr addrspace(3) @__shared_mem_40 to ptr addrspace(3)
  %v342 = ptrtoint ptr addrspace(3) %v341 to i64
  br label %bb137
bb137:
  %v343 = phi i64 [ %v339, %bb135 ], [ %v342, %bb136 ]
  br label %bb138
bb138:
  %v344 = phi i32 [ 0, %bb137 ], [ %v372, %bb143 ]
  %v345 = icmp ult i32 %v344, 4
  %v346 = xor i1 %v345, 1
  br i1 %v346, label %bb144, label %bb139
bb139:
  %v347 = mul i32 %v344, 32
  %v348 = zext i32 %v347 to i64
  %v349 = add i64 %v335, %v348
  %v350 = zext i32 4 to i64
  %v351 = and i64 %v350, 63
  %v352 = lshr i64 %v349, %v351
  %v353 = and i64 %v352, 16383
  %v354 = or i64 %v353, 65536
  %v355 = or i64 %v354, 274877906944
  %v356 = or i64 %v355, 70368744177664
  %v357 = or i64 %v356, 4611686018427387904
  %v358 = add i64 %v343, %v348
  %v359 = zext i32 4 to i64
  %v360 = and i64 %v359, 63
  %v361 = lshr i64 %v358, %v360
  %v362 = and i64 %v361, 16383
  %v363 = or i64 %v362, 65536
  %v364 = or i64 %v363, 274877906944
  %v365 = or i64 %v364, 70368744177664
  %v366 = or i64 %v365, 4611686018427387904
  %v367 = icmp ugt i32 %v308, 0
  %v368 = xor i1 %v367, 1
  br i1 %v368, label %bb141, label %bb140
bb140:
  br label %bb142
bb141:
  %v369 = icmp ugt i32 %v344, 0
  br label %bb142
bb142:
  %v370 = phi i1 [ 1, %bb140 ], [ %v369, %bb141 ]
  %v371 = add i32 %v59, %v290
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v371, i64 %v357, i64 %v366, i32 %v95, i1 %v370) #0
  br label %bb143
bb143:
  %v372 = add i32 %v344, 1
  br label %bb138
bb144:
  %v373 = xor i1 %v315, 1
  br i1 %v373, label %bb146, label %bb145
bb145:
  %v375 = addrspacecast ptr addrspace(3) @__shared_mem_27 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v375) #0
  br label %bb147
bb146:
  %v377 = addrspacecast ptr addrspace(3) @__shared_mem_28 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v377) #0
  br label %bb148
bb147:
  br label %bb149
bb148:
  br label %bb149
bb149:
  br label %bb150
bb150:
  %v378 = add i32 %v308, 1
  %v379 = add i32 %v307, 1
  br label %bb120
bb151:
  %v380 = xor i1 %v273, 1
  br i1 %v380, label %bb158, label %bb152
bb152:
  %v381 = icmp eq i32 %v289, 0
  br i1 %v381, label %bb153, label %bb155
bb153:
  %v383 = addrspacecast ptr addrspace(3) @__shared_mem_29 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v383) #0
  br label %bb154
bb154:
  br label %bb157
bb155:
  %v385 = addrspacecast ptr addrspace(3) @__shared_mem_30 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v385) #0
  br label %bb156
bb156:
  br label %bb157
bb157:
  br label %bb158
bb158:
  %v386 = add i32 %v274, 1
  br label %bb102
bb159:
  %v387 = icmp ult i32 %v575, 4
  %v388 = xor i1 %v387, 1
  br i1 %v388, label %bb207, label %bb160
bb160:
  %v389 = mul i32 %v575, 32
  %v390 = zext i32 %v389 to i64
  %v391 = urem i32 %v576, 8
  %v392 = zext i32 %v391 to i64
  %v393 = icmp uge i32 %v576, 8
  %v394 = xor i1 %v393, 1
  br i1 %v394, label %bb162, label %bb161
bb161:
  %v395 = icmp ult i32 %v576, 16
  br label %bb163
bb162:
  br label %bb163
bb163:
  %v396 = phi i1 [ %v395, %bb161 ], [ 0, %bb162 ]
  %v397 = xor i1 %v396, 1
  br i1 %v397, label %bb165, label %bb164
bb164:
  br label %bb166
bb165:
  br label %bb166
bb166:
  %v398 = phi i64 [ 16, %bb164 ], [ 0, %bb165 ]
  br label %bb167
bb167:
  %v399 = phi i32 [ 0, %bb166 ], [ %v399, %bb170 ], [ %v551, %bb206 ]
  %v400 = phi i32 [ 0, %bb166 ], [ %v400, %bb170 ], [ %v406, %bb206 ]
  %v402 = bitcast ptr addrspace(3) @__shared_mem_33 to ptr addrspace(3)
  %v403 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v402, i32 %v400) #0
  %v404 = trunc i32 %v403 to i1
  br label %bb168
bb168:
  %v405 = xor i1 %v404, 1
  br i1 %v405, label %bb170, label %bb169
bb169:
  %v406 = xor i32 %v400, 1
  %v408 = bitcast ptr addrspace(3) @__shared_mem_36 to ptr addrspace(3)
  %v409 = addrspacecast ptr addrspace(3) %v408 to ptr
  %v410 = getelementptr inbounds i32, ptr %v409, i64 2
  %v411 = load i32, ptr %v410
  %v412 = icmp eq i32 %v411, 0
  br i1 %v412, label %bb171, label %bb172
bb170:
  br label %bb167
bb171:
  br label %bb207
bb172:
  %v413 = bitcast ptr addrspace(3) @__shared_mem_36 to ptr addrspace(3)
  %v414 = addrspacecast ptr addrspace(3) %v413 to ptr
  %v415 = load i32, ptr %v414
  %v416 = bitcast ptr addrspace(3) @__shared_mem_36 to ptr addrspace(3)
  %v417 = addrspacecast ptr addrspace(3) %v416 to ptr
  %v418 = getelementptr inbounds i32, ptr %v417, i64 1
  %v419 = load i32, ptr %v418
  %v420 = urem i32 %v399, 2
  %v421 = mul i32 %v420, 128
  %v422 = udiv i32 %v399, 2
  %v423 = and i32 %v422, 1
  %v424 = icmp eq i32 %v420, 0
  %v425 = icmp eq i32 %v420, 0
  br i1 %v425, label %bb173, label %bb177
bb173:
  %v427 = bitcast ptr addrspace(3) @__shared_mem_29 to ptr addrspace(3)
  %v428 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v427, i32 %v423) #0
  %v429 = trunc i32 %v428 to i1
  br label %bb174
bb174:
  %v430 = xor i1 %v429, 1
  br i1 %v430, label %bb176, label %bb175
bb175:
  br label %bb181
bb176:
  br label %bb173
bb177:
  %v432 = bitcast ptr addrspace(3) @__shared_mem_30 to ptr addrspace(3)
  %v433 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v432, i32 %v423) #0
  %v434 = trunc i32 %v433 to i1
  br label %bb178
bb178:
  %v435 = xor i1 %v434, 1
  br i1 %v435, label %bb180, label %bb179
bb179:
  br label %bb181
bb180:
  br label %bb177
bb181:
  br label %bb182
bb182:
  %v436 = phi i32 [ 0, %bb181 ], [ %v514, %bb196 ]
  %v437 = icmp ult i32 %v436, 2
  %v438 = xor i1 %v437, 1
  br i1 %v438, label %bb197, label %bb183
bb183:
  %v439 = mul i32 %v436, 16
  %v440 = add i32 %v389, %v439
  br label %bb184
bb184:
  %v441 = phi i32 [ 0, %bb183 ], [ %v513, %bb195 ]
  %v442 = icmp ult i32 %v441, 8
  %v443 = xor i1 %v442, 1
  br i1 %v443, label %bb196, label %bb185
bb185:
  %v444 = mul i32 %v441, 16
  %v445 = zext i32 %v444 to i64
  %v446 = add i32 %v59, %v421
  %v447 = and i32 16, 31
  %v448 = shl i32 %v440, %v447
  %v449 = add i32 %v446, %v448
  %v450 = trunc i64 %v445 to i32
  %v451 = add i32 %v449, %v450
  %v452 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v451) #0
  %v453 = extractvalue { float, float, float, float } %v452, 0
  %v454 = extractvalue { float, float, float, float } %v452, 1
  %v455 = extractvalue { float, float, float, float } %v452, 2
  %v456 = extractvalue { float, float, float, float } %v452, 3
  %v457 = insertvalue [4 x float] undef, float %v453, 0
  %v458 = insertvalue [4 x float] %v457, float %v454, 1
  %v459 = insertvalue [4 x float] %v458, float %v455, 2
  %v460 = insertvalue [4 x float] %v459, float %v456, 3
  %v461 = insertvalue { [4 x float] } undef, [4 x float] %v460, 0
  br label %bb186
bb186:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb187
bb187:
  %v462 = add i32 %v451, 8
  %v463 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v462) #0
  %v464 = extractvalue { float, float, float, float } %v463, 0
  %v465 = extractvalue { float, float, float, float } %v463, 1
  %v466 = extractvalue { float, float, float, float } %v463, 2
  %v467 = extractvalue { float, float, float, float } %v463, 3
  %v468 = insertvalue [4 x float] undef, float %v464, 0
  %v469 = insertvalue [4 x float] %v468, float %v465, 1
  %v470 = insertvalue [4 x float] %v469, float %v466, 2
  %v471 = insertvalue [4 x float] %v470, float %v467, 3
  %v472 = insertvalue { [4 x float] } undef, [4 x float] %v471, 0
  br label %bb188
bb188:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb189
bb189:
  %v473 = extractvalue { [4 x float] } %v461, 0
  %v474 = extractvalue [4 x float] %v473, 0
  %v475 = extractvalue { [4 x float] } %v461, 0
  %v476 = extractvalue [4 x float] %v475, 1
  %v477 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v474, float %v476)
  br label %bb190
bb190:
  %v478 = extractvalue { [4 x float] } %v472, 0
  %v479 = extractvalue [4 x float] %v478, 0
  %v480 = extractvalue { [4 x float] } %v472, 0
  %v481 = extractvalue [4 x float] %v480, 1
  %v482 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v479, float %v481)
  br label %bb191
bb191:
  %v483 = zext i32 %v436 to i64
  %v484 = mul i64 %v483, 16
  %v485 = add i64 %v390, %v484
  %v486 = add i64 %v485, %v392
  %v488 = addrspacecast ptr addrspace(3) @__shared_mem_42 to ptr
  %v489 = mul i64 %v486, 256
  %v490 = mul i64 %v445, 2
  %v491 = add i64 %v489, %v490
  %v492 = add i64 %v491, %v398
  %v493 = getelementptr inbounds i8, ptr %v488, i64 %v492
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v493, i32 %v477, i32 %v482) #0
  br label %bb192
bb192:
  %v494 = extractvalue { [4 x float] } %v461, 0
  %v495 = extractvalue [4 x float] %v494, 2
  %v496 = extractvalue { [4 x float] } %v461, 0
  %v497 = extractvalue [4 x float] %v496, 3
  %v498 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v495, float %v497)
  br label %bb193
bb193:
  %v499 = extractvalue { [4 x float] } %v472, 0
  %v500 = extractvalue [4 x float] %v499, 2
  %v501 = extractvalue { [4 x float] } %v472, 0
  %v502 = extractvalue [4 x float] %v501, 3
  %v503 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v500, float %v502)
  br label %bb194
bb194:
  %v504 = zext i32 %v436 to i64
  %v505 = mul i64 %v504, 16
  %v506 = add i64 %v390, %v505
  %v507 = add i64 %v506, 8
  %v508 = add i64 %v507, %v392
  %v509 = mul i64 %v508, 256
  %v510 = add i64 %v509, %v490
  %v511 = add i64 %v510, %v398
  %v512 = getelementptr inbounds i8, ptr %v488, i64 %v511
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v512, i32 %v498, i32 %v503) #0
  br label %bb195
bb195:
  %v513 = add i32 %v441, 1
  br label %bb184
bb196:
  %v514 = add i32 %v436, 1
  br label %bb182
bb197:
  %v515 = udiv i32 %v18, 2
  %v516 = zext i32 %v515 to i64
  %v517 = mul i32 %v415, 128
  %v518 = zext i32 %v517 to i64
  %v519 = mul i32 %v419, 64
  %v520 = zext i32 %v519 to i64
  %v521 = zext i32 %v575 to i64
  %v522 = mul i64 %v521, 32
  %v523 = zext i32 %v576 to i64
  br label %bb198
bb198:
  %v524 = phi i64 [ %v523, %bb197 ], [ %v543, %bb200 ]
  %v525 = icmp ult i64 %v524, 2048
  %v526 = xor i1 %v525, 1
  br i1 %v526, label %bb201, label %bb199
bb199:
  %v527 = udiv i64 %v524, 64
  %v528 = urem i64 %v524, 64
  %v529 = add i64 %v522, %v527
  %v530 = mul i64 %v529, 64
  %v531 = add i64 %v530, %v528
  %v532 = add i64 %v518, %v522
  %v533 = add i64 %v532, %v527
  %v534 = add i64 %v520, %v528
  %v535 = mul i64 %v533, %v516
  %v536 = add i64 %v535, %v534
  %v538 = bitcast ptr addrspace(3) @__shared_mem_42 to ptr addrspace(3)
  %v539 = getelementptr inbounds i32, ptr addrspace(3) %v538, i64 %v531
  br label %bb200
bb200:
  %v540 = load i32, ptr addrspace(3) %v539
  %v541 = extractvalue { ptr, i64 } %v12, 0
  %v542 = getelementptr inbounds i32, ptr %v541, i64 %v536
  store i32 %v540, ptr %v542
  %v543 = add i64 %v524, 32
  br label %bb198
bb201:
  %v544 = xor i1 %v424, 1
  br i1 %v544, label %bb203, label %bb202
bb202:
  %v546 = bitcast ptr addrspace(3) @__shared_mem_31 to ptr addrspace(3)
  %v547 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v546) #0
  br label %bb204
bb203:
  %v549 = bitcast ptr addrspace(3) @__shared_mem_32 to ptr addrspace(3)
  %v550 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v549) #0
  br label %bb205
bb204:
  br label %bb206
bb205:
  br label %bb206
bb206:
  %v551 = add i32 %v399, 1
  br label %bb167
bb207:
  call void @llvm.nvvm.barrier0() #0
  br label %bb208
bb208:
  %v553 = xor i1 %v51, 1
  br i1 %v553, label %bb210, label %bb209
bb209:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v59, i32 512) #0
  br label %bb210
bb210:
  %v554 = xor i1 %v577, 1
  br i1 %v554, label %bb221, label %bb211
bb211:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_25) #0
  br label %bb212
bb212:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_26) #0
  br label %bb213
bb213:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_27) #0
  br label %bb214
bb214:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_28) #0
  br label %bb215
bb215:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_29) #0
  br label %bb216
bb216:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_30) #0
  br label %bb217
bb217:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_31) #0
  br label %bb218
bb218:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_32) #0
  br label %bb219
bb219:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_33) #0
  br label %bb220
bb220:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_34) #0
  br label %bb221
bb221:
  ret void
bb222:
  %v575 = udiv i32 %v21, 32
  %v576 = urem i32 %v20, 32
  %v577 = icmp eq i32 %v20, 0
  %v578 = icmp eq i32 %v20, 0
  br i1 %v578, label %bb3, label %bb14
bb223:
  unreachable
}

define ptx_kernel void @gemm_sol_clc_multicast_4_stage_pipeline(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7) {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi ptr [ %v0, %entry ]
  %v11 = phi ptr [ %v1, %entry ]
  %v12 = phi { ptr, i64 } [ %v9, %entry ]
  %v13 = phi i32 [ %v4, %entry ]
  %v14 = phi i32 [ %v5, %entry ]
  %v15 = phi i32 [ %v6, %entry ]
  %v16 = phi i32 [ %v7, %entry ]
  %v17 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  br label %bb1
bb1:
  %v18 = bitcast i32 %v13 to i32
  %v19 = bitcast i32 %v14 to i32
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb2
bb2:
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb229
bb4:
  %v23 = trunc i32 %v673 to i16
  %v24 = and i16 %v23, 15
  %v25 = shl i16 1, %v24
  %v26 = icmp eq i32 %v20, 0
  %v27 = icmp eq i32 %v20, 0
  br i1 %v27, label %bb5, label %bb20
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_43, i32 1) #0
  br label %bb6
bb6:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_44, i32 1) #0
  br label %bb7
bb7:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_45, i32 1) #0
  br label %bb8
bb8:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_46, i32 1) #0
  br label %bb9
bb9:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_47, i32 1) #0
  br label %bb10
bb10:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_48, i32 1) #0
  br label %bb11
bb11:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_49, i32 1) #0
  br label %bb12
bb12:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_50, i32 1) #0
  br label %bb13
bb13:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_51, i32 1) #0
  br label %bb14
bb14:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_52, i32 1) #0
  br label %bb15
bb15:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_53, i32 256) #0
  br label %bb16
bb16:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_54, i32 256) #0
  br label %bb17
bb17:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_55, i32 1) #0
  br label %bb18
bb18:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_56, i32 1) #0
  br label %bb19
bb19:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb20
bb20:
  call void @llvm.nvvm.barrier0() #0
  br label %bb21
bb21:
  %v57 = xor i1 %v26, 1
  br i1 %v57, label %bb27, label %bb22
bb22:
  %v59 = bitcast ptr addrspace(3) @__shared_mem_47 to ptr addrspace(3)
  %v60 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v59) #0
  br label %bb23
bb23:
  %v62 = bitcast ptr addrspace(3) @__shared_mem_48 to ptr addrspace(3)
  %v63 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v62) #0
  br label %bb24
bb24:
  %v65 = bitcast ptr addrspace(3) @__shared_mem_49 to ptr addrspace(3)
  %v66 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v65) #0
  br label %bb25
bb25:
  %v68 = bitcast ptr addrspace(3) @__shared_mem_50 to ptr addrspace(3)
  %v69 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v68) #0
  br label %bb26
bb26:
  br label %bb27
bb27:
  call void @llvm.nvvm.barrier0() #0
  br label %bb28
bb28:
  %v71 = icmp eq i32 %v671, 0
  %v72 = icmp eq i32 %v671, 0
  br i1 %v72, label %bb29, label %bb31
bb29:
  %v74 = addrspacecast ptr addrspace(3) @__shared_mem_57 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v74, i32 512) #0
  br label %bb30
bb30:
  br label %bb31
bb31:
  call void @llvm.nvvm.barrier0() #0
  br label %bb32
bb32:
  %v77 = bitcast ptr addrspace(3) @__shared_mem_57 to ptr addrspace(3)
  %v78 = addrspacecast ptr addrspace(3) %v77 to ptr
  %v79 = load i32, ptr %v78
  %v80 = icmp eq i32 %v673, 0
  %v81 = insertvalue { i8 } undef, i8 1, 0
  %v82 = insertvalue { i8 } undef, i8 0, 0
  %v83 = insertvalue { i8 } undef, i8 0, 0
  %v84 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v85 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v84, i32 128, 1
  %v86 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v85, i8 0, 2
  %v87 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v86, { i8 } %v82, 3
  %v88 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v87, { i8 } %v83, 4
  %v89 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v88, i1 0, 5
  %v90 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v89, { i8 } %v81, 6
  %v91 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v90, i1 0, 7
  %v92 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v91, i1 0, 8
  %v93 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v92, i1 0, 9
  %v94 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v93, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v94, ptr %v17
  %v95 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 1
  store i32 256, ptr %v95
  %v96 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 0
  store i32 128, ptr %v96
  %v97 = insertvalue { i8 } undef, i8 0, 0
  %v98 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 3
  store { i8 } %v97, ptr %v98
  %v99 = insertvalue { i8 } undef, i8 0, 0
  %v100 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 4
  store { i8 } %v99, ptr %v100
  %v101 = insertvalue { i8 } undef, i8 1, 0
  %v102 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 6
  store { i8 } %v101, ptr %v102
  %v103 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17
  %v104 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 0
  %v105 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 1
  %v106 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 2
  %v107 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 3
  %v108 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 4
  %v109 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 5
  %v110 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 6
  %v111 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 7
  %v112 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 8
  %v113 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 9
  %v114 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v103, 10
  %v115 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v104, i32 %v105, i8 %v106, { i8 } %v107, { i8 } %v108, i1 %v109, { i8 } %v110, i1 %v111, i1 %v112, i1 %v113, i1 %v114)
  br label %bb33
bb33:
  %v116 = extractvalue { i32 } %v115, 0
  %v117 = udiv i32 %v19, 64
  call void asm sideeffect "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}"() #0
  br label %bb34
bb34:
  %v118 = icmp eq i32 %v671, 4
  br i1 %v118, label %bb35, label %bb100
bb35:
  %v119 = icmp eq i32 %v672, 0
  %v120 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb36
bb36:
  %v121 = sub i32 %v120, %v673
  %v122 = udiv i32 %v121, 2
  %v123 = icmp eq i32 %v15, 0
  %v124 = xor i1 %v123, 1
  br i1 %v124, label %bb37, label %bb230
bb37:
  %v125 = urem i32 %v122, %v15
  %v126 = udiv i32 %v122, %v15
  %v127 = xor i1 %v119, 1
  br i1 %v127, label %bb40, label %bb38
bb38:
  %v129 = addrspacecast ptr addrspace(3) @__shared_mem_58 to ptr
  store i32 %v125, ptr %v129
  %v130 = getelementptr inbounds i32, ptr %v129, i64 1
  store i32 %v126, ptr %v130
  %v131 = getelementptr inbounds i32, ptr %v129, i64 2
  store i32 1, ptr %v131
  %v133 = bitcast ptr addrspace(3) @__shared_mem_55 to ptr addrspace(3)
  %v134 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v133) #0
  br label %bb39
bb39:
  br label %bb40
bb40:
  %v135 = mul i32 %v125, 256
  %v136 = mul i32 %v673, 128
  %v137 = add i32 %v135, %v136
  %v138 = bitcast i32 %v137 to i32
  %v139 = mul i32 %v126, 128
  %v140 = mul i32 %v673, 64
  %v141 = add i32 %v139, %v140
  %v142 = bitcast i32 %v141 to i32
  br label %bb41
bb41:
  %v143 = phi i32 [ 0, %bb40 ], [ %v211, %bb59 ]
  %v144 = icmp ult i32 %v143, %v117
  %v145 = xor i1 %v144, 1
  br i1 %v145, label %bb60, label %bb42
bb42:
  %v146 = mul i32 0, %v117
  %v147 = add i32 %v146, %v143
  %v148 = and i32 %v147, 3
  %v149 = and i32 2, 31
  %v150 = lshr i32 %v147, %v149
  %v151 = and i32 %v150, 1
  %v152 = icmp eq i32 %v148, 0
  br i1 %v152, label %bb48, label %bb43
bb43:
  %v153 = icmp eq i32 %v148, 1
  br i1 %v153, label %bb47, label %bb44
bb44:
  %v154 = icmp eq i32 %v148, 2
  br i1 %v154, label %bb46, label %bb45
bb45:
  %v156 = addrspacecast ptr addrspace(3) @__shared_mem_59 to ptr
  %v158 = addrspacecast ptr addrspace(3) @__shared_mem_60 to ptr
  %v160 = bitcast ptr addrspace(3) @__shared_mem_46 to ptr addrspace(3)
  %v162 = bitcast ptr addrspace(3) @__shared_mem_50 to ptr addrspace(3)
  br label %bb49
bb46:
  %v165 = addrspacecast ptr addrspace(3) @__shared_mem_61 to ptr
  %v167 = addrspacecast ptr addrspace(3) @__shared_mem_62 to ptr
  %v169 = bitcast ptr addrspace(3) @__shared_mem_45 to ptr addrspace(3)
  %v171 = bitcast ptr addrspace(3) @__shared_mem_49 to ptr addrspace(3)
  br label %bb49
bb47:
  %v174 = addrspacecast ptr addrspace(3) @__shared_mem_63 to ptr
  %v176 = addrspacecast ptr addrspace(3) @__shared_mem_64 to ptr
  %v178 = bitcast ptr addrspace(3) @__shared_mem_44 to ptr addrspace(3)
  %v180 = bitcast ptr addrspace(3) @__shared_mem_48 to ptr addrspace(3)
  br label %bb49
bb48:
  %v183 = addrspacecast ptr addrspace(3) @__shared_mem_65 to ptr
  %v185 = addrspacecast ptr addrspace(3) @__shared_mem_66 to ptr
  %v187 = bitcast ptr addrspace(3) @__shared_mem_43 to ptr addrspace(3)
  %v189 = bitcast ptr addrspace(3) @__shared_mem_47 to ptr addrspace(3)
  br label %bb49
bb49:
  %v191 = phi ptr [ %v156, %bb45 ], [ %v165, %bb46 ], [ %v174, %bb47 ], [ %v183, %bb48 ]
  %v192 = phi ptr [ %v158, %bb45 ], [ %v167, %bb46 ], [ %v176, %bb47 ], [ %v185, %bb48 ]
  %v193 = phi ptr addrspace(3) [ %v160, %bb45 ], [ %v169, %bb46 ], [ %v178, %bb47 ], [ %v187, %bb48 ]
  %v194 = phi ptr addrspace(3) [ %v162, %bb45 ], [ %v171, %bb46 ], [ %v180, %bb47 ], [ %v189, %bb48 ]
  %v195 = phi ptr addrspace(3) [ @__shared_mem_46, %bb45 ], [ @__shared_mem_45, %bb46 ], [ @__shared_mem_44, %bb47 ], [ @__shared_mem_43, %bb48 ]
  br label %bb50
bb50:
  %v196 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v194, i32 %v151) #0
  %v197 = trunc i32 %v196 to i1
  br label %bb51
bb51:
  %v198 = xor i1 %v197, 1
  br i1 %v198, label %bb53, label %bb52
bb52:
  %v199 = xor i1 %v119, 1
  br i1 %v199, label %bb59, label %bb54
bb53:
  br label %bb50
bb54:
  %v200 = xor i1 %v80, 1
  br i1 %v200, label %bb57, label %bb55
bb55:
  %v201 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v193, i32 49152) #0
  br label %bb56
bb56:
  br label %bb57
bb57:
  %v202 = ptrtoint ptr addrspace(3) %v195 to i32
  %v203 = and i32 %v202, 4278190072
  %v204 = inttoptr i32 %v203 to ptr addrspace(3)
  %v205 = mul i32 %v143, 64
  %v206 = bitcast i32 %v205 to i32
  %v207 = addrspacecast ptr %v191 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v207, ptr addrspace(3) %v204, ptr %v10, i32 %v206, i32 %v138, i16 %v25, i64 0, i1 1, i1 0, i32 2) #0
  br label %bb58
bb58:
  %v209 = addrspacecast ptr %v192 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v209, ptr addrspace(3) %v204, ptr %v11, i32 %v206, i32 %v142, i16 %v25, i64 0, i1 1, i1 0, i32 2) #0
  br label %bb59
bb59:
  %v211 = add i32 %v143, 1
  br label %bb41
bb60:
  %v212 = add i32 0, 1
  %v214 = addrspacecast ptr addrspace(3) @__shared_mem_67 to ptr
  br label %bb61
bb61:
  %v215 = phi i32 [ %v212, %bb60 ], [ %v330, %bb99 ]
  %v216 = phi i32 [ 0, %bb60 ], [ %v331, %bb99 ]
  %v217 = and i32 %v216, 1
  %v218 = xor i1 %v119, 1
  br i1 %v218, label %bb66, label %bb62
bb62:
  %v220 = bitcast ptr addrspace(3) @__shared_mem_56 to ptr addrspace(3)
  %v221 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v220, i32 16) #0
  br label %bb63
bb63:
  %v222 = xor i1 %v80, 1
  br i1 %v222, label %bb66, label %bb64
bb64:
  %v224 = addrspacecast ptr addrspace(3) @__shared_mem_67 to ptr
  call void asm sideeffect "{ .reg .u64 %resp_shared64; .reg .u32 %resp_shared32; cvta.to.shared.u64 %resp_shared64, $0; cvt.u32.u64 %resp_shared32, %resp_shared64; .reg .u64 %mbar_shared64; .reg .u32 %mbar_shared32; cvta.to.shared.u64 %mbar_shared64, $1; cvt.u32.u64 %mbar_shared32, %mbar_shared64; clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%resp_shared32], [%mbar_shared32]; }", "l,l,~{memory}"(ptr %v224, ptr addrspace(3) @__shared_mem_56) #0
  br label %bb65
bb65:
  br label %bb66
bb66:
  %v227 = bitcast ptr addrspace(3) @__shared_mem_56 to ptr addrspace(3)
  %v228 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v227, i32 %v217) #0
  %v229 = trunc i32 %v228 to i1
  br label %bb67
bb67:
  %v230 = xor i1 %v229, 1
  br i1 %v230, label %bb69, label %bb68
bb68:
  %v231 = load i64, ptr %v214
  %v232 = getelementptr inbounds i64, ptr %v214, i64 1
  %v233 = load i64, ptr %v232
  %v234 = call i32 asm sideeffect "{ .reg .b128 %resp; mov.b128 %resp, {$1, $2}; .reg .pred %p; clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %resp; selp.b32 $0, 1, 0, %p; }", "=r,l,l"(i64 %v231, i64 %v233) #0
  br label %bb70
bb69:
  br label %bb66
bb70:
  %v235 = icmp eq i32 %v234, 0
  br i1 %v235, label %bb71, label %bb75
bb71:
  %v236 = xor i1 %v119, 1
  br i1 %v236, label %bb74, label %bb72
bb72:
  %v238 = addrspacecast ptr addrspace(3) @__shared_mem_58 to ptr
  %v239 = getelementptr inbounds i32, ptr %v238, i64 2
  store i32 0, ptr %v239
  %v241 = bitcast ptr addrspace(3) @__shared_mem_55 to ptr addrspace(3)
  %v242 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v241) #0
  br label %bb73
bb73:
  br label %bb74
bb74:
  br label %bb100
bb75:
  %v243 = call i32 asm sideeffect "{ .reg .b128 %resp; mov.b128 %resp, {$1, $2}; clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 $0, %resp; }", "=r,l,l"(i64 %v231, i64 %v233) #0
  br label %bb76
bb76:
  %v244 = udiv i32 %v243, 2
  %v245 = urem i32 %v244, %v15
  %v246 = udiv i32 %v244, %v15
  %v247 = xor i1 %v119, 1
  br i1 %v247, label %bb79, label %bb77
bb77:
  %v249 = addrspacecast ptr addrspace(3) @__shared_mem_58 to ptr
  store i32 %v245, ptr %v249
  %v250 = getelementptr inbounds i32, ptr %v249, i64 1
  store i32 %v246, ptr %v250
  %v251 = getelementptr inbounds i32, ptr %v249, i64 2
  store i32 1, ptr %v251
  %v253 = bitcast ptr addrspace(3) @__shared_mem_55 to ptr addrspace(3)
  %v254 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v253) #0
  br label %bb78
bb78:
  br label %bb79
bb79:
  %v255 = mul i32 %v245, 256
  %v256 = add i32 %v255, %v136
  %v257 = bitcast i32 %v256 to i32
  %v258 = mul i32 %v246, 128
  %v259 = add i32 %v258, %v140
  %v260 = bitcast i32 %v259 to i32
  br label %bb80
bb80:
  %v261 = phi i32 [ 0, %bb79 ], [ %v329, %bb98 ]
  %v262 = icmp ult i32 %v261, %v117
  %v263 = xor i1 %v262, 1
  br i1 %v263, label %bb99, label %bb81
bb81:
  %v264 = mul i32 %v215, %v117
  %v265 = add i32 %v264, %v261
  %v266 = and i32 %v265, 3
  %v267 = and i32 2, 31
  %v268 = lshr i32 %v265, %v267
  %v269 = and i32 %v268, 1
  %v270 = icmp eq i32 %v266, 0
  br i1 %v270, label %bb87, label %bb82
bb82:
  %v271 = icmp eq i32 %v266, 1
  br i1 %v271, label %bb86, label %bb83
bb83:
  %v272 = icmp eq i32 %v266, 2
  br i1 %v272, label %bb85, label %bb84
bb84:
  %v274 = addrspacecast ptr addrspace(3) @__shared_mem_59 to ptr
  %v276 = addrspacecast ptr addrspace(3) @__shared_mem_60 to ptr
  %v278 = bitcast ptr addrspace(3) @__shared_mem_46 to ptr addrspace(3)
  %v280 = bitcast ptr addrspace(3) @__shared_mem_50 to ptr addrspace(3)
  br label %bb88
bb85:
  %v283 = addrspacecast ptr addrspace(3) @__shared_mem_61 to ptr
  %v285 = addrspacecast ptr addrspace(3) @__shared_mem_62 to ptr
  %v287 = bitcast ptr addrspace(3) @__shared_mem_45 to ptr addrspace(3)
  %v289 = bitcast ptr addrspace(3) @__shared_mem_49 to ptr addrspace(3)
  br label %bb88
bb86:
  %v292 = addrspacecast ptr addrspace(3) @__shared_mem_63 to ptr
  %v294 = addrspacecast ptr addrspace(3) @__shared_mem_64 to ptr
  %v296 = bitcast ptr addrspace(3) @__shared_mem_44 to ptr addrspace(3)
  %v298 = bitcast ptr addrspace(3) @__shared_mem_48 to ptr addrspace(3)
  br label %bb88
bb87:
  %v301 = addrspacecast ptr addrspace(3) @__shared_mem_65 to ptr
  %v303 = addrspacecast ptr addrspace(3) @__shared_mem_66 to ptr
  %v305 = bitcast ptr addrspace(3) @__shared_mem_43 to ptr addrspace(3)
  %v307 = bitcast ptr addrspace(3) @__shared_mem_47 to ptr addrspace(3)
  br label %bb88
bb88:
  %v309 = phi ptr [ %v274, %bb84 ], [ %v283, %bb85 ], [ %v292, %bb86 ], [ %v301, %bb87 ]
  %v310 = phi ptr [ %v276, %bb84 ], [ %v285, %bb85 ], [ %v294, %bb86 ], [ %v303, %bb87 ]
  %v311 = phi ptr addrspace(3) [ %v278, %bb84 ], [ %v287, %bb85 ], [ %v296, %bb86 ], [ %v305, %bb87 ]
  %v312 = phi ptr addrspace(3) [ %v280, %bb84 ], [ %v289, %bb85 ], [ %v298, %bb86 ], [ %v307, %bb87 ]
  %v313 = phi ptr addrspace(3) [ @__shared_mem_46, %bb84 ], [ @__shared_mem_45, %bb85 ], [ @__shared_mem_44, %bb86 ], [ @__shared_mem_43, %bb87 ]
  br label %bb89
bb89:
  %v314 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v312, i32 %v269) #0
  %v315 = trunc i32 %v314 to i1
  br label %bb90
bb90:
  %v316 = xor i1 %v315, 1
  br i1 %v316, label %bb92, label %bb91
bb91:
  %v317 = xor i1 %v119, 1
  br i1 %v317, label %bb98, label %bb93
bb92:
  br label %bb89
bb93:
  %v318 = xor i1 %v80, 1
  br i1 %v318, label %bb96, label %bb94
bb94:
  %v319 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v311, i32 49152) #0
  br label %bb95
bb95:
  br label %bb96
bb96:
  %v320 = ptrtoint ptr addrspace(3) %v313 to i32
  %v321 = and i32 %v320, 4278190072
  %v322 = inttoptr i32 %v321 to ptr addrspace(3)
  %v323 = mul i32 %v261, 64
  %v324 = bitcast i32 %v323 to i32
  %v325 = addrspacecast ptr %v309 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v325, ptr addrspace(3) %v322, ptr %v10, i32 %v324, i32 %v257, i16 %v25, i64 0, i1 1, i1 0, i32 2) #0
  br label %bb97
bb97:
  %v327 = addrspacecast ptr %v310 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v327, ptr addrspace(3) %v322, ptr %v11, i32 %v324, i32 %v260, i16 %v25, i64 0, i1 1, i1 0, i32 2) #0
  br label %bb98
bb98:
  %v329 = add i32 %v261, 1
  br label %bb80
bb99:
  %v330 = add i32 %v215, 1
  %v331 = add i32 %v216, 1
  br label %bb61
bb100:
  %v332 = icmp eq i32 %v671, 5
  br i1 %v332, label %bb101, label %bb156
bb101:
  %v333 = icmp eq i32 %v672, 0
  br label %bb102
bb102:
  %v334 = phi i32 [ 0, %bb101 ], [ %v334, %bb105 ], [ %v463, %bb153 ]
  %v335 = phi i32 [ 0, %bb101 ], [ %v335, %bb105 ], [ %v341, %bb153 ]
  %v337 = bitcast ptr addrspace(3) @__shared_mem_55 to ptr addrspace(3)
  %v338 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v337, i32 %v335) #0
  %v339 = trunc i32 %v338 to i1
  br label %bb103
bb103:
  %v340 = xor i1 %v339, 1
  br i1 %v340, label %bb105, label %bb104
bb104:
  %v341 = xor i32 %v335, 1
  %v343 = bitcast ptr addrspace(3) @__shared_mem_58 to ptr addrspace(3)
  %v344 = addrspacecast ptr addrspace(3) %v343 to ptr
  %v345 = getelementptr inbounds i32, ptr %v344, i64 2
  %v346 = load i32, ptr %v345
  %v347 = icmp eq i32 %v346, 0
  br i1 %v347, label %bb106, label %bb107
bb105:
  br label %bb102
bb106:
  %v348 = xor i1 %v80, 1
  br i1 %v348, label %bb155, label %bb154
bb107:
  %v349 = urem i32 %v334, 2
  %v350 = mul i32 %v349, 128
  %v351 = xor i1 %v80, 1
  br i1 %v351, label %bb121, label %bb108
bb108:
  %v352 = icmp uge i32 %v334, 2
  %v353 = xor i1 %v352, 1
  br i1 %v353, label %bb119, label %bb109
bb109:
  %v354 = sub i32 %v334, 2
  %v355 = udiv i32 %v354, 2
  %v356 = and i32 %v355, 1
  %v357 = icmp eq i32 %v349, 0
  br i1 %v357, label %bb110, label %bb114
bb110:
  %v359 = bitcast ptr addrspace(3) @__shared_mem_53 to ptr addrspace(3)
  %v360 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v359, i32 %v356) #0
  %v361 = trunc i32 %v360 to i1
  br label %bb111
bb111:
  %v362 = xor i1 %v361, 1
  br i1 %v362, label %bb113, label %bb112
bb112:
  br label %bb118
bb113:
  br label %bb110
bb114:
  %v364 = bitcast ptr addrspace(3) @__shared_mem_54 to ptr addrspace(3)
  %v365 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v364, i32 %v356) #0
  %v366 = trunc i32 %v365 to i1
  br label %bb115
bb115:
  %v367 = xor i1 %v366, 1
  br i1 %v367, label %bb117, label %bb116
bb116:
  br label %bb118
bb117:
  br label %bb114
bb118:
  br label %bb120
bb119:
  br label %bb120
bb120:
  br label %bb121
bb121:
  %v368 = mul i32 %v334, %v117
  br label %bb122
bb122:
  %v369 = phi i32 [ 0, %bb121 ], [ %v455, %bb144 ]
  %v370 = icmp ult i32 %v369, %v117
  %v371 = xor i1 %v370, 1
  br i1 %v371, label %bb145, label %bb123
bb123:
  %v372 = add i32 %v368, %v369
  %v373 = and i32 %v372, 3
  %v374 = and i32 2, 31
  %v375 = lshr i32 %v372, %v374
  %v376 = and i32 %v375, 1
  %v377 = icmp eq i32 %v373, 0
  br i1 %v377, label %bb129, label %bb124
bb124:
  %v378 = icmp eq i32 %v373, 1
  br i1 %v378, label %bb128, label %bb125
bb125:
  %v379 = icmp eq i32 %v373, 2
  br i1 %v379, label %bb127, label %bb126
bb126:
  %v381 = bitcast ptr addrspace(3) @__shared_mem_59 to ptr addrspace(3)
  %v382 = ptrtoint ptr addrspace(3) %v381 to i64
  %v384 = bitcast ptr addrspace(3) @__shared_mem_60 to ptr addrspace(3)
  %v385 = ptrtoint ptr addrspace(3) %v384 to i64
  %v387 = bitcast ptr addrspace(3) @__shared_mem_46 to ptr addrspace(3)
  br label %bb130
bb127:
  %v390 = bitcast ptr addrspace(3) @__shared_mem_61 to ptr addrspace(3)
  %v391 = ptrtoint ptr addrspace(3) %v390 to i64
  %v393 = bitcast ptr addrspace(3) @__shared_mem_62 to ptr addrspace(3)
  %v394 = ptrtoint ptr addrspace(3) %v393 to i64
  %v396 = bitcast ptr addrspace(3) @__shared_mem_45 to ptr addrspace(3)
  br label %bb130
bb128:
  %v399 = bitcast ptr addrspace(3) @__shared_mem_63 to ptr addrspace(3)
  %v400 = ptrtoint ptr addrspace(3) %v399 to i64
  %v402 = bitcast ptr addrspace(3) @__shared_mem_64 to ptr addrspace(3)
  %v403 = ptrtoint ptr addrspace(3) %v402 to i64
  %v405 = bitcast ptr addrspace(3) @__shared_mem_44 to ptr addrspace(3)
  br label %bb130
bb129:
  %v408 = bitcast ptr addrspace(3) @__shared_mem_65 to ptr addrspace(3)
  %v409 = ptrtoint ptr addrspace(3) %v408 to i64
  %v411 = bitcast ptr addrspace(3) @__shared_mem_66 to ptr addrspace(3)
  %v412 = ptrtoint ptr addrspace(3) %v411 to i64
  %v414 = bitcast ptr addrspace(3) @__shared_mem_43 to ptr addrspace(3)
  br label %bb130
bb130:
  %v416 = phi i64 [ %v382, %bb126 ], [ %v391, %bb127 ], [ %v400, %bb128 ], [ %v409, %bb129 ]
  %v417 = phi i64 [ %v385, %bb126 ], [ %v394, %bb127 ], [ %v403, %bb128 ], [ %v412, %bb129 ]
  %v418 = phi ptr addrspace(3) [ %v387, %bb126 ], [ %v396, %bb127 ], [ %v405, %bb128 ], [ %v414, %bb129 ]
  %v419 = phi ptr addrspace(3) [ @__shared_mem_50, %bb126 ], [ @__shared_mem_49, %bb127 ], [ @__shared_mem_48, %bb128 ], [ @__shared_mem_47, %bb129 ]
  %v420 = xor i1 %v80, 1
  br i1 %v420, label %bb144, label %bb131
bb131:
  %v421 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v418, i32 %v376) #0
  %v422 = trunc i32 %v421 to i1
  br label %bb132
bb132:
  %v423 = xor i1 %v422, 1
  br i1 %v423, label %bb134, label %bb133
bb133:
  %v424 = xor i1 %v333, 1
  br i1 %v424, label %bb144, label %bb135
bb134:
  br label %bb131
bb135:
  br label %bb136
bb136:
  %v425 = phi i32 [ 0, %bb135 ], [ %v453, %bb141 ]
  %v426 = icmp ult i32 %v425, 4
  %v427 = xor i1 %v426, 1
  br i1 %v427, label %bb142, label %bb137
bb137:
  %v428 = mul i32 %v425, 32
  %v429 = zext i32 %v428 to i64
  %v430 = add i64 %v416, %v429
  %v431 = zext i32 4 to i64
  %v432 = and i64 %v431, 63
  %v433 = lshr i64 %v430, %v432
  %v434 = and i64 %v433, 16383
  %v435 = or i64 %v434, 65536
  %v436 = or i64 %v435, 274877906944
  %v437 = or i64 %v436, 70368744177664
  %v438 = or i64 %v437, 4611686018427387904
  %v439 = add i64 %v417, %v429
  %v440 = zext i32 4 to i64
  %v441 = and i64 %v440, 63
  %v442 = lshr i64 %v439, %v441
  %v443 = and i64 %v442, 16383
  %v444 = or i64 %v443, 65536
  %v445 = or i64 %v444, 274877906944
  %v446 = or i64 %v445, 70368744177664
  %v447 = or i64 %v446, 4611686018427387904
  %v448 = icmp ugt i32 %v369, 0
  %v449 = xor i1 %v448, 1
  br i1 %v449, label %bb139, label %bb138
bb138:
  br label %bb140
bb139:
  %v450 = icmp ugt i32 %v425, 0
  br label %bb140
bb140:
  %v451 = phi i1 [ 1, %bb138 ], [ %v450, %bb139 ]
  %v452 = add i32 %v79, %v350
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::2.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z, %z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v452, i64 %v438, i64 %v447, i32 %v116, i1 %v451) #0
  br label %bb141
bb141:
  %v453 = add i32 %v425, 1
  br label %bb136
bb142:
  %v454 = addrspacecast ptr addrspace(3) %v419 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;", "r,h,~{memory}"(ptr %v454, i16 3) #0
  br label %bb143
bb143:
  br label %bb144
bb144:
  %v455 = add i32 %v369, 1
  br label %bb122
bb145:
  %v456 = xor i1 %v80, 1
  br i1 %v456, label %bb153, label %bb146
bb146:
  %v457 = xor i1 %v333, 1
  br i1 %v457, label %bb153, label %bb147
bb147:
  %v458 = icmp eq i32 %v349, 0
  br i1 %v458, label %bb148, label %bb150
bb148:
  %v460 = addrspacecast ptr addrspace(3) @__shared_mem_51 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;", "r,h,~{memory}"(ptr %v460, i16 3) #0
  br label %bb149
bb149:
  br label %bb152
bb150:
  %v462 = addrspacecast ptr addrspace(3) @__shared_mem_52 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;", "r,h,~{memory}"(ptr %v462, i16 3) #0
  br label %bb151
bb151:
  br label %bb152
bb152:
  br label %bb153
bb153:
  %v463 = add i32 %v334, 1
  br label %bb102
bb154:
  call void asm sideeffect "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;", "~{memory}"() #0
  br label %bb155
bb155:
  br label %bb156
bb156:
  %v464 = icmp ult i32 %v671, 4
  %v465 = xor i1 %v464, 1
  br i1 %v465, label %bb210, label %bb157
bb157:
  %v467 = bitcast ptr addrspace(3) @__shared_mem_53 to ptr addrspace(3)
  %v468 = call i64 asm sideeffect "mapa.shared::cluster.u64 $0, $1, $2;", "=l,l,r"(ptr addrspace(3) %v467, i32 0) #0
  %v469 = inttoptr i64 %v468 to ptr addrspace(3)
  br label %bb158
bb158:
  %v470 = ptrtoint ptr addrspace(3) %v469 to i64
  %v472 = bitcast ptr addrspace(3) @__shared_mem_54 to ptr addrspace(3)
  %v473 = call i64 asm sideeffect "mapa.shared::cluster.u64 $0, $1, $2;", "=l,l,r"(ptr addrspace(3) %v472, i32 0) #0
  %v474 = inttoptr i64 %v473 to ptr addrspace(3)
  br label %bb159
bb159:
  %v475 = ptrtoint ptr addrspace(3) %v474 to i64
  %v476 = mul i32 %v671, 32
  %v477 = zext i32 %v476 to i64
  %v478 = urem i32 %v672, 8
  %v479 = zext i32 %v478 to i64
  %v480 = icmp uge i32 %v672, 8
  %v481 = xor i1 %v480, 1
  br i1 %v481, label %bb161, label %bb160
bb160:
  %v482 = icmp ult i32 %v672, 16
  br label %bb162
bb161:
  br label %bb162
bb162:
  %v483 = phi i1 [ %v482, %bb160 ], [ 0, %bb161 ]
  %v484 = xor i1 %v483, 1
  br i1 %v484, label %bb164, label %bb163
bb163:
  br label %bb165
bb164:
  br label %bb165
bb165:
  %v485 = phi i64 [ 16, %bb163 ], [ 0, %bb164 ]
  br label %bb166
bb166:
  %v486 = phi i32 [ 0, %bb165 ], [ %v486, %bb169 ], [ %v640, %bb209 ]
  %v487 = phi i32 [ 0, %bb165 ], [ %v487, %bb169 ], [ %v493, %bb209 ]
  %v489 = bitcast ptr addrspace(3) @__shared_mem_55 to ptr addrspace(3)
  %v490 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v489, i32 %v487) #0
  %v491 = trunc i32 %v490 to i1
  br label %bb167
bb167:
  %v492 = xor i1 %v491, 1
  br i1 %v492, label %bb169, label %bb168
bb168:
  %v493 = xor i32 %v487, 1
  %v495 = bitcast ptr addrspace(3) @__shared_mem_58 to ptr addrspace(3)
  %v496 = addrspacecast ptr addrspace(3) %v495 to ptr
  %v497 = getelementptr inbounds i32, ptr %v496, i64 2
  %v498 = load i32, ptr %v497
  %v499 = icmp eq i32 %v498, 0
  br i1 %v499, label %bb170, label %bb171
bb169:
  br label %bb166
bb170:
  br label %bb210
bb171:
  %v500 = bitcast ptr addrspace(3) @__shared_mem_58 to ptr addrspace(3)
  %v501 = addrspacecast ptr addrspace(3) %v500 to ptr
  %v502 = load i32, ptr %v501
  %v503 = bitcast ptr addrspace(3) @__shared_mem_58 to ptr addrspace(3)
  %v504 = addrspacecast ptr addrspace(3) %v503 to ptr
  %v505 = getelementptr inbounds i32, ptr %v504, i64 1
  %v506 = load i32, ptr %v505
  %v507 = urem i32 %v486, 2
  %v508 = mul i32 %v507, 128
  %v509 = udiv i32 %v486, 2
  %v510 = and i32 %v509, 1
  %v511 = icmp eq i32 %v507, 0
  %v512 = icmp eq i32 %v507, 0
  br i1 %v512, label %bb172, label %bb176
bb172:
  %v514 = bitcast ptr addrspace(3) @__shared_mem_51 to ptr addrspace(3)
  %v515 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v514, i32 %v510) #0
  %v516 = trunc i32 %v515 to i1
  br label %bb173
bb173:
  %v517 = xor i1 %v516, 1
  br i1 %v517, label %bb175, label %bb174
bb174:
  br label %bb180
bb175:
  br label %bb172
bb176:
  %v519 = bitcast ptr addrspace(3) @__shared_mem_52 to ptr addrspace(3)
  %v520 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v519, i32 %v510) #0
  %v521 = trunc i32 %v520 to i1
  br label %bb177
bb177:
  %v522 = xor i1 %v521, 1
  br i1 %v522, label %bb179, label %bb178
bb178:
  br label %bb180
bb179:
  br label %bb176
bb180:
  br label %bb181
bb181:
  %v523 = phi i32 [ 0, %bb180 ], [ %v601, %bb195 ]
  %v524 = icmp ult i32 %v523, 2
  %v525 = xor i1 %v524, 1
  br i1 %v525, label %bb196, label %bb182
bb182:
  %v526 = mul i32 %v523, 16
  %v527 = add i32 %v476, %v526
  br label %bb183
bb183:
  %v528 = phi i32 [ 0, %bb182 ], [ %v600, %bb194 ]
  %v529 = icmp ult i32 %v528, 8
  %v530 = xor i1 %v529, 1
  br i1 %v530, label %bb195, label %bb184
bb184:
  %v531 = mul i32 %v528, 16
  %v532 = zext i32 %v531 to i64
  %v533 = add i32 %v79, %v508
  %v534 = and i32 16, 31
  %v535 = shl i32 %v527, %v534
  %v536 = add i32 %v533, %v535
  %v537 = trunc i64 %v532 to i32
  %v538 = add i32 %v536, %v537
  %v539 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v538) #0
  %v540 = extractvalue { float, float, float, float } %v539, 0
  %v541 = extractvalue { float, float, float, float } %v539, 1
  %v542 = extractvalue { float, float, float, float } %v539, 2
  %v543 = extractvalue { float, float, float, float } %v539, 3
  %v544 = insertvalue [4 x float] undef, float %v540, 0
  %v545 = insertvalue [4 x float] %v544, float %v541, 1
  %v546 = insertvalue [4 x float] %v545, float %v542, 2
  %v547 = insertvalue [4 x float] %v546, float %v543, 3
  %v548 = insertvalue { [4 x float] } undef, [4 x float] %v547, 0
  br label %bb185
bb185:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb186
bb186:
  %v549 = add i32 %v538, 8
  %v550 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v549) #0
  %v551 = extractvalue { float, float, float, float } %v550, 0
  %v552 = extractvalue { float, float, float, float } %v550, 1
  %v553 = extractvalue { float, float, float, float } %v550, 2
  %v554 = extractvalue { float, float, float, float } %v550, 3
  %v555 = insertvalue [4 x float] undef, float %v551, 0
  %v556 = insertvalue [4 x float] %v555, float %v552, 1
  %v557 = insertvalue [4 x float] %v556, float %v553, 2
  %v558 = insertvalue [4 x float] %v557, float %v554, 3
  %v559 = insertvalue { [4 x float] } undef, [4 x float] %v558, 0
  br label %bb187
bb187:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb188
bb188:
  %v560 = extractvalue { [4 x float] } %v548, 0
  %v561 = extractvalue [4 x float] %v560, 0
  %v562 = extractvalue { [4 x float] } %v548, 0
  %v563 = extractvalue [4 x float] %v562, 1
  %v564 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v561, float %v563)
  br label %bb189
bb189:
  %v565 = extractvalue { [4 x float] } %v559, 0
  %v566 = extractvalue [4 x float] %v565, 0
  %v567 = extractvalue { [4 x float] } %v559, 0
  %v568 = extractvalue [4 x float] %v567, 1
  %v569 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v566, float %v568)
  br label %bb190
bb190:
  %v570 = zext i32 %v523 to i64
  %v571 = mul i64 %v570, 16
  %v572 = add i64 %v477, %v571
  %v573 = add i64 %v572, %v479
  %v575 = addrspacecast ptr addrspace(3) @__shared_mem_68 to ptr
  %v576 = mul i64 %v573, 256
  %v577 = mul i64 %v532, 2
  %v578 = add i64 %v576, %v577
  %v579 = add i64 %v578, %v485
  %v580 = getelementptr inbounds i8, ptr %v575, i64 %v579
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v580, i32 %v564, i32 %v569) #0
  br label %bb191
bb191:
  %v581 = extractvalue { [4 x float] } %v548, 0
  %v582 = extractvalue [4 x float] %v581, 2
  %v583 = extractvalue { [4 x float] } %v548, 0
  %v584 = extractvalue [4 x float] %v583, 3
  %v585 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v582, float %v584)
  br label %bb192
bb192:
  %v586 = extractvalue { [4 x float] } %v559, 0
  %v587 = extractvalue [4 x float] %v586, 2
  %v588 = extractvalue { [4 x float] } %v559, 0
  %v589 = extractvalue [4 x float] %v588, 3
  %v590 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v587, float %v589)
  br label %bb193
bb193:
  %v591 = zext i32 %v523 to i64
  %v592 = mul i64 %v591, 16
  %v593 = add i64 %v477, %v592
  %v594 = add i64 %v593, 8
  %v595 = add i64 %v594, %v479
  %v596 = mul i64 %v595, 256
  %v597 = add i64 %v596, %v577
  %v598 = add i64 %v597, %v485
  %v599 = getelementptr inbounds i8, ptr %v575, i64 %v598
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v599, i32 %v585, i32 %v590) #0
  br label %bb194
bb194:
  %v600 = add i32 %v528, 1
  br label %bb183
bb195:
  %v601 = add i32 %v523, 1
  br label %bb181
bb196:
  %v602 = udiv i32 %v18, 2
  %v603 = zext i32 %v602 to i64
  %v604 = mul i32 %v502, 256
  %v605 = mul i32 %v673, 128
  %v606 = add i32 %v604, %v605
  %v607 = zext i32 %v606 to i64
  %v608 = mul i32 %v506, 64
  %v609 = zext i32 %v608 to i64
  %v610 = zext i32 %v671 to i64
  %v611 = mul i64 %v610, 32
  %v612 = zext i32 %v672 to i64
  br label %bb197
bb197:
  %v613 = phi i64 [ %v612, %bb196 ], [ %v632, %bb199 ]
  %v614 = icmp ult i64 %v613, 2048
  %v615 = xor i1 %v614, 1
  br i1 %v615, label %bb200, label %bb198
bb198:
  %v616 = udiv i64 %v613, 64
  %v617 = urem i64 %v613, 64
  %v618 = add i64 %v611, %v616
  %v619 = mul i64 %v618, 64
  %v620 = add i64 %v619, %v617
  %v621 = add i64 %v607, %v611
  %v622 = add i64 %v621, %v616
  %v623 = add i64 %v609, %v617
  %v624 = mul i64 %v622, %v603
  %v625 = add i64 %v624, %v623
  %v627 = bitcast ptr addrspace(3) @__shared_mem_68 to ptr addrspace(3)
  %v628 = getelementptr inbounds i32, ptr addrspace(3) %v627, i64 %v620
  br label %bb199
bb199:
  %v629 = load i32, ptr addrspace(3) %v628
  %v630 = extractvalue { ptr, i64 } %v12, 0
  %v631 = getelementptr inbounds i32, ptr %v630, i64 %v625
  store i32 %v629, ptr %v631
  %v632 = add i64 %v613, 32
  br label %bb197
bb200:
  %v633 = xor i1 %v80, 1
  br i1 %v633, label %bb202, label %bb201
bb201:
  %v634 = xor i1 %v511, 1
  br i1 %v634, label %bb205, label %bb203
bb202:
  %v635 = xor i1 %v511, 1
  br i1 %v635, label %bb208, label %bb207
bb203:
  %v636 = bitcast ptr addrspace(3) @__shared_mem_53 to ptr addrspace(3)
  %v637 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v636) #0
  br label %bb204
bb204:
  br label %bb209
bb205:
  %v638 = bitcast ptr addrspace(3) @__shared_mem_54 to ptr addrspace(3)
  %v639 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v638) #0
  br label %bb206
bb206:
  br label %bb209
bb207:
  call void asm sideeffect "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];", "l,~{memory}"(i64 %v470) #0
  br label %bb209
bb208:
  call void asm sideeffect "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];", "l,~{memory}"(i64 %v475) #0
  br label %bb209
bb209:
  %v640 = add i32 %v486, 1
  br label %bb166
bb210:
  call void asm sideeffect "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}"() #0
  br label %bb211
bb211:
  %v641 = xor i1 %v71, 1
  br i1 %v641, label %bb213, label %bb212
bb212:
  call void asm sideeffect "tcgen05.dealloc.cta_group::2.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v79, i32 512) #0
  br label %bb213
bb213:
  %v642 = xor i1 %v26, 1
  br i1 %v642, label %bb228, label %bb214
bb214:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_43) #0
  br label %bb215
bb215:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_44) #0
  br label %bb216
bb216:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_45) #0
  br label %bb217
bb217:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_46) #0
  br label %bb218
bb218:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_47) #0
  br label %bb219
bb219:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_48) #0
  br label %bb220
bb220:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_49) #0
  br label %bb221
bb221:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_50) #0
  br label %bb222
bb222:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_51) #0
  br label %bb223
bb223:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_52) #0
  br label %bb224
bb224:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_53) #0
  br label %bb225
bb225:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_54) #0
  br label %bb226
bb226:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_55) #0
  br label %bb227
bb227:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_56) #0
  br label %bb228
bb228:
  ret void
bb229:
  %v671 = udiv i32 %v22, 32
  %v672 = urem i32 %v20, 32
  %v673 = call i32 asm sideeffect "mov.u32 $0, %cluster_ctaid.x;", "=r"() #0
  br label %bb4
bb230:
  unreachable
}

define ptx_kernel void @gemm_sol_tiled(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi ptr [ %v0, %entry ]
  %v9 = phi ptr [ %v1, %entry ]
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = phi i32 [ %v5, %entry ]
  %v13 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  %v14 = bitcast i32 %v11 to i32
  %v15 = bitcast i32 %v12 to i32
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb76
bb2:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb3
bb3:
  %v19 = xor i1 %v278, 1
  br i1 %v19, label %bb7, label %bb4
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_69, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_70, i32 1) #0
  br label %bb6
bb6:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb7
bb7:
  call void @llvm.nvvm.barrier0() #0
  br label %bb8
bb8:
  %v25 = icmp eq i32 %v276, 0
  %v26 = icmp eq i32 %v276, 0
  br i1 %v26, label %bb9, label %bb11
bb9:
  %v28 = addrspacecast ptr addrspace(3) @__shared_mem_71 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v28, i32 512) #0
  br label %bb10
bb10:
  br label %bb11
bb11:
  call void @llvm.nvvm.barrier0() #0
  br label %bb12
bb12:
  %v31 = bitcast ptr addrspace(3) @__shared_mem_71 to ptr addrspace(3)
  %v32 = addrspacecast ptr addrspace(3) %v31 to ptr
  %v33 = load i32, ptr %v32
  %v34 = insertvalue { i8 } undef, i8 1, 0
  %v35 = insertvalue { i8 } undef, i8 0, 0
  %v36 = insertvalue { i8 } undef, i8 0, 0
  %v37 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v38 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v37, i32 128, 1
  %v39 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v38, i8 0, 2
  %v40 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v39, { i8 } %v35, 3
  %v41 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v40, { i8 } %v36, 4
  %v42 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v41, i1 0, 5
  %v43 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v42, { i8 } %v34, 6
  %v44 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v43, i1 0, 7
  %v45 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v44, i1 0, 8
  %v46 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v45, i1 0, 9
  %v47 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v46, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v47, ptr %v13
  %v48 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 1
  store i32 128, ptr %v48
  %v49 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 0
  store i32 128, ptr %v49
  %v50 = insertvalue { i8 } undef, i8 0, 0
  %v51 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 3
  store { i8 } %v50, ptr %v51
  %v52 = insertvalue { i8 } undef, i8 0, 0
  %v53 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 4
  store { i8 } %v52, ptr %v53
  %v54 = insertvalue { i8 } undef, i8 1, 0
  %v55 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 6
  store { i8 } %v54, ptr %v55
  %v56 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13
  %v57 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 0
  %v58 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 1
  %v59 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 2
  %v60 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 3
  %v61 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 4
  %v62 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 5
  %v63 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 6
  %v64 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 7
  %v65 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 8
  %v66 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 9
  %v67 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 10
  %v68 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v57, i32 %v58, i8 %v59, { i8 } %v60, { i8 } %v61, i1 %v62, { i8 } %v63, i1 %v64, i1 %v65, i1 %v66, i1 %v67)
  br label %bb13
bb13:
  %v69 = extractvalue { i32 } %v68, 0
  %v70 = udiv i32 %v15, 64
  br label %bb14
bb14:
  %v71 = phi i32 [ 0, %bb13 ], [ %v154, %bb41 ]
  %v72 = icmp ult i32 %v71, %v70
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb42, label %bb15
bb15:
  %v74 = and i32 %v71, 1
  %v75 = xor i1 %v278, 1
  br i1 %v75, label %bb23, label %bb16
bb16:
  %v76 = mul i32 %v71, 64
  %v77 = bitcast i32 %v76 to i32
  %v78 = mul i32 %v279, 128
  %v79 = bitcast i32 %v78 to i32
  %v80 = mul i32 %v18, 128
  %v81 = bitcast i32 %v80 to i32
  %v83 = addrspacecast ptr addrspace(3) @__shared_mem_72 to ptr
  %v85 = addrspacecast ptr addrspace(3) @__shared_mem_73 to ptr
  br label %bb17
bb17:
  %v86 = phi i32 [ 0, %bb16 ], [ %v102, %bb20 ]
  %v87 = icmp ult i32 %v86, 8
  %v88 = xor i1 %v87, 1
  br i1 %v88, label %bb21, label %bb18
bb18:
  %v89 = mul i32 %v86, 8
  %v90 = bitcast i32 %v89 to i32
  %v91 = add i32 %v77, %v90
  %v92 = mul i32 %v86, 2048
  %v93 = zext i32 %v92 to i64
  %v94 = getelementptr inbounds i8, ptr %v83, i64 %v93
  %v96 = addrspacecast ptr %v94 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v96, ptr addrspace(3) @__shared_mem_69, ptr %v8, i32 %v91, i32 %v79, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb19
bb19:
  %v98 = getelementptr inbounds i8, ptr %v85, i64 %v93
  %v100 = addrspacecast ptr %v98 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v100, ptr addrspace(3) @__shared_mem_69, ptr %v9, i32 %v91, i32 %v81, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb20
bb20:
  %v102 = add i32 %v86, 1
  br label %bb17
bb21:
  %v104 = bitcast ptr addrspace(3) @__shared_mem_69 to ptr addrspace(3)
  %v105 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v104, i32 32768) #0
  br label %bb22
bb22:
  br label %bb23
bb23:
  %v107 = bitcast ptr addrspace(3) @__shared_mem_69 to ptr addrspace(3)
  %v108 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v107, i32 %v74) #0
  %v109 = trunc i32 %v108 to i1
  br label %bb24
bb24:
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb26, label %bb25
bb25:
  call void @llvm.nvvm.barrier0() #0
  br label %bb27
bb26:
  br label %bb23
bb27:
  %v112 = xor i1 %v278, 1
  br i1 %v112, label %bb37, label %bb28
bb28:
  %v114 = bitcast ptr addrspace(3) @__shared_mem_72 to ptr addrspace(3)
  %v115 = ptrtoint ptr addrspace(3) %v114 to i64
  %v117 = bitcast ptr addrspace(3) @__shared_mem_73 to ptr addrspace(3)
  %v118 = ptrtoint ptr addrspace(3) %v117 to i64
  br label %bb29
bb29:
  %v119 = phi i32 [ 0, %bb28 ], [ %v145, %bb34 ]
  %v120 = icmp ult i32 %v119, 4
  %v121 = xor i1 %v120, 1
  br i1 %v121, label %bb35, label %bb30
bb30:
  %v122 = mul i32 %v119, 2
  %v123 = mul i32 %v122, 2048
  %v124 = zext i32 %v123 to i64
  %v125 = add i64 %v115, %v124
  %v126 = zext i32 4 to i64
  %v127 = and i64 %v126, 63
  %v128 = lshr i64 %v125, %v127
  %v129 = and i64 %v128, 16383
  %v130 = or i64 %v129, 8388608
  %v131 = or i64 %v130, 34359738368
  %v132 = or i64 %v131, 70368744177664
  %v133 = add i64 %v118, %v124
  %v134 = zext i32 4 to i64
  %v135 = and i64 %v134, 63
  %v136 = lshr i64 %v133, %v135
  %v137 = and i64 %v136, 16383
  %v138 = or i64 %v137, 8388608
  %v139 = or i64 %v138, 34359738368
  %v140 = or i64 %v139, 70368744177664
  %v141 = icmp ugt i32 %v71, 0
  %v142 = xor i1 %v141, 1
  br i1 %v142, label %bb32, label %bb31
bb31:
  br label %bb33
bb32:
  %v143 = icmp ugt i32 %v119, 0
  br label %bb33
bb33:
  %v144 = phi i1 [ 1, %bb31 ], [ %v143, %bb32 ]
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v33, i64 %v132, i64 %v140, i32 %v69, i1 %v144) #0
  br label %bb34
bb34:
  %v145 = add i32 %v119, 1
  br label %bb29
bb35:
  %v147 = addrspacecast ptr addrspace(3) @__shared_mem_70 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v147) #0
  br label %bb36
bb36:
  br label %bb37
bb37:
  %v149 = bitcast ptr addrspace(3) @__shared_mem_70 to ptr addrspace(3)
  %v150 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v149, i32 %v74) #0
  %v151 = trunc i32 %v150 to i1
  br label %bb38
bb38:
  %v152 = xor i1 %v151, 1
  br i1 %v152, label %bb40, label %bb39
bb39:
  call void @llvm.nvvm.barrier0() #0
  br label %bb41
bb40:
  br label %bb37
bb41:
  %v154 = add i32 %v71, 1
  br label %bb14
bb42:
  %v155 = mul i32 %v276, 32
  %v156 = zext i32 %v155 to i64
  %v157 = urem i32 %v277, 8
  %v158 = zext i32 %v157 to i64
  %v159 = icmp uge i32 %v277, 8
  %v160 = xor i1 %v159, 1
  br i1 %v160, label %bb44, label %bb43
bb43:
  %v161 = icmp ult i32 %v277, 16
  br label %bb45
bb44:
  br label %bb45
bb45:
  %v162 = phi i1 [ %v161, %bb43 ], [ 0, %bb44 ]
  %v163 = xor i1 %v162, 1
  br i1 %v163, label %bb47, label %bb46
bb46:
  br label %bb48
bb47:
  br label %bb48
bb48:
  %v164 = phi i64 [ 16, %bb46 ], [ 0, %bb47 ]
  br label %bb49
bb49:
  %v165 = phi i32 [ 0, %bb48 ], [ %v242, %bb63 ]
  %v166 = icmp ult i32 %v165, 2
  %v167 = xor i1 %v166, 1
  br i1 %v167, label %bb64, label %bb50
bb50:
  %v168 = mul i32 %v165, 16
  %v169 = add i32 %v155, %v168
  br label %bb51
bb51:
  %v170 = phi i32 [ 0, %bb50 ], [ %v241, %bb62 ]
  %v171 = icmp ult i32 %v170, 8
  %v172 = xor i1 %v171, 1
  br i1 %v172, label %bb63, label %bb52
bb52:
  %v173 = mul i32 %v170, 16
  %v174 = zext i32 %v173 to i64
  %v175 = and i32 16, 31
  %v176 = shl i32 %v169, %v175
  %v177 = add i32 %v33, %v176
  %v178 = trunc i64 %v174 to i32
  %v179 = add i32 %v177, %v178
  %v180 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v179) #0
  %v181 = extractvalue { float, float, float, float } %v180, 0
  %v182 = extractvalue { float, float, float, float } %v180, 1
  %v183 = extractvalue { float, float, float, float } %v180, 2
  %v184 = extractvalue { float, float, float, float } %v180, 3
  %v185 = insertvalue [4 x float] undef, float %v181, 0
  %v186 = insertvalue [4 x float] %v185, float %v182, 1
  %v187 = insertvalue [4 x float] %v186, float %v183, 2
  %v188 = insertvalue [4 x float] %v187, float %v184, 3
  %v189 = insertvalue { [4 x float] } undef, [4 x float] %v188, 0
  br label %bb53
bb53:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb54
bb54:
  %v190 = add i32 %v179, 8
  %v191 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v190) #0
  %v192 = extractvalue { float, float, float, float } %v191, 0
  %v193 = extractvalue { float, float, float, float } %v191, 1
  %v194 = extractvalue { float, float, float, float } %v191, 2
  %v195 = extractvalue { float, float, float, float } %v191, 3
  %v196 = insertvalue [4 x float] undef, float %v192, 0
  %v197 = insertvalue [4 x float] %v196, float %v193, 1
  %v198 = insertvalue [4 x float] %v197, float %v194, 2
  %v199 = insertvalue [4 x float] %v198, float %v195, 3
  %v200 = insertvalue { [4 x float] } undef, [4 x float] %v199, 0
  br label %bb55
bb55:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb56
bb56:
  %v201 = extractvalue { [4 x float] } %v189, 0
  %v202 = extractvalue [4 x float] %v201, 0
  %v203 = extractvalue { [4 x float] } %v189, 0
  %v204 = extractvalue [4 x float] %v203, 1
  %v205 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v202, float %v204)
  br label %bb57
bb57:
  %v206 = extractvalue { [4 x float] } %v200, 0
  %v207 = extractvalue [4 x float] %v206, 0
  %v208 = extractvalue { [4 x float] } %v200, 0
  %v209 = extractvalue [4 x float] %v208, 1
  %v210 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v207, float %v209)
  br label %bb58
bb58:
  %v211 = zext i32 %v165 to i64
  %v212 = mul i64 %v211, 16
  %v213 = add i64 %v156, %v212
  %v214 = add i64 %v213, %v158
  %v216 = addrspacecast ptr addrspace(3) @__shared_mem_74 to ptr
  %v217 = mul i64 %v214, 256
  %v218 = mul i64 %v174, 2
  %v219 = add i64 %v217, %v218
  %v220 = add i64 %v219, %v164
  %v221 = getelementptr inbounds i8, ptr %v216, i64 %v220
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v221, i32 %v205, i32 %v210) #0
  br label %bb59
bb59:
  %v222 = extractvalue { [4 x float] } %v189, 0
  %v223 = extractvalue [4 x float] %v222, 2
  %v224 = extractvalue { [4 x float] } %v189, 0
  %v225 = extractvalue [4 x float] %v224, 3
  %v226 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v223, float %v225)
  br label %bb60
bb60:
  %v227 = extractvalue { [4 x float] } %v200, 0
  %v228 = extractvalue [4 x float] %v227, 2
  %v229 = extractvalue { [4 x float] } %v200, 0
  %v230 = extractvalue [4 x float] %v229, 3
  %v231 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v228, float %v230)
  br label %bb61
bb61:
  %v232 = zext i32 %v165 to i64
  %v233 = mul i64 %v232, 16
  %v234 = add i64 %v156, %v233
  %v235 = add i64 %v234, 8
  %v236 = add i64 %v235, %v158
  %v237 = mul i64 %v236, 256
  %v238 = add i64 %v237, %v218
  %v239 = add i64 %v238, %v164
  %v240 = getelementptr inbounds i8, ptr %v216, i64 %v239
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v240, i32 %v226, i32 %v231) #0
  br label %bb62
bb62:
  %v241 = add i32 %v170, 1
  br label %bb51
bb63:
  %v242 = add i32 %v165, 1
  br label %bb49
bb64:
  call void @llvm.nvvm.barrier0() #0
  br label %bb65
bb65:
  %v244 = udiv i32 %v14, 2
  %v245 = zext i32 %v244 to i64
  %v246 = mul i32 %v279, 128
  %v247 = zext i32 %v246 to i64
  %v248 = mul i32 %v18, 64
  %v249 = zext i32 %v248 to i64
  %v250 = zext i32 %v16 to i64
  br label %bb66
bb66:
  %v251 = phi i64 [ %v250, %bb65 ], [ %v268, %bb68 ]
  %v252 = icmp ult i64 %v251, 8192
  %v253 = xor i1 %v252, 1
  br i1 %v253, label %bb69, label %bb67
bb67:
  %v254 = zext i32 6 to i64
  %v255 = and i64 %v254, 63
  %v256 = lshr i64 %v251, %v255
  %v257 = and i64 %v251, 63
  %v258 = add i64 %v247, %v256
  %v259 = add i64 %v249, %v257
  %v260 = mul i64 %v258, %v245
  %v261 = add i64 %v260, %v259
  %v263 = bitcast ptr addrspace(3) @__shared_mem_74 to ptr addrspace(3)
  %v264 = getelementptr inbounds i32, ptr addrspace(3) %v263, i64 %v251
  br label %bb68
bb68:
  %v265 = load i32, ptr addrspace(3) %v264
  %v266 = extractvalue { ptr, i64 } %v10, 0
  %v267 = getelementptr inbounds i32, ptr %v266, i64 %v261
  store i32 %v265, ptr %v267
  %v268 = add i64 %v251, 128
  br label %bb66
bb69:
  call void @llvm.nvvm.barrier0() #0
  br label %bb70
bb70:
  %v270 = xor i1 %v25, 1
  br i1 %v270, label %bb72, label %bb71
bb71:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v33, i32 512) #0
  br label %bb72
bb72:
  %v271 = xor i1 %v278, 1
  br i1 %v271, label %bb75, label %bb73
bb73:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_69) #0
  br label %bb74
bb74:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_70) #0
  br label %bb75
bb75:
  ret void
bb76:
  %v276 = udiv i32 %v17, 32
  %v277 = urem i32 %v16, 32
  %v278 = icmp eq i32 %v16, 0
  %v279 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
}

define ptx_kernel void @gemm_sol_warp_spec(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi ptr [ %v0, %entry ]
  %v9 = phi ptr [ %v1, %entry ]
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = phi i32 [ %v5, %entry ]
  %v13 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  %v14 = bitcast i32 %v11 to i32
  %v15 = bitcast i32 %v12 to i32
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb124
bb2:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb3
bb3:
  %v19 = icmp eq i32 %v16, 0
  %v20 = icmp eq i32 %v16, 0
  br i1 %v20, label %bb4, label %bb10
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_75, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_76, i32 1) #0
  br label %bb6
bb6:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_77, i32 1) #0
  br label %bb7
bb7:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_78, i32 1) #0
  br label %bb8
bb8:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_79, i32 1) #0
  br label %bb9
bb9:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb10
bb10:
  call void @llvm.nvvm.barrier0() #0
  br label %bb11
bb11:
  %v32 = xor i1 %v19, 1
  br i1 %v32, label %bb15, label %bb12
bb12:
  %v34 = bitcast ptr addrspace(3) @__shared_mem_77 to ptr addrspace(3)
  %v35 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v34) #0
  br label %bb13
bb13:
  %v37 = bitcast ptr addrspace(3) @__shared_mem_78 to ptr addrspace(3)
  %v38 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v37) #0
  br label %bb14
bb14:
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier0() #0
  br label %bb16
bb16:
  %v40 = icmp eq i32 %v352, 0
  %v41 = icmp eq i32 %v352, 0
  br i1 %v41, label %bb17, label %bb19
bb17:
  %v43 = addrspacecast ptr addrspace(3) @__shared_mem_80 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v43, i32 512) #0
  br label %bb18
bb18:
  br label %bb19
bb19:
  call void @llvm.nvvm.barrier0() #0
  br label %bb20
bb20:
  %v46 = bitcast ptr addrspace(3) @__shared_mem_80 to ptr addrspace(3)
  %v47 = addrspacecast ptr addrspace(3) %v46 to ptr
  %v48 = load i32, ptr %v47
  %v49 = insertvalue { i8 } undef, i8 1, 0
  %v50 = insertvalue { i8 } undef, i8 0, 0
  %v51 = insertvalue { i8 } undef, i8 0, 0
  %v52 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v53 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v52, i32 128, 1
  %v54 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v53, i8 0, 2
  %v55 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v54, { i8 } %v50, 3
  %v56 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v55, { i8 } %v51, 4
  %v57 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, i1 0, 5
  %v58 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v57, { i8 } %v49, 6
  %v59 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v58, i1 0, 7
  %v60 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v59, i1 0, 8
  %v61 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v60, i1 0, 9
  %v62 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v61, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v62, ptr %v13
  %v63 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 1
  store i32 128, ptr %v63
  %v64 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 0
  store i32 128, ptr %v64
  %v65 = insertvalue { i8 } undef, i8 0, 0
  %v66 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 3
  store { i8 } %v65, ptr %v66
  %v67 = insertvalue { i8 } undef, i8 0, 0
  %v68 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 4
  store { i8 } %v67, ptr %v68
  %v69 = insertvalue { i8 } undef, i8 1, 0
  %v70 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 6
  store { i8 } %v69, ptr %v70
  %v71 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13
  %v72 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 0
  %v73 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 1
  %v74 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 2
  %v75 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 3
  %v76 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 4
  %v77 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 5
  %v78 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 6
  %v79 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 7
  %v80 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 8
  %v81 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 9
  %v82 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v71, 10
  %v83 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v72, i32 %v73, i8 %v74, { i8 } %v75, { i8 } %v76, i1 %v77, { i8 } %v78, i1 %v79, i1 %v80, i1 %v81, i1 %v82)
  br label %bb21
bb21:
  %v84 = extractvalue { i32 } %v83, 0
  %v85 = udiv i32 %v15, 64
  %v86 = mul i32 %v354, 128
  %v87 = bitcast i32 %v86 to i32
  %v88 = mul i32 %v18, 128
  %v89 = bitcast i32 %v88 to i32
  %v90 = icmp eq i32 %v352, 4
  br i1 %v90, label %bb22, label %bb45
bb22:
  %v91 = icmp eq i32 %v353, 0
  br label %bb23
bb23:
  %v92 = phi i32 [ 0, %bb22 ], [ %v141, %bb43 ]
  %v93 = icmp ult i32 %v92, %v85
  %v94 = xor i1 %v93, 1
  br i1 %v94, label %bb44, label %bb24
bb24:
  %v95 = and i32 %v92, 1
  %v96 = and i32 1, 31
  %v97 = lshr i32 %v92, %v96
  %v98 = and i32 %v97, 1
  %v99 = icmp eq i32 %v95, 0
  %v100 = icmp eq i32 %v95, 0
  br i1 %v100, label %bb25, label %bb29
bb25:
  %v102 = bitcast ptr addrspace(3) @__shared_mem_77 to ptr addrspace(3)
  %v103 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v102, i32 %v98) #0
  %v104 = trunc i32 %v103 to i1
  br label %bb26
bb26:
  %v105 = xor i1 %v104, 1
  br i1 %v105, label %bb28, label %bb27
bb27:
  br label %bb33
bb28:
  br label %bb25
bb29:
  %v107 = bitcast ptr addrspace(3) @__shared_mem_78 to ptr addrspace(3)
  %v108 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v107, i32 %v98) #0
  %v109 = trunc i32 %v108 to i1
  br label %bb30
bb30:
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb32, label %bb31
bb31:
  br label %bb33
bb32:
  br label %bb29
bb33:
  %v111 = xor i1 %v91, 1
  br i1 %v111, label %bb43, label %bb34
bb34:
  %v112 = mul i32 %v92, 64
  %v113 = bitcast i32 %v112 to i32
  %v114 = xor i1 %v99, 1
  br i1 %v114, label %bb39, label %bb35
bb35:
  %v116 = addrspacecast ptr addrspace(3) @__shared_mem_81 to ptr
  %v119 = addrspacecast ptr %v116 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v119, ptr addrspace(3) @__shared_mem_75, ptr %v8, i32 %v113, i32 %v87, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb36
bb36:
  %v122 = addrspacecast ptr addrspace(3) @__shared_mem_82 to ptr
  %v124 = addrspacecast ptr %v122 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v124, ptr addrspace(3) @__shared_mem_75, ptr %v9, i32 %v113, i32 %v89, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb37
bb37:
  %v126 = bitcast ptr addrspace(3) @__shared_mem_75 to ptr addrspace(3)
  %v127 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v126, i32 32768) #0
  br label %bb38
bb38:
  br label %bb43
bb39:
  %v129 = addrspacecast ptr addrspace(3) @__shared_mem_83 to ptr
  %v132 = addrspacecast ptr %v129 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v132, ptr addrspace(3) @__shared_mem_76, ptr %v8, i32 %v113, i32 %v87, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb40
bb40:
  %v135 = addrspacecast ptr addrspace(3) @__shared_mem_84 to ptr
  %v137 = addrspacecast ptr %v135 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v137, ptr addrspace(3) @__shared_mem_76, ptr %v9, i32 %v113, i32 %v89, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb41
bb41:
  %v139 = bitcast ptr addrspace(3) @__shared_mem_76 to ptr addrspace(3)
  %v140 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v139, i32 32768) #0
  br label %bb42
bb42:
  br label %bb43
bb43:
  %v141 = add i32 %v92, 1
  br label %bb23
bb44:
  br label %bb45
bb45:
  %v142 = icmp eq i32 %v352, 5
  br i1 %v142, label %bb46, label %bb82
bb46:
  %v143 = icmp eq i32 %v353, 0
  br label %bb47
bb47:
  %v144 = phi i32 [ 0, %bb46 ], [ %v218, %bb80 ]
  %v145 = icmp ult i32 %v144, %v85
  %v146 = xor i1 %v145, 1
  br i1 %v146, label %bb81, label %bb48
bb48:
  %v147 = and i32 %v144, 1
  %v148 = and i32 1, 31
  %v149 = lshr i32 %v144, %v148
  %v150 = and i32 %v149, 1
  %v151 = add i32 %v144, 1
  %v152 = icmp eq i32 %v151, %v85
  %v153 = icmp eq i32 %v147, 0
  %v154 = icmp eq i32 %v147, 0
  br i1 %v154, label %bb49, label %bb53
bb49:
  %v156 = bitcast ptr addrspace(3) @__shared_mem_75 to ptr addrspace(3)
  %v157 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v156, i32 %v150) #0
  %v158 = trunc i32 %v157 to i1
  br label %bb50
bb50:
  %v159 = xor i1 %v158, 1
  br i1 %v159, label %bb52, label %bb51
bb51:
  br label %bb57
bb52:
  br label %bb49
bb53:
  %v161 = bitcast ptr addrspace(3) @__shared_mem_76 to ptr addrspace(3)
  %v162 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v161, i32 %v150) #0
  %v163 = trunc i32 %v162 to i1
  br label %bb54
bb54:
  %v164 = xor i1 %v163, 1
  br i1 %v164, label %bb56, label %bb55
bb55:
  br label %bb57
bb56:
  br label %bb53
bb57:
  %v165 = xor i1 %v143, 1
  br i1 %v165, label %bb80, label %bb58
bb58:
  %v166 = xor i1 %v153, 1
  br i1 %v166, label %bb60, label %bb59
bb59:
  %v168 = bitcast ptr addrspace(3) @__shared_mem_81 to ptr addrspace(3)
  %v169 = ptrtoint ptr addrspace(3) %v168 to i64
  br label %bb61
bb60:
  %v171 = bitcast ptr addrspace(3) @__shared_mem_83 to ptr addrspace(3)
  %v172 = ptrtoint ptr addrspace(3) %v171 to i64
  br label %bb61
bb61:
  %v173 = phi i64 [ %v169, %bb59 ], [ %v172, %bb60 ]
  %v174 = xor i1 %v153, 1
  br i1 %v174, label %bb63, label %bb62
bb62:
  %v176 = bitcast ptr addrspace(3) @__shared_mem_82 to ptr addrspace(3)
  %v177 = ptrtoint ptr addrspace(3) %v176 to i64
  br label %bb64
bb63:
  %v179 = bitcast ptr addrspace(3) @__shared_mem_84 to ptr addrspace(3)
  %v180 = ptrtoint ptr addrspace(3) %v179 to i64
  br label %bb64
bb64:
  %v181 = phi i64 [ %v177, %bb62 ], [ %v180, %bb63 ]
  br label %bb65
bb65:
  %v182 = phi i32 [ 0, %bb64 ], [ %v209, %bb70 ]
  %v183 = icmp ult i32 %v182, 4
  %v184 = xor i1 %v183, 1
  br i1 %v184, label %bb71, label %bb66
bb66:
  %v185 = mul i32 %v182, 32
  %v186 = zext i32 %v185 to i64
  %v187 = add i64 %v173, %v186
  %v188 = zext i32 4 to i64
  %v189 = and i64 %v188, 63
  %v190 = lshr i64 %v187, %v189
  %v191 = and i64 %v190, 16383
  %v192 = or i64 %v191, 65536
  %v193 = or i64 %v192, 274877906944
  %v194 = or i64 %v193, 70368744177664
  %v195 = or i64 %v194, 4611686018427387904
  %v196 = add i64 %v181, %v186
  %v197 = zext i32 4 to i64
  %v198 = and i64 %v197, 63
  %v199 = lshr i64 %v196, %v198
  %v200 = and i64 %v199, 16383
  %v201 = or i64 %v200, 65536
  %v202 = or i64 %v201, 274877906944
  %v203 = or i64 %v202, 70368744177664
  %v204 = or i64 %v203, 4611686018427387904
  %v205 = icmp ugt i32 %v144, 0
  %v206 = xor i1 %v205, 1
  br i1 %v206, label %bb68, label %bb67
bb67:
  br label %bb69
bb68:
  %v207 = icmp ugt i32 %v182, 0
  br label %bb69
bb69:
  %v208 = phi i1 [ 1, %bb67 ], [ %v207, %bb68 ]
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v48, i64 %v195, i64 %v204, i32 %v84, i1 %v208) #0
  br label %bb70
bb70:
  %v209 = add i32 %v182, 1
  br label %bb65
bb71:
  %v210 = xor i1 %v152, 1
  br i1 %v210, label %bb73, label %bb72
bb72:
  %v212 = addrspacecast ptr addrspace(3) @__shared_mem_79 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v212) #0
  br label %bb74
bb73:
  %v213 = xor i1 %v153, 1
  br i1 %v213, label %bb76, label %bb75
bb74:
  br label %bb79
bb75:
  %v215 = addrspacecast ptr addrspace(3) @__shared_mem_77 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v215) #0
  br label %bb77
bb76:
  %v217 = addrspacecast ptr addrspace(3) @__shared_mem_78 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v217) #0
  br label %bb78
bb77:
  br label %bb79
bb78:
  br label %bb79
bb79:
  br label %bb80
bb80:
  %v218 = add i32 %v144, 1
  br label %bb47
bb81:
  br label %bb82
bb82:
  %v219 = icmp ult i32 %v352, 4
  %v220 = xor i1 %v219, 1
  br i1 %v220, label %bb109, label %bb83
bb83:
  %v222 = bitcast ptr addrspace(3) @__shared_mem_79 to ptr addrspace(3)
  %v223 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v222, i32 0) #0
  %v224 = trunc i32 %v223 to i1
  br label %bb84
bb84:
  %v225 = xor i1 %v224, 1
  br i1 %v225, label %bb86, label %bb85
bb85:
  %v226 = mul i32 %v352, 32
  %v227 = zext i32 %v226 to i64
  %v228 = urem i32 %v353, 8
  %v229 = zext i32 %v228 to i64
  %v230 = icmp uge i32 %v353, 8
  %v231 = xor i1 %v230, 1
  br i1 %v231, label %bb88, label %bb87
bb86:
  br label %bb83
bb87:
  %v232 = icmp ult i32 %v353, 16
  br label %bb89
bb88:
  br label %bb89
bb89:
  %v233 = phi i1 [ %v232, %bb87 ], [ 0, %bb88 ]
  %v234 = xor i1 %v233, 1
  br i1 %v234, label %bb91, label %bb90
bb90:
  br label %bb92
bb91:
  br label %bb92
bb92:
  %v235 = phi i64 [ 16, %bb90 ], [ 0, %bb91 ]
  br label %bb93
bb93:
  %v236 = phi i32 [ 0, %bb92 ], [ %v313, %bb107 ]
  %v237 = icmp ult i32 %v236, 2
  %v238 = xor i1 %v237, 1
  br i1 %v238, label %bb108, label %bb94
bb94:
  %v239 = mul i32 %v236, 16
  %v240 = add i32 %v226, %v239
  br label %bb95
bb95:
  %v241 = phi i32 [ 0, %bb94 ], [ %v312, %bb106 ]
  %v242 = icmp ult i32 %v241, 8
  %v243 = xor i1 %v242, 1
  br i1 %v243, label %bb107, label %bb96
bb96:
  %v244 = mul i32 %v241, 16
  %v245 = zext i32 %v244 to i64
  %v246 = and i32 16, 31
  %v247 = shl i32 %v240, %v246
  %v248 = add i32 %v48, %v247
  %v249 = trunc i64 %v245 to i32
  %v250 = add i32 %v248, %v249
  %v251 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v250) #0
  %v252 = extractvalue { float, float, float, float } %v251, 0
  %v253 = extractvalue { float, float, float, float } %v251, 1
  %v254 = extractvalue { float, float, float, float } %v251, 2
  %v255 = extractvalue { float, float, float, float } %v251, 3
  %v256 = insertvalue [4 x float] undef, float %v252, 0
  %v257 = insertvalue [4 x float] %v256, float %v253, 1
  %v258 = insertvalue [4 x float] %v257, float %v254, 2
  %v259 = insertvalue [4 x float] %v258, float %v255, 3
  %v260 = insertvalue { [4 x float] } undef, [4 x float] %v259, 0
  br label %bb97
bb97:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb98
bb98:
  %v261 = add i32 %v250, 8
  %v262 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v261) #0
  %v263 = extractvalue { float, float, float, float } %v262, 0
  %v264 = extractvalue { float, float, float, float } %v262, 1
  %v265 = extractvalue { float, float, float, float } %v262, 2
  %v266 = extractvalue { float, float, float, float } %v262, 3
  %v267 = insertvalue [4 x float] undef, float %v263, 0
  %v268 = insertvalue [4 x float] %v267, float %v264, 1
  %v269 = insertvalue [4 x float] %v268, float %v265, 2
  %v270 = insertvalue [4 x float] %v269, float %v266, 3
  %v271 = insertvalue { [4 x float] } undef, [4 x float] %v270, 0
  br label %bb99
bb99:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb100
bb100:
  %v272 = extractvalue { [4 x float] } %v260, 0
  %v273 = extractvalue [4 x float] %v272, 0
  %v274 = extractvalue { [4 x float] } %v260, 0
  %v275 = extractvalue [4 x float] %v274, 1
  %v276 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v273, float %v275)
  br label %bb101
bb101:
  %v277 = extractvalue { [4 x float] } %v271, 0
  %v278 = extractvalue [4 x float] %v277, 0
  %v279 = extractvalue { [4 x float] } %v271, 0
  %v280 = extractvalue [4 x float] %v279, 1
  %v281 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v278, float %v280)
  br label %bb102
bb102:
  %v282 = zext i32 %v236 to i64
  %v283 = mul i64 %v282, 16
  %v284 = add i64 %v227, %v283
  %v285 = add i64 %v284, %v229
  %v287 = addrspacecast ptr addrspace(3) @__shared_mem_85 to ptr
  %v288 = mul i64 %v285, 256
  %v289 = mul i64 %v245, 2
  %v290 = add i64 %v288, %v289
  %v291 = add i64 %v290, %v235
  %v292 = getelementptr inbounds i8, ptr %v287, i64 %v291
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v292, i32 %v276, i32 %v281) #0
  br label %bb103
bb103:
  %v293 = extractvalue { [4 x float] } %v260, 0
  %v294 = extractvalue [4 x float] %v293, 2
  %v295 = extractvalue { [4 x float] } %v260, 0
  %v296 = extractvalue [4 x float] %v295, 3
  %v297 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v294, float %v296)
  br label %bb104
bb104:
  %v298 = extractvalue { [4 x float] } %v271, 0
  %v299 = extractvalue [4 x float] %v298, 2
  %v300 = extractvalue { [4 x float] } %v271, 0
  %v301 = extractvalue [4 x float] %v300, 3
  %v302 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v299, float %v301)
  br label %bb105
bb105:
  %v303 = zext i32 %v236 to i64
  %v304 = mul i64 %v303, 16
  %v305 = add i64 %v227, %v304
  %v306 = add i64 %v305, 8
  %v307 = add i64 %v306, %v229
  %v308 = mul i64 %v307, 256
  %v309 = add i64 %v308, %v289
  %v310 = add i64 %v309, %v235
  %v311 = getelementptr inbounds i8, ptr %v287, i64 %v310
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v311, i32 %v297, i32 %v302) #0
  br label %bb106
bb106:
  %v312 = add i32 %v241, 1
  br label %bb95
bb107:
  %v313 = add i32 %v236, 1
  br label %bb93
bb108:
  br label %bb109
bb109:
  call void @llvm.nvvm.barrier0() #0
  br label %bb110
bb110:
  %v315 = udiv i32 %v14, 2
  %v316 = zext i32 %v315 to i64
  %v317 = zext i32 %v86 to i64
  %v318 = mul i32 %v18, 64
  %v319 = zext i32 %v318 to i64
  %v320 = zext i32 %v16 to i64
  br label %bb111
bb111:
  %v321 = phi i64 [ %v320, %bb110 ], [ %v338, %bb113 ]
  %v322 = icmp ult i64 %v321, 8192
  %v323 = xor i1 %v322, 1
  br i1 %v323, label %bb114, label %bb112
bb112:
  %v324 = zext i32 6 to i64
  %v325 = and i64 %v324, 63
  %v326 = lshr i64 %v321, %v325
  %v327 = and i64 %v321, 63
  %v328 = add i64 %v317, %v326
  %v329 = add i64 %v319, %v327
  %v330 = mul i64 %v328, %v316
  %v331 = add i64 %v330, %v329
  %v333 = bitcast ptr addrspace(3) @__shared_mem_85 to ptr addrspace(3)
  %v334 = getelementptr inbounds i32, ptr addrspace(3) %v333, i64 %v321
  br label %bb113
bb113:
  %v335 = load i32, ptr addrspace(3) %v334
  %v336 = extractvalue { ptr, i64 } %v10, 0
  %v337 = getelementptr inbounds i32, ptr %v336, i64 %v331
  store i32 %v335, ptr %v337
  %v338 = add i64 %v321, 192
  br label %bb111
bb114:
  call void @llvm.nvvm.barrier0() #0
  br label %bb115
bb115:
  %v340 = xor i1 %v40, 1
  br i1 %v340, label %bb117, label %bb116
bb116:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v48, i32 512) #0
  br label %bb117
bb117:
  %v341 = xor i1 %v19, 1
  br i1 %v341, label %bb123, label %bb118
bb118:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_75) #0
  br label %bb119
bb119:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_76) #0
  br label %bb120
bb120:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_77) #0
  br label %bb121
bb121:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_78) #0
  br label %bb122
bb122:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_79) #0
  br label %bb123
bb123:
  ret void
bb124:
  %v352 = udiv i32 %v17, 32
  %v353 = urem i32 %v16, 32
  %v354 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
}

define ptx_kernel void @gemm_sol_swizzled(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v3, 1
  br label %bb0
bb0:
  %v8 = phi ptr [ %v0, %entry ]
  %v9 = phi ptr [ %v1, %entry ]
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = phi i32 [ %v5, %entry ]
  %v13 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  %v14 = bitcast i32 %v11 to i32
  %v15 = bitcast i32 %v12 to i32
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb73
bb2:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb3
bb3:
  %v19 = xor i1 %v268, 1
  br i1 %v19, label %bb7, label %bb4
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_86, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_87, i32 1) #0
  br label %bb6
bb6:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb7
bb7:
  call void @llvm.nvvm.barrier0() #0
  br label %bb8
bb8:
  %v25 = icmp eq i32 %v266, 0
  %v26 = icmp eq i32 %v266, 0
  br i1 %v26, label %bb9, label %bb11
bb9:
  %v28 = addrspacecast ptr addrspace(3) @__shared_mem_88 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v28, i32 512) #0
  br label %bb10
bb10:
  br label %bb11
bb11:
  call void @llvm.nvvm.barrier0() #0
  br label %bb12
bb12:
  %v31 = bitcast ptr addrspace(3) @__shared_mem_88 to ptr addrspace(3)
  %v32 = addrspacecast ptr addrspace(3) %v31 to ptr
  %v33 = load i32, ptr %v32
  %v34 = insertvalue { i8 } undef, i8 1, 0
  %v35 = insertvalue { i8 } undef, i8 0, 0
  %v36 = insertvalue { i8 } undef, i8 0, 0
  %v37 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v38 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v37, i32 128, 1
  %v39 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v38, i8 0, 2
  %v40 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v39, { i8 } %v35, 3
  %v41 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v40, { i8 } %v36, 4
  %v42 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v41, i1 0, 5
  %v43 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v42, { i8 } %v34, 6
  %v44 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v43, i1 0, 7
  %v45 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v44, i1 0, 8
  %v46 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v45, i1 0, 9
  %v47 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v46, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v47, ptr %v13
  %v48 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 1
  store i32 128, ptr %v48
  %v49 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 0
  store i32 128, ptr %v49
  %v50 = insertvalue { i8 } undef, i8 0, 0
  %v51 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 3
  store { i8 } %v50, ptr %v51
  %v52 = insertvalue { i8 } undef, i8 0, 0
  %v53 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 4
  store { i8 } %v52, ptr %v53
  %v54 = insertvalue { i8 } undef, i8 1, 0
  %v55 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13, i32 0, i32 6
  store { i8 } %v54, ptr %v55
  %v56 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v13
  %v57 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 0
  %v58 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 1
  %v59 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 2
  %v60 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 3
  %v61 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 4
  %v62 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 5
  %v63 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 6
  %v64 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 7
  %v65 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 8
  %v66 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 9
  %v67 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v56, 10
  %v68 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v57, i32 %v58, i8 %v59, { i8 } %v60, { i8 } %v61, i1 %v62, { i8 } %v63, i1 %v64, i1 %v65, i1 %v66, i1 %v67)
  br label %bb13
bb13:
  %v69 = extractvalue { i32 } %v68, 0
  %v70 = udiv i32 %v15, 64
  br label %bb14
bb14:
  %v71 = phi i32 [ 0, %bb13 ], [ %v144, %bb38 ]
  %v72 = icmp ult i32 %v71, %v70
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb39, label %bb15
bb15:
  %v74 = and i32 %v71, 1
  %v75 = xor i1 %v268, 1
  br i1 %v75, label %bb20, label %bb16
bb16:
  %v76 = mul i32 %v71, 64
  %v77 = bitcast i32 %v76 to i32
  %v78 = mul i32 %v269, 128
  %v79 = bitcast i32 %v78 to i32
  %v80 = mul i32 %v18, 128
  %v81 = bitcast i32 %v80 to i32
  %v83 = addrspacecast ptr addrspace(3) @__shared_mem_89 to ptr
  %v85 = addrspacecast ptr addrspace(3) @__shared_mem_90 to ptr
  %v88 = addrspacecast ptr %v83 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v88, ptr addrspace(3) @__shared_mem_86, ptr %v8, i32 %v77, i32 %v79, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb17
bb17:
  %v91 = addrspacecast ptr %v85 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v91, ptr addrspace(3) @__shared_mem_86, ptr %v9, i32 %v77, i32 %v81, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb18
bb18:
  %v93 = bitcast ptr addrspace(3) @__shared_mem_86 to ptr addrspace(3)
  %v94 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v93, i32 32768) #0
  br label %bb19
bb19:
  br label %bb20
bb20:
  %v96 = bitcast ptr addrspace(3) @__shared_mem_86 to ptr addrspace(3)
  %v97 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v96, i32 %v74) #0
  %v98 = trunc i32 %v97 to i1
  br label %bb21
bb21:
  %v99 = xor i1 %v98, 1
  br i1 %v99, label %bb23, label %bb22
bb22:
  call void @llvm.nvvm.barrier0() #0
  br label %bb24
bb23:
  br label %bb20
bb24:
  %v101 = xor i1 %v268, 1
  br i1 %v101, label %bb34, label %bb25
bb25:
  %v103 = bitcast ptr addrspace(3) @__shared_mem_89 to ptr addrspace(3)
  %v104 = ptrtoint ptr addrspace(3) %v103 to i64
  %v106 = bitcast ptr addrspace(3) @__shared_mem_90 to ptr addrspace(3)
  %v107 = ptrtoint ptr addrspace(3) %v106 to i64
  br label %bb26
bb26:
  %v108 = phi i32 [ 0, %bb25 ], [ %v135, %bb31 ]
  %v109 = icmp ult i32 %v108, 4
  %v110 = xor i1 %v109, 1
  br i1 %v110, label %bb32, label %bb27
bb27:
  %v111 = mul i32 %v108, 32
  %v112 = zext i32 %v111 to i64
  %v113 = add i64 %v104, %v112
  %v114 = zext i32 4 to i64
  %v115 = and i64 %v114, 63
  %v116 = lshr i64 %v113, %v115
  %v117 = and i64 %v116, 16383
  %v118 = or i64 %v117, 65536
  %v119 = or i64 %v118, 274877906944
  %v120 = or i64 %v119, 70368744177664
  %v121 = or i64 %v120, 4611686018427387904
  %v122 = add i64 %v107, %v112
  %v123 = zext i32 4 to i64
  %v124 = and i64 %v123, 63
  %v125 = lshr i64 %v122, %v124
  %v126 = and i64 %v125, 16383
  %v127 = or i64 %v126, 65536
  %v128 = or i64 %v127, 274877906944
  %v129 = or i64 %v128, 70368744177664
  %v130 = or i64 %v129, 4611686018427387904
  %v131 = icmp ugt i32 %v71, 0
  %v132 = xor i1 %v131, 1
  br i1 %v132, label %bb29, label %bb28
bb28:
  br label %bb30
bb29:
  %v133 = icmp ugt i32 %v108, 0
  br label %bb30
bb30:
  %v134 = phi i1 [ 1, %bb28 ], [ %v133, %bb29 ]
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v33, i64 %v121, i64 %v130, i32 %v69, i1 %v134) #0
  br label %bb31
bb31:
  %v135 = add i32 %v108, 1
  br label %bb26
bb32:
  %v137 = addrspacecast ptr addrspace(3) @__shared_mem_87 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v137) #0
  br label %bb33
bb33:
  br label %bb34
bb34:
  %v139 = bitcast ptr addrspace(3) @__shared_mem_87 to ptr addrspace(3)
  %v140 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v139, i32 %v74) #0
  %v141 = trunc i32 %v140 to i1
  br label %bb35
bb35:
  %v142 = xor i1 %v141, 1
  br i1 %v142, label %bb37, label %bb36
bb36:
  call void @llvm.nvvm.barrier0() #0
  br label %bb38
bb37:
  br label %bb34
bb38:
  %v144 = add i32 %v71, 1
  br label %bb14
bb39:
  %v145 = mul i32 %v266, 32
  %v146 = zext i32 %v145 to i64
  %v147 = urem i32 %v267, 8
  %v148 = zext i32 %v147 to i64
  %v149 = icmp uge i32 %v267, 8
  %v150 = xor i1 %v149, 1
  br i1 %v150, label %bb41, label %bb40
bb40:
  %v151 = icmp ult i32 %v267, 16
  br label %bb42
bb41:
  br label %bb42
bb42:
  %v152 = phi i1 [ %v151, %bb40 ], [ 0, %bb41 ]
  %v153 = xor i1 %v152, 1
  br i1 %v153, label %bb44, label %bb43
bb43:
  br label %bb45
bb44:
  br label %bb45
bb45:
  %v154 = phi i64 [ 16, %bb43 ], [ 0, %bb44 ]
  br label %bb46
bb46:
  %v155 = phi i32 [ 0, %bb45 ], [ %v232, %bb60 ]
  %v156 = icmp ult i32 %v155, 2
  %v157 = xor i1 %v156, 1
  br i1 %v157, label %bb61, label %bb47
bb47:
  %v158 = mul i32 %v155, 16
  %v159 = add i32 %v145, %v158
  br label %bb48
bb48:
  %v160 = phi i32 [ 0, %bb47 ], [ %v231, %bb59 ]
  %v161 = icmp ult i32 %v160, 8
  %v162 = xor i1 %v161, 1
  br i1 %v162, label %bb60, label %bb49
bb49:
  %v163 = mul i32 %v160, 16
  %v164 = zext i32 %v163 to i64
  %v165 = and i32 16, 31
  %v166 = shl i32 %v159, %v165
  %v167 = add i32 %v33, %v166
  %v168 = trunc i64 %v164 to i32
  %v169 = add i32 %v167, %v168
  %v170 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v169) #0
  %v171 = extractvalue { float, float, float, float } %v170, 0
  %v172 = extractvalue { float, float, float, float } %v170, 1
  %v173 = extractvalue { float, float, float, float } %v170, 2
  %v174 = extractvalue { float, float, float, float } %v170, 3
  %v175 = insertvalue [4 x float] undef, float %v171, 0
  %v176 = insertvalue [4 x float] %v175, float %v172, 1
  %v177 = insertvalue [4 x float] %v176, float %v173, 2
  %v178 = insertvalue [4 x float] %v177, float %v174, 3
  %v179 = insertvalue { [4 x float] } undef, [4 x float] %v178, 0
  br label %bb50
bb50:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb51
bb51:
  %v180 = add i32 %v169, 8
  %v181 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v180) #0
  %v182 = extractvalue { float, float, float, float } %v181, 0
  %v183 = extractvalue { float, float, float, float } %v181, 1
  %v184 = extractvalue { float, float, float, float } %v181, 2
  %v185 = extractvalue { float, float, float, float } %v181, 3
  %v186 = insertvalue [4 x float] undef, float %v182, 0
  %v187 = insertvalue [4 x float] %v186, float %v183, 1
  %v188 = insertvalue [4 x float] %v187, float %v184, 2
  %v189 = insertvalue [4 x float] %v188, float %v185, 3
  %v190 = insertvalue { [4 x float] } undef, [4 x float] %v189, 0
  br label %bb52
bb52:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb53
bb53:
  %v191 = extractvalue { [4 x float] } %v179, 0
  %v192 = extractvalue [4 x float] %v191, 0
  %v193 = extractvalue { [4 x float] } %v179, 0
  %v194 = extractvalue [4 x float] %v193, 1
  %v195 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v192, float %v194)
  br label %bb54
bb54:
  %v196 = extractvalue { [4 x float] } %v190, 0
  %v197 = extractvalue [4 x float] %v196, 0
  %v198 = extractvalue { [4 x float] } %v190, 0
  %v199 = extractvalue [4 x float] %v198, 1
  %v200 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v197, float %v199)
  br label %bb55
bb55:
  %v201 = zext i32 %v155 to i64
  %v202 = mul i64 %v201, 16
  %v203 = add i64 %v146, %v202
  %v204 = add i64 %v203, %v148
  %v206 = addrspacecast ptr addrspace(3) @__shared_mem_91 to ptr
  %v207 = mul i64 %v204, 256
  %v208 = mul i64 %v164, 2
  %v209 = add i64 %v207, %v208
  %v210 = add i64 %v209, %v154
  %v211 = getelementptr inbounds i8, ptr %v206, i64 %v210
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v211, i32 %v195, i32 %v200) #0
  br label %bb56
bb56:
  %v212 = extractvalue { [4 x float] } %v179, 0
  %v213 = extractvalue [4 x float] %v212, 2
  %v214 = extractvalue { [4 x float] } %v179, 0
  %v215 = extractvalue [4 x float] %v214, 3
  %v216 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v213, float %v215)
  br label %bb57
bb57:
  %v217 = extractvalue { [4 x float] } %v190, 0
  %v218 = extractvalue [4 x float] %v217, 2
  %v219 = extractvalue { [4 x float] } %v190, 0
  %v220 = extractvalue [4 x float] %v219, 3
  %v221 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v218, float %v220)
  br label %bb58
bb58:
  %v222 = zext i32 %v155 to i64
  %v223 = mul i64 %v222, 16
  %v224 = add i64 %v146, %v223
  %v225 = add i64 %v224, 8
  %v226 = add i64 %v225, %v148
  %v227 = mul i64 %v226, 256
  %v228 = add i64 %v227, %v208
  %v229 = add i64 %v228, %v154
  %v230 = getelementptr inbounds i8, ptr %v206, i64 %v229
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v230, i32 %v216, i32 %v221) #0
  br label %bb59
bb59:
  %v231 = add i32 %v160, 1
  br label %bb48
bb60:
  %v232 = add i32 %v155, 1
  br label %bb46
bb61:
  call void @llvm.nvvm.barrier0() #0
  br label %bb62
bb62:
  %v234 = udiv i32 %v14, 2
  %v235 = zext i32 %v234 to i64
  %v236 = mul i32 %v269, 128
  %v237 = zext i32 %v236 to i64
  %v238 = mul i32 %v18, 64
  %v239 = zext i32 %v238 to i64
  %v240 = zext i32 %v16 to i64
  br label %bb63
bb63:
  %v241 = phi i64 [ %v240, %bb62 ], [ %v258, %bb65 ]
  %v242 = icmp ult i64 %v241, 8192
  %v243 = xor i1 %v242, 1
  br i1 %v243, label %bb66, label %bb64
bb64:
  %v244 = zext i32 6 to i64
  %v245 = and i64 %v244, 63
  %v246 = lshr i64 %v241, %v245
  %v247 = and i64 %v241, 63
  %v248 = add i64 %v237, %v246
  %v249 = add i64 %v239, %v247
  %v250 = mul i64 %v248, %v235
  %v251 = add i64 %v250, %v249
  %v253 = bitcast ptr addrspace(3) @__shared_mem_91 to ptr addrspace(3)
  %v254 = getelementptr inbounds i32, ptr addrspace(3) %v253, i64 %v241
  br label %bb65
bb65:
  %v255 = load i32, ptr addrspace(3) %v254
  %v256 = extractvalue { ptr, i64 } %v10, 0
  %v257 = getelementptr inbounds i32, ptr %v256, i64 %v251
  store i32 %v255, ptr %v257
  %v258 = add i64 %v241, 128
  br label %bb63
bb66:
  call void @llvm.nvvm.barrier0() #0
  br label %bb67
bb67:
  %v260 = xor i1 %v25, 1
  br i1 %v260, label %bb69, label %bb68
bb68:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v33, i32 512) #0
  br label %bb69
bb69:
  %v261 = xor i1 %v268, 1
  br i1 %v261, label %bb72, label %bb70
bb70:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_86) #0
  br label %bb71
bb71:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_87) #0
  br label %bb72
bb72:
  ret void
bb73:
  %v266 = udiv i32 %v17, 32
  %v267 = urem i32 %v16, 32
  %v268 = icmp eq i32 %v16, 0
  %v269 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb2
}

define ptx_kernel void @gemm_sol_clc_multicast(ptr %v0, ptr %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7) {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi ptr [ %v0, %entry ]
  %v11 = phi ptr [ %v1, %entry ]
  %v12 = phi { ptr, i64 } [ %v9, %entry ]
  %v13 = phi i32 [ %v4, %entry ]
  %v14 = phi i32 [ %v5, %entry ]
  %v15 = phi i32 [ %v6, %entry ]
  %v16 = phi i32 [ %v7, %entry ]
  %v17 = alloca { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }
  br label %bb1
bb1:
  %v18 = bitcast i32 %v13 to i32
  %v19 = bitcast i32 %v14 to i32
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb2
bb2:
  %v21 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb259
bb3:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_92, i32 1) #0
  br label %bb4
bb4:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_93, i32 1) #0
  br label %bb5
bb5:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_94, i32 1) #0
  br label %bb6
bb6:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_95, i32 1) #0
  br label %bb7
bb7:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_96, i32 1) #0
  br label %bb8
bb8:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_97, i32 1) #0
  br label %bb9
bb9:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_98, i32 128) #0
  br label %bb10
bb10:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_99, i32 128) #0
  br label %bb11
bb11:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_100, i32 1) #0
  br label %bb12
bb12:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_101, i32 1) #0
  br label %bb13
bb13:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_102, i32 4) #0
  br label %bb14
bb14:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_103, i32 4) #0
  br label %bb15
bb15:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb16
bb16:
  call void @llvm.nvvm.barrier0() #0
  br label %bb17
bb17:
  %v47 = call i32 asm sideeffect "mov.u32 $0, %cluster_ctaid.x;", "=r"() #0
  br label %bb18
bb18:
  %v49 = bitcast ptr addrspace(3) @__shared_mem_102 to ptr addrspace(3)
  %v50 = call i64 asm sideeffect "mapa.shared::cluster.u64 $0, $1, $2;", "=l,l,r"(ptr addrspace(3) %v49, i32 0) #0
  %v51 = inttoptr i64 %v50 to ptr addrspace(3)
  br label %bb19
bb19:
  %v52 = ptrtoint ptr addrspace(3) %v51 to i64
  %v54 = bitcast ptr addrspace(3) @__shared_mem_103 to ptr addrspace(3)
  %v55 = call i64 asm sideeffect "mapa.shared::cluster.u64 $0, $1, $2;", "=l,l,r"(ptr addrspace(3) %v54, i32 0) #0
  %v56 = inttoptr i64 %v55 to ptr addrspace(3)
  br label %bb20
bb20:
  %v57 = ptrtoint ptr addrspace(3) %v56 to i64
  %v58 = xor i1 %v626, 1
  br i1 %v58, label %bb24, label %bb21
bb21:
  %v60 = bitcast ptr addrspace(3) @__shared_mem_94 to ptr addrspace(3)
  %v61 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v60) #0
  br label %bb22
bb22:
  %v63 = bitcast ptr addrspace(3) @__shared_mem_95 to ptr addrspace(3)
  %v64 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v63) #0
  br label %bb23
bb23:
  br label %bb24
bb24:
  call void @llvm.nvvm.barrier0() #0
  br label %bb25
bb25:
  %v66 = icmp eq i32 %v624, 0
  %v67 = icmp eq i32 %v624, 0
  br i1 %v67, label %bb26, label %bb28
bb26:
  %v69 = addrspacecast ptr addrspace(3) @__shared_mem_104 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v69, i32 512) #0
  br label %bb27
bb27:
  br label %bb28
bb28:
  call void @llvm.nvvm.barrier0() #0
  br label %bb29
bb29:
  %v72 = bitcast ptr addrspace(3) @__shared_mem_104 to ptr addrspace(3)
  %v73 = addrspacecast ptr addrspace(3) %v72 to ptr
  %v74 = load i32, ptr %v73
  %v75 = insertvalue { i8 } undef, i8 1, 0
  %v76 = insertvalue { i8 } undef, i8 0, 0
  %v77 = insertvalue { i8 } undef, i8 0, 0
  %v78 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 256, 0
  %v79 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v78, i32 128, 1
  %v80 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v79, i8 0, 2
  %v81 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v80, { i8 } %v76, 3
  %v82 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v81, { i8 } %v77, 4
  %v83 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v82, i1 0, 5
  %v84 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v83, { i8 } %v75, 6
  %v85 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v84, i1 0, 7
  %v86 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v85, i1 0, 8
  %v87 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v86, i1 0, 9
  %v88 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v87, i1 0, 10
  store { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v88, ptr %v17
  %v89 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 1
  store i32 128, ptr %v89
  %v90 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 0
  store i32 128, ptr %v90
  %v91 = insertvalue { i8 } undef, i8 0, 0
  %v92 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 3
  store { i8 } %v91, ptr %v92
  %v93 = insertvalue { i8 } undef, i8 0, 0
  %v94 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 4
  store { i8 } %v93, ptr %v94
  %v95 = insertvalue { i8 } undef, i8 1, 0
  %v96 = getelementptr inbounds { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17, i32 0, i32 6
  store { i8 } %v95, ptr %v96
  %v97 = load { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] }, ptr %v17
  %v98 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 0
  %v99 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 1
  %v100 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 2
  %v101 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 3
  %v102 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 4
  %v103 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 5
  %v104 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 6
  %v105 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 7
  %v106 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 8
  %v107 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 9
  %v108 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v97, 10
  %v109 = call { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v98, i32 %v99, i8 %v100, { i8 } %v101, { i8 } %v102, i1 %v103, { i8 } %v104, i1 %v105, i1 %v106, i1 %v107, i1 %v108)
  br label %bb30
bb30:
  %v110 = extractvalue { i32 } %v109, 0
  %v111 = udiv i32 %v19, 64
  call void asm sideeffect "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;", "~{memory}"() #0
  br label %bb31
bb31:
  %v112 = icmp eq i32 %v624, 4
  br i1 %v112, label %bb32, label %bb135
bb32:
  %v113 = icmp eq i32 %v625, 0
  %v114 = icmp eq i32 %v47, 0
  %v115 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb33
bb33:
  %v116 = icmp eq i32 %v15, 0
  %v117 = xor i1 %v116, 1
  br i1 %v117, label %bb34, label %bb260
bb34:
  %v118 = urem i32 %v115, %v15
  %v119 = udiv i32 %v115, %v15
  %v120 = xor i1 %v113, 1
  br i1 %v120, label %bb37, label %bb35
bb35:
  %v122 = addrspacecast ptr addrspace(3) @__shared_mem_105 to ptr
  store i32 %v118, ptr %v122
  %v123 = getelementptr inbounds i32, ptr %v122, i64 1
  store i32 %v119, ptr %v123
  %v124 = getelementptr inbounds i32, ptr %v122, i64 2
  store i32 1, ptr %v124
  %v126 = bitcast ptr addrspace(3) @__shared_mem_100 to ptr addrspace(3)
  %v127 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v126) #0
  br label %bb36
bb36:
  br label %bb37
bb37:
  %v128 = mul i32 %v118, 128
  %v129 = bitcast i32 %v128 to i32
  %v130 = mul i32 %v119, 128
  %v131 = bitcast i32 %v130 to i32
  br label %bb38
bb38:
  %v132 = phi i32 [ 0, %bb37 ], [ %v200, %bb75 ]
  %v133 = phi i32 [ 0, %bb37 ], [ %v199, %bb75 ]
  %v134 = icmp ult i32 %v133, %v111
  %v135 = xor i1 %v134, 1
  br i1 %v135, label %bb76, label %bb39
bb39:
  %v136 = and i32 %v132, 1
  %v137 = and i32 1, 31
  %v138 = lshr i32 %v132, %v137
  %v139 = and i32 %v138, 1
  %v140 = icmp eq i32 %v136, 0
  %v141 = icmp eq i32 %v136, 0
  br i1 %v141, label %bb40, label %bb44
bb40:
  %v143 = bitcast ptr addrspace(3) @__shared_mem_94 to ptr addrspace(3)
  %v144 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v143, i32 %v139) #0
  %v145 = trunc i32 %v144 to i1
  br label %bb41
bb41:
  %v146 = xor i1 %v145, 1
  br i1 %v146, label %bb43, label %bb42
bb42:
  br label %bb48
bb43:
  br label %bb40
bb44:
  %v148 = bitcast ptr addrspace(3) @__shared_mem_95 to ptr addrspace(3)
  %v149 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v148, i32 %v139) #0
  %v150 = trunc i32 %v149 to i1
  br label %bb45
bb45:
  %v151 = xor i1 %v150, 1
  br i1 %v151, label %bb47, label %bb46
bb46:
  br label %bb48
bb47:
  br label %bb44
bb48:
  %v152 = xor i1 %v113, 1
  br i1 %v152, label %bb53, label %bb49
bb49:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb50
bb50:
  %v153 = xor i1 %v140, 1
  br i1 %v153, label %bb52, label %bb51
bb51:
  call void asm sideeffect "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];", "l,~{memory}"(i64 %v52) #0
  br label %bb53
bb52:
  call void asm sideeffect "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];", "l,~{memory}"(i64 %v57) #0
  br label %bb53
bb53:
  %v154 = and i32 1, 31
  %v155 = lshr i32 %v132, %v154
  %v156 = and i32 %v155, 1
  %v157 = xor i1 %v114, 1
  br i1 %v157, label %bb63, label %bb54
bb54:
  %v158 = xor i1 %v140, 1
  br i1 %v158, label %bb59, label %bb55
bb55:
  %v159 = bitcast ptr addrspace(3) @__shared_mem_102 to ptr addrspace(3)
  %v160 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v159, i32 %v156) #0
  %v161 = trunc i32 %v160 to i1
  br label %bb56
bb56:
  %v162 = xor i1 %v161, 1
  br i1 %v162, label %bb58, label %bb57
bb57:
  br label %bb63
bb58:
  br label %bb55
bb59:
  %v163 = bitcast ptr addrspace(3) @__shared_mem_103 to ptr addrspace(3)
  %v164 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v163, i32 %v156) #0
  %v165 = trunc i32 %v164 to i1
  br label %bb60
bb60:
  %v166 = xor i1 %v165, 1
  br i1 %v166, label %bb62, label %bb61
bb61:
  br label %bb63
bb62:
  br label %bb59
bb63:
  %v167 = xor i1 %v113, 1
  br i1 %v167, label %bb75, label %bb64
bb64:
  %v168 = mul i32 %v133, 64
  %v169 = bitcast i32 %v168 to i32
  %v170 = xor i1 %v140, 1
  br i1 %v170, label %bb70, label %bb65
bb65:
  %v172 = bitcast ptr addrspace(3) @__shared_mem_92 to ptr addrspace(3)
  %v173 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v172, i32 32768) #0
  br label %bb66
bb66:
  %v175 = addrspacecast ptr addrspace(3) @__shared_mem_106 to ptr
  %v177 = addrspacecast ptr %v175 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v177, ptr addrspace(3) @__shared_mem_92, ptr %v10, i32 %v169, i32 %v129, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb67
bb67:
  %v179 = xor i1 %v114, 1
  br i1 %v179, label %bb75, label %bb68
bb68:
  %v181 = addrspacecast ptr addrspace(3) @__shared_mem_107 to ptr
  %v183 = addrspacecast ptr %v181 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v183, ptr addrspace(3) @__shared_mem_92, ptr %v11, i32 %v169, i32 %v131, i16 15, i64 0, i1 1, i1 0, i32 0) #0
  br label %bb69
bb69:
  br label %bb75
bb70:
  %v186 = bitcast ptr addrspace(3) @__shared_mem_93 to ptr addrspace(3)
  %v187 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v186, i32 32768) #0
  br label %bb71
bb71:
  %v189 = addrspacecast ptr addrspace(3) @__shared_mem_108 to ptr
  %v191 = addrspacecast ptr %v189 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v191, ptr addrspace(3) @__shared_mem_93, ptr %v10, i32 %v169, i32 %v129, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb72
bb72:
  %v193 = xor i1 %v114, 1
  br i1 %v193, label %bb75, label %bb73
bb73:
  %v195 = addrspacecast ptr addrspace(3) @__shared_mem_109 to ptr
  %v197 = addrspacecast ptr %v195 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v197, ptr addrspace(3) @__shared_mem_93, ptr %v11, i32 %v169, i32 %v131, i16 15, i64 0, i1 1, i1 0, i32 0) #0
  br label %bb74
bb74:
  br label %bb75
bb75:
  %v199 = add i32 %v133, 1
  %v200 = add i32 %v132, 1
  br label %bb38
bb76:
  %v202 = addrspacecast ptr addrspace(3) @__shared_mem_110 to ptr
  br label %bb77
bb77:
  %v203 = phi i32 [ %v132, %bb76 ], [ %v247, %bb134 ]
  %v204 = phi i32 [ 0, %bb76 ], [ %v316, %bb134 ]
  %v205 = and i32 %v204, 1
  %v206 = xor i1 %v113, 1
  br i1 %v206, label %bb82, label %bb78
bb78:
  %v208 = bitcast ptr addrspace(3) @__shared_mem_101 to ptr addrspace(3)
  %v209 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v208, i32 16) #0
  br label %bb79
bb79:
  %v210 = xor i1 %v114, 1
  br i1 %v210, label %bb82, label %bb80
bb80:
  %v212 = addrspacecast ptr addrspace(3) @__shared_mem_110 to ptr
  call void asm sideeffect "{ .reg .u64 %resp_shared64; .reg .u32 %resp_shared32; cvta.to.shared.u64 %resp_shared64, $0; cvt.u32.u64 %resp_shared32, %resp_shared64; .reg .u64 %mbar_shared64; .reg .u32 %mbar_shared32; cvta.to.shared.u64 %mbar_shared64, $1; cvt.u32.u64 %mbar_shared32, %mbar_shared64; clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%resp_shared32], [%mbar_shared32]; }", "l,l,~{memory}"(ptr %v212, ptr addrspace(3) @__shared_mem_101) #0
  br label %bb81
bb81:
  br label %bb82
bb82:
  %v215 = bitcast ptr addrspace(3) @__shared_mem_101 to ptr addrspace(3)
  %v216 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v215, i32 %v205) #0
  %v217 = trunc i32 %v216 to i1
  br label %bb83
bb83:
  %v218 = xor i1 %v217, 1
  br i1 %v218, label %bb85, label %bb84
bb84:
  %v219 = load i64, ptr %v202
  %v220 = getelementptr inbounds i64, ptr %v202, i64 1
  %v221 = load i64, ptr %v220
  %v222 = call i32 asm sideeffect "{ .reg .b128 %resp; mov.b128 %resp, {$1, $2}; .reg .pred %p; clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %resp; selp.b32 $0, 1, 0, %p; }", "=r,l,l"(i64 %v219, i64 %v221) #0
  br label %bb86
bb85:
  br label %bb82
bb86:
  %v223 = icmp eq i32 %v222, 0
  br i1 %v223, label %bb87, label %bb91
bb87:
  %v224 = xor i1 %v113, 1
  br i1 %v224, label %bb90, label %bb88
bb88:
  %v226 = addrspacecast ptr addrspace(3) @__shared_mem_105 to ptr
  %v227 = getelementptr inbounds i32, ptr %v226, i64 2
  store i32 0, ptr %v227
  %v229 = bitcast ptr addrspace(3) @__shared_mem_100 to ptr addrspace(3)
  %v230 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v229) #0
  br label %bb89
bb89:
  br label %bb90
bb90:
  br label %bb135
bb91:
  %v231 = call i32 asm sideeffect "{ .reg .b128 %resp; mov.b128 %resp, {$1, $2}; clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 $0, %resp; }", "=r,l,l"(i64 %v219, i64 %v221) #0
  br label %bb92
bb92:
  %v232 = add i32 %v231, %v47
  %v233 = urem i32 %v232, %v15
  %v234 = udiv i32 %v232, %v15
  %v235 = xor i1 %v113, 1
  br i1 %v235, label %bb95, label %bb93
bb93:
  %v237 = addrspacecast ptr addrspace(3) @__shared_mem_105 to ptr
  store i32 %v233, ptr %v237
  %v238 = getelementptr inbounds i32, ptr %v237, i64 1
  store i32 %v234, ptr %v238
  %v239 = getelementptr inbounds i32, ptr %v237, i64 2
  store i32 1, ptr %v239
  %v241 = bitcast ptr addrspace(3) @__shared_mem_100 to ptr addrspace(3)
  %v242 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v241) #0
  br label %bb94
bb94:
  br label %bb95
bb95:
  %v243 = mul i32 %v233, 128
  %v244 = bitcast i32 %v243 to i32
  %v245 = mul i32 %v234, 128
  %v246 = bitcast i32 %v245 to i32
  br label %bb96
bb96:
  %v247 = phi i32 [ %v203, %bb95 ], [ %v315, %bb133 ]
  %v248 = phi i32 [ 0, %bb95 ], [ %v314, %bb133 ]
  %v249 = icmp ult i32 %v248, %v111
  %v250 = xor i1 %v249, 1
  br i1 %v250, label %bb134, label %bb97
bb97:
  %v251 = and i32 %v247, 1
  %v252 = and i32 1, 31
  %v253 = lshr i32 %v247, %v252
  %v254 = and i32 %v253, 1
  %v255 = icmp eq i32 %v251, 0
  %v256 = icmp eq i32 %v251, 0
  br i1 %v256, label %bb98, label %bb102
bb98:
  %v258 = bitcast ptr addrspace(3) @__shared_mem_94 to ptr addrspace(3)
  %v259 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v258, i32 %v254) #0
  %v260 = trunc i32 %v259 to i1
  br label %bb99
bb99:
  %v261 = xor i1 %v260, 1
  br i1 %v261, label %bb101, label %bb100
bb100:
  br label %bb106
bb101:
  br label %bb98
bb102:
  %v263 = bitcast ptr addrspace(3) @__shared_mem_95 to ptr addrspace(3)
  %v264 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v263, i32 %v254) #0
  %v265 = trunc i32 %v264 to i1
  br label %bb103
bb103:
  %v266 = xor i1 %v265, 1
  br i1 %v266, label %bb105, label %bb104
bb104:
  br label %bb106
bb105:
  br label %bb102
bb106:
  %v267 = xor i1 %v113, 1
  br i1 %v267, label %bb111, label %bb107
bb107:
  call void asm sideeffect "fence.proxy.async.shared::cta;", "~{memory}"() #0
  ; Unknown op: nvvm.fence_proxy_async_shared_cta
  br label %bb108
bb108:
  %v268 = xor i1 %v255, 1
  br i1 %v268, label %bb110, label %bb109
bb109:
  call void asm sideeffect "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];", "l,~{memory}"(i64 %v52) #0
  br label %bb111
bb110:
  call void asm sideeffect "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];", "l,~{memory}"(i64 %v57) #0
  br label %bb111
bb111:
  %v269 = and i32 1, 31
  %v270 = lshr i32 %v247, %v269
  %v271 = and i32 %v270, 1
  %v272 = xor i1 %v114, 1
  br i1 %v272, label %bb121, label %bb112
bb112:
  %v273 = xor i1 %v255, 1
  br i1 %v273, label %bb117, label %bb113
bb113:
  %v274 = bitcast ptr addrspace(3) @__shared_mem_102 to ptr addrspace(3)
  %v275 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v274, i32 %v271) #0
  %v276 = trunc i32 %v275 to i1
  br label %bb114
bb114:
  %v277 = xor i1 %v276, 1
  br i1 %v277, label %bb116, label %bb115
bb115:
  br label %bb121
bb116:
  br label %bb113
bb117:
  %v278 = bitcast ptr addrspace(3) @__shared_mem_103 to ptr addrspace(3)
  %v279 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v278, i32 %v271) #0
  %v280 = trunc i32 %v279 to i1
  br label %bb118
bb118:
  %v281 = xor i1 %v280, 1
  br i1 %v281, label %bb120, label %bb119
bb119:
  br label %bb121
bb120:
  br label %bb117
bb121:
  %v282 = xor i1 %v113, 1
  br i1 %v282, label %bb133, label %bb122
bb122:
  %v283 = mul i32 %v248, 64
  %v284 = bitcast i32 %v283 to i32
  %v285 = xor i1 %v255, 1
  br i1 %v285, label %bb128, label %bb123
bb123:
  %v287 = bitcast ptr addrspace(3) @__shared_mem_92 to ptr addrspace(3)
  %v288 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v287, i32 32768) #0
  br label %bb124
bb124:
  %v290 = addrspacecast ptr addrspace(3) @__shared_mem_106 to ptr
  %v292 = addrspacecast ptr %v290 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v292, ptr addrspace(3) @__shared_mem_92, ptr %v10, i32 %v284, i32 %v244, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb125
bb125:
  %v294 = xor i1 %v114, 1
  br i1 %v294, label %bb133, label %bb126
bb126:
  %v296 = addrspacecast ptr addrspace(3) @__shared_mem_107 to ptr
  %v298 = addrspacecast ptr %v296 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v298, ptr addrspace(3) @__shared_mem_92, ptr %v11, i32 %v284, i32 %v246, i16 15, i64 0, i1 1, i1 0, i32 0) #0
  br label %bb127
bb127:
  br label %bb133
bb128:
  %v301 = bitcast ptr addrspace(3) @__shared_mem_93 to ptr addrspace(3)
  %v302 = call i64 asm sideeffect "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;", "=l,l,r,~{memory}"(ptr addrspace(3) %v301, i32 32768) #0
  br label %bb129
bb129:
  %v304 = addrspacecast ptr addrspace(3) @__shared_mem_108 to ptr
  %v306 = addrspacecast ptr %v304 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v306, ptr addrspace(3) @__shared_mem_93, ptr %v10, i32 %v284, i32 %v244, i16 0, i64 0, i1 0, i1 0, i32 0) #0
  br label %bb130
bb130:
  %v308 = xor i1 %v114, 1
  br i1 %v308, label %bb133, label %bb131
bb131:
  %v310 = addrspacecast ptr addrspace(3) @__shared_mem_109 to ptr
  %v312 = addrspacecast ptr %v310 to ptr addrspace(7)
  call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %v312, ptr addrspace(3) @__shared_mem_93, ptr %v11, i32 %v284, i32 %v246, i16 15, i64 0, i1 1, i1 0, i32 0) #0
  br label %bb132
bb132:
  br label %bb133
bb133:
  %v314 = add i32 %v248, 1
  %v315 = add i32 %v247, 1
  br label %bb96
bb134:
  %v316 = add i32 %v204, 1
  br label %bb77
bb135:
  %v317 = icmp eq i32 %v624, 5
  br i1 %v317, label %bb136, label %bb194
bb136:
  %v318 = icmp eq i32 %v625, 0
  br label %bb137
bb137:
  %v319 = phi i32 [ 0, %bb136 ], [ %v319, %bb140 ], [ %v431, %bb193 ]
  %v320 = phi i32 [ 0, %bb136 ], [ %v320, %bb140 ], [ %v327, %bb193 ]
  %v321 = phi i32 [ 0, %bb136 ], [ %v321, %bb140 ], [ %v352, %bb193 ]
  %v323 = bitcast ptr addrspace(3) @__shared_mem_100 to ptr addrspace(3)
  %v324 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v323, i32 %v320) #0
  %v325 = trunc i32 %v324 to i1
  br label %bb138
bb138:
  %v326 = xor i1 %v325, 1
  br i1 %v326, label %bb140, label %bb139
bb139:
  %v327 = xor i32 %v320, 1
  %v329 = bitcast ptr addrspace(3) @__shared_mem_105 to ptr addrspace(3)
  %v330 = addrspacecast ptr addrspace(3) %v329 to ptr
  %v331 = getelementptr inbounds i32, ptr %v330, i64 2
  %v332 = load i32, ptr %v331
  %v333 = icmp eq i32 %v332, 0
  br i1 %v333, label %bb141, label %bb142
bb140:
  br label %bb137
bb141:
  br label %bb194
bb142:
  %v334 = urem i32 %v319, 2
  %v335 = mul i32 %v334, 128
  %v336 = icmp uge i32 %v319, 2
  %v337 = xor i1 %v336, 1
  br i1 %v337, label %bb153, label %bb143
bb143:
  %v338 = sub i32 %v319, 2
  %v339 = udiv i32 %v338, 2
  %v340 = and i32 %v339, 1
  %v341 = icmp eq i32 %v334, 0
  br i1 %v341, label %bb144, label %bb148
bb144:
  %v343 = bitcast ptr addrspace(3) @__shared_mem_98 to ptr addrspace(3)
  %v344 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v343, i32 %v340) #0
  %v345 = trunc i32 %v344 to i1
  br label %bb145
bb145:
  %v346 = xor i1 %v345, 1
  br i1 %v346, label %bb147, label %bb146
bb146:
  br label %bb152
bb147:
  br label %bb144
bb148:
  %v348 = bitcast ptr addrspace(3) @__shared_mem_99 to ptr addrspace(3)
  %v349 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v348, i32 %v340) #0
  %v350 = trunc i32 %v349 to i1
  br label %bb149
bb149:
  %v351 = xor i1 %v350, 1
  br i1 %v351, label %bb151, label %bb150
bb150:
  br label %bb152
bb151:
  br label %bb148
bb152:
  br label %bb154
bb153:
  br label %bb154
bb154:
  br label %bb155
bb155:
  %v352 = phi i32 [ %v321, %bb154 ], [ %v424, %bb185 ]
  %v353 = phi i32 [ 0, %bb154 ], [ %v423, %bb185 ]
  %v354 = icmp ult i32 %v353, %v111
  %v355 = xor i1 %v354, 1
  br i1 %v355, label %bb186, label %bb156
bb156:
  %v356 = and i32 %v352, 1
  %v357 = and i32 1, 31
  %v358 = lshr i32 %v352, %v357
  %v359 = and i32 %v358, 1
  %v360 = icmp eq i32 %v356, 0
  %v361 = icmp eq i32 %v356, 0
  br i1 %v361, label %bb157, label %bb161
bb157:
  %v363 = bitcast ptr addrspace(3) @__shared_mem_92 to ptr addrspace(3)
  %v364 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v363, i32 %v359) #0
  %v365 = trunc i32 %v364 to i1
  br label %bb158
bb158:
  %v366 = xor i1 %v365, 1
  br i1 %v366, label %bb160, label %bb159
bb159:
  br label %bb165
bb160:
  br label %bb157
bb161:
  %v368 = bitcast ptr addrspace(3) @__shared_mem_93 to ptr addrspace(3)
  %v369 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v368, i32 %v359) #0
  %v370 = trunc i32 %v369 to i1
  br label %bb162
bb162:
  %v371 = xor i1 %v370, 1
  br i1 %v371, label %bb164, label %bb163
bb163:
  br label %bb165
bb164:
  br label %bb161
bb165:
  %v372 = xor i1 %v318, 1
  br i1 %v372, label %bb185, label %bb166
bb166:
  %v373 = xor i1 %v360, 1
  br i1 %v373, label %bb168, label %bb167
bb167:
  %v375 = bitcast ptr addrspace(3) @__shared_mem_106 to ptr addrspace(3)
  %v376 = ptrtoint ptr addrspace(3) %v375 to i64
  br label %bb169
bb168:
  %v378 = bitcast ptr addrspace(3) @__shared_mem_108 to ptr addrspace(3)
  %v379 = ptrtoint ptr addrspace(3) %v378 to i64
  br label %bb169
bb169:
  %v380 = phi i64 [ %v376, %bb167 ], [ %v379, %bb168 ]
  %v381 = xor i1 %v360, 1
  br i1 %v381, label %bb171, label %bb170
bb170:
  %v383 = bitcast ptr addrspace(3) @__shared_mem_107 to ptr addrspace(3)
  %v384 = ptrtoint ptr addrspace(3) %v383 to i64
  br label %bb172
bb171:
  %v386 = bitcast ptr addrspace(3) @__shared_mem_109 to ptr addrspace(3)
  %v387 = ptrtoint ptr addrspace(3) %v386 to i64
  br label %bb172
bb172:
  %v388 = phi i64 [ %v384, %bb170 ], [ %v387, %bb171 ]
  br label %bb173
bb173:
  %v389 = phi i32 [ 0, %bb172 ], [ %v417, %bb178 ]
  %v390 = icmp ult i32 %v389, 4
  %v391 = xor i1 %v390, 1
  br i1 %v391, label %bb179, label %bb174
bb174:
  %v392 = mul i32 %v389, 32
  %v393 = zext i32 %v392 to i64
  %v394 = add i64 %v380, %v393
  %v395 = zext i32 4 to i64
  %v396 = and i64 %v395, 63
  %v397 = lshr i64 %v394, %v396
  %v398 = and i64 %v397, 16383
  %v399 = or i64 %v398, 65536
  %v400 = or i64 %v399, 274877906944
  %v401 = or i64 %v400, 70368744177664
  %v402 = or i64 %v401, 4611686018427387904
  %v403 = add i64 %v388, %v393
  %v404 = zext i32 4 to i64
  %v405 = and i64 %v404, 63
  %v406 = lshr i64 %v403, %v405
  %v407 = and i64 %v406, 16383
  %v408 = or i64 %v407, 65536
  %v409 = or i64 %v408, 274877906944
  %v410 = or i64 %v409, 70368744177664
  %v411 = or i64 %v410, 4611686018427387904
  %v412 = icmp ugt i32 %v353, 0
  %v413 = xor i1 %v412, 1
  br i1 %v413, label %bb176, label %bb175
bb175:
  br label %bb177
bb176:
  %v414 = icmp ugt i32 %v389, 0
  br label %bb177
bb177:
  %v415 = phi i1 [ 1, %bb175 ], [ %v414, %bb176 ]
  %v416 = add i32 %v74, %v335
  call void asm sideeffect "{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; .reg .u32 %z; mov.u32 %z, 0; tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; }", "r,l,l,r,r,~{memory}"(i32 %v416, i64 %v402, i64 %v411, i32 %v110, i1 %v415) #0
  br label %bb178
bb178:
  %v417 = add i32 %v389, 1
  br label %bb173
bb179:
  %v418 = xor i1 %v360, 1
  br i1 %v418, label %bb181, label %bb180
bb180:
  %v420 = addrspacecast ptr addrspace(3) @__shared_mem_94 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v420) #0
  br label %bb182
bb181:
  %v422 = addrspacecast ptr addrspace(3) @__shared_mem_95 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v422) #0
  br label %bb183
bb182:
  br label %bb184
bb183:
  br label %bb184
bb184:
  br label %bb185
bb185:
  %v423 = add i32 %v353, 1
  %v424 = add i32 %v352, 1
  br label %bb155
bb186:
  %v425 = xor i1 %v318, 1
  br i1 %v425, label %bb193, label %bb187
bb187:
  %v426 = icmp eq i32 %v334, 0
  br i1 %v426, label %bb188, label %bb190
bb188:
  %v428 = addrspacecast ptr addrspace(3) @__shared_mem_96 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v428) #0
  br label %bb189
bb189:
  br label %bb192
bb190:
  %v430 = addrspacecast ptr addrspace(3) @__shared_mem_97 to ptr
  call void asm sideeffect "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];", "r,~{memory}"(ptr %v430) #0
  br label %bb191
bb191:
  br label %bb192
bb192:
  br label %bb193
bb193:
  %v431 = add i32 %v319, 1
  br label %bb137
bb194:
  %v432 = icmp ult i32 %v624, 4
  %v433 = xor i1 %v432, 1
  br i1 %v433, label %bb242, label %bb195
bb195:
  %v434 = mul i32 %v624, 32
  %v435 = zext i32 %v434 to i64
  %v436 = urem i32 %v625, 8
  %v437 = zext i32 %v436 to i64
  %v438 = icmp uge i32 %v625, 8
  %v439 = xor i1 %v438, 1
  br i1 %v439, label %bb197, label %bb196
bb196:
  %v440 = icmp ult i32 %v625, 16
  br label %bb198
bb197:
  br label %bb198
bb198:
  %v441 = phi i1 [ %v440, %bb196 ], [ 0, %bb197 ]
  %v442 = xor i1 %v441, 1
  br i1 %v442, label %bb200, label %bb199
bb199:
  br label %bb201
bb200:
  br label %bb201
bb201:
  %v443 = phi i64 [ 16, %bb199 ], [ 0, %bb200 ]
  br label %bb202
bb202:
  %v444 = phi i32 [ 0, %bb201 ], [ %v444, %bb205 ], [ %v596, %bb241 ]
  %v445 = phi i32 [ 0, %bb201 ], [ %v445, %bb205 ], [ %v451, %bb241 ]
  %v447 = bitcast ptr addrspace(3) @__shared_mem_100 to ptr addrspace(3)
  %v448 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v447, i32 %v445) #0
  %v449 = trunc i32 %v448 to i1
  br label %bb203
bb203:
  %v450 = xor i1 %v449, 1
  br i1 %v450, label %bb205, label %bb204
bb204:
  %v451 = xor i32 %v445, 1
  %v453 = bitcast ptr addrspace(3) @__shared_mem_105 to ptr addrspace(3)
  %v454 = addrspacecast ptr addrspace(3) %v453 to ptr
  %v455 = getelementptr inbounds i32, ptr %v454, i64 2
  %v456 = load i32, ptr %v455
  %v457 = icmp eq i32 %v456, 0
  br i1 %v457, label %bb206, label %bb207
bb205:
  br label %bb202
bb206:
  br label %bb242
bb207:
  %v458 = bitcast ptr addrspace(3) @__shared_mem_105 to ptr addrspace(3)
  %v459 = addrspacecast ptr addrspace(3) %v458 to ptr
  %v460 = load i32, ptr %v459
  %v461 = bitcast ptr addrspace(3) @__shared_mem_105 to ptr addrspace(3)
  %v462 = addrspacecast ptr addrspace(3) %v461 to ptr
  %v463 = getelementptr inbounds i32, ptr %v462, i64 1
  %v464 = load i32, ptr %v463
  %v465 = urem i32 %v444, 2
  %v466 = mul i32 %v465, 128
  %v467 = udiv i32 %v444, 2
  %v468 = and i32 %v467, 1
  %v469 = icmp eq i32 %v465, 0
  %v470 = icmp eq i32 %v465, 0
  br i1 %v470, label %bb208, label %bb212
bb208:
  %v472 = bitcast ptr addrspace(3) @__shared_mem_96 to ptr addrspace(3)
  %v473 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v472, i32 %v468) #0
  %v474 = trunc i32 %v473 to i1
  br label %bb209
bb209:
  %v475 = xor i1 %v474, 1
  br i1 %v475, label %bb211, label %bb210
bb210:
  br label %bb216
bb211:
  br label %bb208
bb212:
  %v477 = bitcast ptr addrspace(3) @__shared_mem_97 to ptr addrspace(3)
  %v478 = call i32 asm sideeffect "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }", "=r,l,r,~{memory}"(ptr addrspace(3) %v477, i32 %v468) #0
  %v479 = trunc i32 %v478 to i1
  br label %bb213
bb213:
  %v480 = xor i1 %v479, 1
  br i1 %v480, label %bb215, label %bb214
bb214:
  br label %bb216
bb215:
  br label %bb212
bb216:
  br label %bb217
bb217:
  %v481 = phi i32 [ 0, %bb216 ], [ %v559, %bb231 ]
  %v482 = icmp ult i32 %v481, 2
  %v483 = xor i1 %v482, 1
  br i1 %v483, label %bb232, label %bb218
bb218:
  %v484 = mul i32 %v481, 16
  %v485 = add i32 %v434, %v484
  br label %bb219
bb219:
  %v486 = phi i32 [ 0, %bb218 ], [ %v558, %bb230 ]
  %v487 = icmp ult i32 %v486, 8
  %v488 = xor i1 %v487, 1
  br i1 %v488, label %bb231, label %bb220
bb220:
  %v489 = mul i32 %v486, 16
  %v490 = zext i32 %v489 to i64
  %v491 = add i32 %v74, %v466
  %v492 = and i32 16, 31
  %v493 = shl i32 %v485, %v492
  %v494 = add i32 %v491, %v493
  %v495 = trunc i64 %v490 to i32
  %v496 = add i32 %v494, %v495
  %v497 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v496) #0
  %v498 = extractvalue { float, float, float, float } %v497, 0
  %v499 = extractvalue { float, float, float, float } %v497, 1
  %v500 = extractvalue { float, float, float, float } %v497, 2
  %v501 = extractvalue { float, float, float, float } %v497, 3
  %v502 = insertvalue [4 x float] undef, float %v498, 0
  %v503 = insertvalue [4 x float] %v502, float %v499, 1
  %v504 = insertvalue [4 x float] %v503, float %v500, 2
  %v505 = insertvalue [4 x float] %v504, float %v501, 3
  %v506 = insertvalue { [4 x float] } undef, [4 x float] %v505, 0
  br label %bb221
bb221:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb222
bb222:
  %v507 = add i32 %v496, 8
  %v508 = call { float, float, float, float } asm sideeffect "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];", "=f,=f,=f,=f,r"(i32 %v507) #0
  %v509 = extractvalue { float, float, float, float } %v508, 0
  %v510 = extractvalue { float, float, float, float } %v508, 1
  %v511 = extractvalue { float, float, float, float } %v508, 2
  %v512 = extractvalue { float, float, float, float } %v508, 3
  %v513 = insertvalue [4 x float] undef, float %v509, 0
  %v514 = insertvalue [4 x float] %v513, float %v510, 1
  %v515 = insertvalue [4 x float] %v514, float %v511, 2
  %v516 = insertvalue [4 x float] %v515, float %v512, 3
  %v517 = insertvalue { [4 x float] } undef, [4 x float] %v516, 0
  br label %bb223
bb223:
  call void asm sideeffect "tcgen05.wait::ld.sync.aligned;", "~{memory}"() #0
  br label %bb224
bb224:
  %v518 = extractvalue { [4 x float] } %v506, 0
  %v519 = extractvalue [4 x float] %v518, 0
  %v520 = extractvalue { [4 x float] } %v506, 0
  %v521 = extractvalue [4 x float] %v520, 1
  %v522 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v519, float %v521)
  br label %bb225
bb225:
  %v523 = extractvalue { [4 x float] } %v517, 0
  %v524 = extractvalue [4 x float] %v523, 0
  %v525 = extractvalue { [4 x float] } %v517, 0
  %v526 = extractvalue [4 x float] %v525, 1
  %v527 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v524, float %v526)
  br label %bb226
bb226:
  %v528 = zext i32 %v481 to i64
  %v529 = mul i64 %v528, 16
  %v530 = add i64 %v435, %v529
  %v531 = add i64 %v530, %v437
  %v533 = addrspacecast ptr addrspace(3) @__shared_mem_111 to ptr
  %v534 = mul i64 %v531, 256
  %v535 = mul i64 %v490, 2
  %v536 = add i64 %v534, %v535
  %v537 = add i64 %v536, %v443
  %v538 = getelementptr inbounds i8, ptr %v533, i64 %v537
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v538, i32 %v522, i32 %v527) #0
  br label %bb227
bb227:
  %v539 = extractvalue { [4 x float] } %v506, 0
  %v540 = extractvalue [4 x float] %v539, 2
  %v541 = extractvalue { [4 x float] } %v506, 0
  %v542 = extractvalue [4 x float] %v541, 3
  %v543 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v540, float %v542)
  br label %bb228
bb228:
  %v544 = extractvalue { [4 x float] } %v517, 0
  %v545 = extractvalue [4 x float] %v544, 2
  %v546 = extractvalue { [4 x float] } %v517, 0
  %v547 = extractvalue [4 x float] %v546, 3
  %v548 = call i32 asm sideeffect "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f"(float %v545, float %v547)
  br label %bb229
bb229:
  %v549 = zext i32 %v481 to i64
  %v550 = mul i64 %v549, 16
  %v551 = add i64 %v435, %v550
  %v552 = add i64 %v551, 8
  %v553 = add i64 %v552, %v437
  %v554 = mul i64 %v553, 256
  %v555 = add i64 %v554, %v535
  %v556 = add i64 %v555, %v443
  %v557 = getelementptr inbounds i8, ptr %v533, i64 %v556
  call void asm sideeffect "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }", "l,r,r,~{memory}"(ptr %v557, i32 %v543, i32 %v548) #0
  br label %bb230
bb230:
  %v558 = add i32 %v486, 1
  br label %bb219
bb231:
  %v559 = add i32 %v481, 1
  br label %bb217
bb232:
  %v560 = udiv i32 %v18, 2
  %v561 = zext i32 %v560 to i64
  %v562 = mul i32 %v460, 128
  %v563 = zext i32 %v562 to i64
  %v564 = mul i32 %v464, 64
  %v565 = zext i32 %v564 to i64
  %v566 = zext i32 %v624 to i64
  %v567 = mul i64 %v566, 32
  %v568 = zext i32 %v625 to i64
  br label %bb233
bb233:
  %v569 = phi i64 [ %v568, %bb232 ], [ %v588, %bb235 ]
  %v570 = icmp ult i64 %v569, 2048
  %v571 = xor i1 %v570, 1
  br i1 %v571, label %bb236, label %bb234
bb234:
  %v572 = udiv i64 %v569, 64
  %v573 = urem i64 %v569, 64
  %v574 = add i64 %v567, %v572
  %v575 = mul i64 %v574, 64
  %v576 = add i64 %v575, %v573
  %v577 = add i64 %v563, %v567
  %v578 = add i64 %v577, %v572
  %v579 = add i64 %v565, %v573
  %v580 = mul i64 %v578, %v561
  %v581 = add i64 %v580, %v579
  %v583 = bitcast ptr addrspace(3) @__shared_mem_111 to ptr addrspace(3)
  %v584 = getelementptr inbounds i32, ptr addrspace(3) %v583, i64 %v576
  br label %bb235
bb235:
  %v585 = load i32, ptr addrspace(3) %v584
  %v586 = extractvalue { ptr, i64 } %v12, 0
  %v587 = getelementptr inbounds i32, ptr %v586, i64 %v581
  store i32 %v585, ptr %v587
  %v588 = add i64 %v569, 32
  br label %bb233
bb236:
  %v589 = xor i1 %v469, 1
  br i1 %v589, label %bb238, label %bb237
bb237:
  %v591 = bitcast ptr addrspace(3) @__shared_mem_98 to ptr addrspace(3)
  %v592 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v591) #0
  br label %bb239
bb238:
  %v594 = bitcast ptr addrspace(3) @__shared_mem_99 to ptr addrspace(3)
  %v595 = call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3) %v594) #0
  br label %bb240
bb239:
  br label %bb241
bb240:
  br label %bb241
bb241:
  %v596 = add i32 %v444, 1
  br label %bb202
bb242:
  call void @llvm.nvvm.barrier0() #0
  br label %bb243
bb243:
  %v598 = xor i1 %v66, 1
  br i1 %v598, label %bb245, label %bb244
bb244:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v74, i32 512) #0
  br label %bb245
bb245:
  %v599 = xor i1 %v626, 1
  br i1 %v599, label %bb258, label %bb246
bb246:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_92) #0
  br label %bb247
bb247:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_93) #0
  br label %bb248
bb248:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_94) #0
  br label %bb249
bb249:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_95) #0
  br label %bb250
bb250:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_96) #0
  br label %bb251
bb251:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_97) #0
  br label %bb252
bb252:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_98) #0
  br label %bb253
bb253:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_99) #0
  br label %bb254
bb254:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_100) #0
  br label %bb255
bb255:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_101) #0
  br label %bb256
bb256:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_102) #0
  br label %bb257
bb257:
  call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3) @__shared_mem_103) #0
  br label %bb258
bb258:
  ret void
bb259:
  %v624 = udiv i32 %v21, 32
  %v625 = urem i32 %v20, 32
  %v626 = icmp eq i32 %v20, 0
  %v627 = icmp eq i32 %v20, 0
  br i1 %v627, label %bb3, label %bb16
bb260:
  unreachable
}

define { i32 } @cuda_device__tcgen05__Tcgen05InstructionDescriptorBuilder__build(i32 %v0, i32 %v1, i8 %v2, { i8 } %v3, { i8 } %v4, i1 %v5, { i8 } %v6, i1 %v7, i1 %v8, i1 %v9, i1 %v10) {
entry:
  %v11 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } undef, i32 %v0, 0
  %v12 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v11, i32 %v1, 1
  %v13 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v12, i8 %v2, 2
  %v14 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v13, { i8 } %v3, 3
  %v15 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v14, { i8 } %v4, 4
  %v16 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v15, i1 %v5, 5
  %v17 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v16, { i8 } %v6, 6
  %v18 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v17, i1 %v7, 7
  %v19 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v18, i1 %v8, 8
  %v20 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v19, i1 %v9, 9
  %v21 = insertvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v20, i1 %v10, 10
  br label %bb0
bb0:
  %v22 = phi { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } [ %v21, %entry ]
  %v23 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 5
  %v24 = xor i1 %v23, 1
  br i1 %v24, label %bb2, label %bb1
bb1:
  %v25 = or i32 0, 4
  br label %bb2
bb2:
  %v26 = phi i32 [ 0, %bb0 ], [ %v25, %bb1 ]
  %v27 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 6
  %v28 = extractvalue { i8 } %v27, 0
  %v29 = zext i8 %v28 to i32
  %v30 = and i32 4, 31
  %v31 = shl i32 %v29, %v30
  %v32 = or i32 %v26, %v31
  %v33 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 3
  %v34 = extractvalue { i8 } %v33, 0
  %v35 = zext i8 %v34 to i32
  %v36 = and i32 7, 31
  %v37 = shl i32 %v35, %v36
  %v38 = or i32 %v32, %v37
  %v39 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 4
  %v40 = extractvalue { i8 } %v39, 0
  %v41 = zext i8 %v40 to i32
  %v42 = and i32 10, 31
  %v43 = shl i32 %v41, %v42
  %v44 = or i32 %v38, %v43
  %v45 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 7
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb4, label %bb3
bb3:
  %v47 = or i32 %v44, 8192
  br label %bb4
bb4:
  %v48 = phi i32 [ %v44, %bb2 ], [ %v47, %bb3 ]
  %v49 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 8
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb6, label %bb5
bb5:
  %v51 = or i32 %v48, 16384
  br label %bb6
bb6:
  %v52 = phi i32 [ %v48, %bb4 ], [ %v51, %bb5 ]
  %v53 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 9
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb8, label %bb7
bb7:
  %v55 = or i32 %v52, 32768
  br label %bb8
bb8:
  %v56 = phi i32 [ %v52, %bb6 ], [ %v55, %bb7 ]
  %v57 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 10
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb10, label %bb9
bb9:
  %v59 = or i32 %v56, 65536
  br label %bb10
bb10:
  %v60 = phi i32 [ %v56, %bb8 ], [ %v59, %bb9 ]
  %v61 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 0
  %v62 = and i32 3, 31
  %v63 = lshr i32 %v61, %v62
  %v64 = and i32 %v63, 63
  %v65 = and i32 17, 31
  %v66 = shl i32 %v64, %v65
  %v67 = or i32 %v60, %v66
  %v68 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 1
  %v69 = and i32 4, 31
  %v70 = lshr i32 %v68, %v69
  %v71 = and i32 %v70, 31
  %v72 = and i32 24, 31
  %v73 = shl i32 %v71, %v72
  %v74 = or i32 %v67, %v73
  %v75 = extractvalue { i32, i32, i8, { i8 }, { i8 }, i1, { i8 }, i1, i1, i1, i1, [3 x i8] } %v22, 2
  %v76 = zext i8 %v75 to i32
  %v77 = and i32 %v76, 3
  %v78 = and i32 30, 31
  %v79 = shl i32 %v77, %v78
  %v80 = or i32 %v74, %v79
  %v81 = insertvalue { i32 } undef, i32 %v80, 0
  ret { i32 } %v81
}

define void @_RINvNtCsfIVmHVe5YxZ_11cuda_device7cluster16___cluster_configKm4_Km1_KB11_ECs9V1XQbg6Nj2_14oxide_gemm_sol() {
entry:
  br label %bb0
bb0:
  ret void
}

define void @_RINvNtCsfIVmHVe5YxZ_11cuda_device7cluster16___cluster_configKm2_Km1_KB11_ECs9V1XQbg6Nj2_14oxide_gemm_sol() {
entry:
  br label %bb0
bb0:
  ret void
}


attributes #0 = { convergent }

!0 = !{ptr @gemm_sol_persistent, !"kernel", i32 1, !"cluster_dim_x", i32 4, !"cluster_dim_y", i32 1, !"cluster_dim_z", i32 1}
!1 = !{ptr @gemm_sol_clc, !"kernel", i32 1, !"cluster_dim_x", i32 4, !"cluster_dim_y", i32 1, !"cluster_dim_z", i32 1}
!2 = !{ptr @gemm_sol_clc_multicast_4_stage_pipeline, !"kernel", i32 1, !"cluster_dim_x", i32 2, !"cluster_dim_y", i32 1, !"cluster_dim_z", i32 1}
!3 = !{ptr @gemm_sol_clc_multicast, !"kernel", i32 1, !"cluster_dim_x", i32 4, !"cluster_dim_y", i32 1, !"cluster_dim_z", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3}
