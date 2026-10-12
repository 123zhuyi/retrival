{
  "op_name": "chunk_gated_delta_fwd_h",
  "warmup": 5,
  "repeats": 50,
  "max_retries": 3,
  "total_cases": 8,
  "passed_cases": 8,
  "failed_cases": 0,
  "nan_indices": [],
  "inf_indices": [],
  "zero_indices": [],
  "negative_indices": [],
  "none_indices": [],
  "framework": {
    "avg_latency_ms": 14.7163,
    "peak_memory_mb": 1785.73,
    "operators": {
      "aclnnAdd_AddAiCore_Add": {
        "avg_us": 1189.5442
      },
      "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
        "avg_us": 5979.6024
      },
      "aclnnExp_ExpAiCore_Exp": {
        "avg_us": 423.3834
      },
      "aclnnExp_SliceAiCore_Slice": {
        "avg_us": 535.4524
      },
      "aclnnInplaceCopy_CastAiCore_Cast": {
        "avg_us": 1334.1637
      },
      "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
        "avg_us": 16.4234
      },
      "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
        "avg_us": 2505.2997
      },
      "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
        "avg_us": 212.8046
      },
      "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
        "avg_us": 2268.812
      },
      "aclnnMatmul_SliceAiCore_Slice": {
        "avg_us": 1734.6835
      },
      "aclnnMatmul_TransposeAiCore_Transpose": {
        "avg_us": 1980.7589
      },
      "aclnnMul_MulAiCore_Mul": {
        "avg_us": 3757.9667
      },
      "aclnnSub_SliceAiCore_Slice": {
        "avg_us": 1797.9499
      },
      "aclnnSub_SubAiCore_Sub": {
        "avg_us": 2345.2784
      },
      "aclnnArange_ArangeAiCore_Range": {
        "avg_us": 2.1013
      },
      "aclnnFloorDivides_FloorDivAiCore_FloorDiv": {
        "avg_us": 1.8847
      },
      "aclnnIndexSelect_GatherV3AiCore_GatherV3": {
        "avg_us": 16.4528
      },
      "aclnnAdds_AddAiCore_Add": {
        "avg_us": 1.5507
      },
      "aclnnCat_ConcatD_ConcatD": {
        "avg_us": 0.7546
      },
      "aclnnCumsum_CumsumAiCpu_Cumsum": {
        "avg_us": 13.1742
      },
      "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum": {
        "avg_us": 0.703
      },
      "aclnnSubs_SubAiCore_Sub": {
        "avg_us": 1.4504
      }
    }
  },
  "implementation": {
    "avg_latency_ms": 2.0689,
    "peak_memory_mb": 549.89,
    "operators": {
      "_gdn_chunk_fwd_h_kernel": {
        "avg_us": 4245.6072
      },
      "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
        "avg_us": 1.0786
      }
    }
  },
  "speedup_vs_torch": 7.1132,
  "per_shape_results": [
    {
      "case_idx": 1,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": 64
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 5.9361,
        "peak_memory_mb": 161.36,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 362.2329,
            "launch_count": 51
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 1296.8413,
            "launch_count": 153
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 125.6784,
            "launch_count": 102
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 96.6515,
            "launch_count": 51
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 43.9125,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 19.5052,
            "launch_count": 3
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 788.4028,
            "launch_count": 102
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 46.5268,
            "launch_count": 6
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 540.3074,
            "launch_count": 102
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 269.0847,
            "launch_count": 102
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 599.9346,
            "launch_count": 51
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 884.874,
            "launch_count": 102
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 369.9333,
            "launch_count": 153
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 492.1959,
            "launch_count": 102
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 0.4429,
        "peak_memory_mb": 56.38,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 441.6128,
            "launch_count": 1
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 1.3368,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 13.4028,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 2,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            3,
            10,
            1054
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": 128
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 3.6082,
        "peak_memory_mb": 138.67,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 192.0694,
            "launch_count": 27
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 699.4525,
            "launch_count": 81
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 68.3607,
            "launch_count": 54
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 50.4974,
            "launch_count": 27
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 40.4115,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 19.5824,
            "launch_count": 3
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 435.811,
            "launch_count": 54
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 42.7535,
            "launch_count": 6
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 307.7893,
            "launch_count": 54
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 171.743,
            "launch_count": 54
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 351.6769,
            "launch_count": 27
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 573.9898,
            "launch_count": 54
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 206.1623,
            "launch_count": 81
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 447.8633,
            "launch_count": 54
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 0.2759,
        "peak_memory_mb": 55.44,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 274.5563,
            "launch_count": 1
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 1.3492,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 13.0779,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 3,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": 64
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 25.4043,
        "peak_memory_mb": 1934.02,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 1074.9125,
            "launch_count": 128
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 6534.5202,
            "launch_count": 384
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 361.72,
            "launch_count": 256
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 476.6227,
            "launch_count": 128
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 1433.4485,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 7.6798,
            "launch_count": 1
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 2803.9337,
            "launch_count": 256
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 375.0732,
            "launch_count": 4
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 1754.2166,
            "launch_count": 256
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 1460.1955,
            "launch_count": 256
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 1711.3074,
            "launch_count": 128
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 3700.4688,
            "launch_count": 256
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 1581.8075,
            "launch_count": 384
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 2128.3938,
            "launch_count": 256
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 5.2094,
        "peak_memory_mb": 593.01,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 5207.7472,
            "launch_count": 1
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 1.6392,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 4.8766,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 4,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            48,
            8192
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": 128
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 15.2205,
        "peak_memory_mb": 1648.03,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 557.8219,
            "launch_count": 64
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 3952.9499,
            "launch_count": 192
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 208.6722,
            "launch_count": 128
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 241.9489,
            "launch_count": 64
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 1173.8819,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 7.5097,
            "launch_count": 1
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 1501.3006,
            "launch_count": 128
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 240.9801,
            "launch_count": 4
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 1319.6653,
            "launch_count": 128
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 1084.9857,
            "launch_count": 128
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 981.7323,
            "launch_count": 64
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 1895.4159,
            "launch_count": 128
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 956.5144,
            "launch_count": 192
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 1097.0729,
            "launch_count": 128
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 3.2134,
        "peak_memory_mb": 506.01,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 3211.6954,
            "launch_count": 1
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 1.656,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 4.7366,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 5,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            1,
            8,
            4096,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            24,
            4096,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            24,
            4096,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            24,
            4096
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": 64
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 10.0659,
        "peak_memory_mb": 470.26,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 486.5517,
            "launch_count": 64
          },
          "aclnnArange_ArangeAiCore_Range": {
            "avg_us": 8.4193,
            "launch_count": 1
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 2399.8308,
            "launch_count": 192
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 163.5309,
            "launch_count": 128
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 227.5955,
            "launch_count": 64
          },
          "aclnnFloorDivides_FloorDivAiCore_FloorDiv": {
            "avg_us": 2.8645,
            "launch_count": 1
          },
          "aclnnIndexSelect_GatherV3AiCore_GatherV3": {
            "avg_us": 68.1838,
            "launch_count": 1
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 269.9894,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 6.8545,
            "launch_count": 1
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 994.8447,
            "launch_count": 128
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 82.6652,
            "launch_count": 4
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 692.3191,
            "launch_count": 128
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 495.6954,
            "launch_count": 128
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 847.1563,
            "launch_count": 64
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 1707.2758,
            "launch_count": 128
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 652.7689,
            "launch_count": 192
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 959.3132,
            "launch_count": 128
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 1.501,
        "peak_memory_mb": 135.88,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 1499.6756,
            "launch_count": 1
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 1.3265,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 6.7061,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 6,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            1,
            8,
            4096,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            24,
            4096,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            24,
            4096,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            24,
            4096
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "scalar",
          "value": 128
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 5.7225,
        "peak_memory_mb": 400.77,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 244.2289,
            "launch_count": 32
          },
          "aclnnArange_ArangeAiCore_Range": {
            "avg_us": 8.391,
            "launch_count": 1
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 1472.4611,
            "launch_count": 96
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 88.2633,
            "launch_count": 64
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 116.1141,
            "launch_count": 32
          },
          "aclnnFloorDivides_FloorDivAiCore_FloorDiv": {
            "avg_us": 2.7989,
            "launch_count": 1
          },
          "aclnnIndexSelect_GatherV3AiCore_GatherV3": {
            "avg_us": 63.4385,
            "launch_count": 1
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 190.8235,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 7.1506,
            "launch_count": 1
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 510.3039,
            "launch_count": 64
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 62.0287,
            "launch_count": 4
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 448.7672,
            "launch_count": 64
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 333.2294,
            "launch_count": 64
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 443.8775,
            "launch_count": 32
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 866.2649,
            "launch_count": 64
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 383.1303,
            "launch_count": 96
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 481.2584,
            "launch_count": 64
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 0.9312,
        "peak_memory_mb": 116.38,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 929.9117,
            "launch_count": 1
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 1.3212,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 6.1453,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 7,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "tensor",
          "shape": [
            5
          ],
          "dtype": "torch.int64"
        },
        {
          "type": "scalar",
          "value": 64
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 93.8879,
        "peak_memory_mb": 5148.7,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 4578.2869,
            "launch_count": 515
          },
          "aclnnAdds_AddAiCore_Add": {
            "avg_us": 7.2257,
            "launch_count": 2
          },
          "aclnnCat_ConcatD_ConcatD": {
            "avg_us": 3.7496,
            "launch_count": 1
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 20243.9176,
            "launch_count": 1545
          },
          "aclnnCumsum_CumsumAiCpu_Cumsum": {
            "avg_us": 58.7684,
            "launch_count": 1
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 1618.302,
            "launch_count": 1030
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 2167.8992,
            "launch_count": 515
          },
          "aclnnFloorDivides_FloorDivAiCore_FloorDiv": {
            "avg_us": 5.3018,
            "launch_count": 2
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 4205.86,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 34.1631,
            "launch_count": 4
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 8795.7507,
            "launch_count": 1030
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 242.8118,
            "launch_count": 7
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 9069.2847,
            "launch_count": 1030
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 6445.8847,
            "launch_count": 1030
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 7281.8238,
            "launch_count": 515
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 13633.1416,
            "launch_count": 1030
          },
          "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum": {
            "avg_us": 3.0797,
            "launch_count": 1
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 6815.5049,
            "launch_count": 1545
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 8670.8943,
            "launch_count": 1031
          },
          "aclnnSubs_SubAiCore_Sub": {
            "avg_us": 6.2873,
            "launch_count": 2
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 14.0356,
        "peak_memory_mb": 1584.01,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 14035.6252,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 6.6893,
      "error_type": null,
      "error_msg": null
    },
    {
      "case_idx": 8,
      "input_desc": [
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768,
            128
          ],
          "dtype": "torch.bfloat16"
        },
        {
          "type": "tensor",
          "shape": [
            1,
            32,
            32768
          ],
          "dtype": "torch.float32"
        },
        {
          "type": "scalar",
          "value": "None"
        },
        {
          "type": "tensor",
          "shape": [
            5
          ],
          "dtype": "torch.int64"
        },
        {
          "type": "scalar",
          "value": 128
        },
        {
          "type": "scalar",
          "value": false
        }
      ],
      "status": "pass",
      "framework": {
        "avg_latency_ms": 49.1162,
        "peak_memory_mb": 4384.02,
        "operators": {
          "aclnnAdd_AddAiCore_Add": {
            "avg_us": 2020.2493,
            "launch_count": 259
          },
          "aclnnAdds_AddAiCore_Add": {
            "avg_us": 5.1797,
            "launch_count": 2
          },
          "aclnnCat_ConcatD_ConcatD": {
            "avg_us": 2.2872,
            "launch_count": 1
          },
          "aclnnClamp_ClipByValueV2AiCore_ClipByValueV2": {
            "avg_us": 11236.8459,
            "launch_count": 777
          },
          "aclnnCumsum_CumsumAiCpu_Cumsum": {
            "avg_us": 46.6248,
            "launch_count": 1
          },
          "aclnnExp_ExpAiCore_Exp": {
            "avg_us": 752.5396,
            "launch_count": 518
          },
          "aclnnExp_SliceAiCore_Slice": {
            "avg_us": 906.2898,
            "launch_count": 259
          },
          "aclnnFloorDivides_FloorDivAiCore_FloorDiv": {
            "avg_us": 4.1124,
            "launch_count": 2
          },
          "aclnnInplaceCopy_CastAiCore_Cast": {
            "avg_us": 3314.9823,
            "launch_count": 5
          },
          "aclnnInplaceCopy_TensorMoveAiCore_TensorMove": {
            "avg_us": 28.9419,
            "launch_count": 4
          },
          "aclnnInplaceCopy_ViewCopyAiCore_ViewCopy": {
            "avg_us": 4212.0502,
            "launch_count": 518
          },
          "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike": {
            "avg_us": 609.5972,
            "launch_count": 7
          },
          "aclnnMatmul_BatchMatMulNd_BatchMatMulV2": {
            "avg_us": 4018.1465,
            "launch_count": 518
          },
          "aclnnMatmul_SliceAiCore_Slice": {
            "avg_us": 3616.6499,
            "launch_count": 518
          },
          "aclnnMatmul_TransposeAiCore_Transpose": {
            "avg_us": 3628.5626,
            "launch_count": 259
          },
          "aclnnMul_MulAiCore_Mul": {
            "avg_us": 6802.303,
            "launch_count": 518
          },
          "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum": {
            "avg_us": 2.5441,
            "launch_count": 1
          },
          "aclnnSub_SliceAiCore_Slice": {
            "avg_us": 3417.7778,
            "launch_count": 777
          },
          "aclnnSub_SubAiCore_Sub": {
            "avg_us": 4485.2353,
            "launch_count": 519
          },
          "aclnnSubs_SubAiCore_Sub": {
            "avg_us": 5.3161,
            "launch_count": 2
          }
        }
      },
      "implementation": {
        "avg_latency_ms": 8.364,
        "peak_memory_mb": 1352.01,
        "operators": {
          "_gdn_chunk_fwd_h_kernel": {
            "avg_us": 8364.0332,
            "launch_count": 1
          }
        }
      },
      "speedup_vs_torch": 5.8723,
      "error_type": null,
      "error_msg": null
    }
  ]
}