ZoeDepth(
  (core): DepthAnythingCore(
    (core): DPT_DINOv2(
      (pretrained): DinoVisionTransformer(
        (patch_embed): PatchEmbed(
          (proj): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14))
          (norm): Identity()
        )
        (blocks): ModuleList(
          (0-23): 24 x NestedTensorBlock(
            (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (attn): MemEffAttention(
              (qkv): Linear(in_features=1024, out_features=3072, bias=True)
              (attn_drop): Dropout(p=0.0, inplace=False)
              (proj): Linear(in_features=1024, out_features=1024, bias=True)
              (proj_drop): Dropout(p=0.0, inplace=False)
            )
            (ls1): LayerScale()
            (drop_path1): Identity()
            (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
            (mlp): Mlp(
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (act): GELU(approximate='none')
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
              (drop): Dropout(p=0.0, inplace=False)
            )
            (ls2): LayerScale()
            (drop_path2): Identity()
          )
        )
        (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
        (head): Identity()
      )

      
      (depth_head): DPTHead(
        (projects): ModuleList(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
          (2-3): 2 x Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        )
        (resize_layers): ModuleList(
          (0): ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(4, 4))
          (1): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
          (2): Identity()
          (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        (scratch): Module(
          (layer1_rn): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (layer2_rn): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (layer3_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (layer4_rn): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (refinenet1): FeatureFusionBlock(
            (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (resConfUnit1): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (resConfUnit2): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (refinenet2): FeatureFusionBlock(
            (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (resConfUnit1): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (resConfUnit2): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (refinenet3): FeatureFusionBlock(
            (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (resConfUnit1): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (resConfUnit2): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (refinenet4): FeatureFusionBlock(
            (out_conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (resConfUnit1): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (resConfUnit2): ResidualConvUnit(
              (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              (activation): ReLU()
              (skip_add): FloatFunctional(
                (activation_post_process): Identity()
              )
            )
            (skip_add): FloatFunctional(
              (activation_post_process): Identity()
            )
          )
          (output_conv1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (output_conv2): Sequential(
            (0): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
            (3): ReLU(inplace=True)
            (4): Identity()
          )
        )
      )
    )
  )
  (conv2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
  (seed_bin_regressor): SeedBinRegressorUnnormed(
    (_net): Sequential(
      (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      (3): Softplus(beta=1.0, threshold=20.0)
    )
  )
  (seed_projector): Projector(
    (_net): Sequential(
      (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (projectors): ModuleList(
    (0-3): 4 x Projector(
      (_net): Sequential(
        (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      )
    )
  )
  (attractors): ModuleList(
    (0): AttractorLayerUnnormed(
      (_net): Sequential(
        (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
        (3): Softplus(beta=1.0, threshold=20.0)
      )
    )
    (1): AttractorLayerUnnormed(
      (_net): Sequential(
        (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(128, 8, kernel_size=(1, 1), stride=(1, 1))
        (3): Softplus(beta=1.0, threshold=20.0)
      )
    )
    (2): AttractorLayerUnnormed(
      (_net): Sequential(
        (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(128, 4, kernel_size=(1, 1), stride=(1, 1))
        (3): Softplus(beta=1.0, threshold=20.0)
      )
    )
    (3): AttractorLayerUnnormed(
      (_net): Sequential(
        (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
        (3): Softplus(beta=1.0, threshold=20.0)
      )
    )
  )
  (conditional_log_binomial): ConditionalLogBinomial(
    (log_binomial_transform): LogBinomial()
    (mlp): Sequential(
      (0): Conv2d(161, 80, kernel_size=(1, 1), stride=(1, 1))
      (1): GELU(approximate='none')
      (2): Conv2d(80, 4, kernel_size=(1, 1), stride=(1, 1))
      (3): Softplus(beta=1.0, threshold=20.0)
    )
  )
)
