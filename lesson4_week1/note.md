# 卷积神经网络

- ##### 一步一步手搭CNN

  - 卷积模块

    - 使用0扩展矩阵边界

      防止卷积后矩阵过度缩小有利于搭建深层网络，同时可以帮助我们保留更多的边界信息，防止边界丢失。

    - 卷积窗口

      `conv_single_step()`单步卷积，使用W卷积，卷积后加上偏置b

    - 前向卷积

      在前向传播的过程中，我们将使用多种过滤器对输入的数据进行卷积操作，每个过滤器会产生一个2D的矩阵，我们可以把它们堆叠起来，于是这些2D的卷积矩阵就变成了高维的矩阵。（注意：矩阵切片操作，n_C为这一层的过滤器数量）

      ![]()

    - 反向卷积

      前向传播我们使用了卷积窗口pad，那么反向传播也需要。

      ```python
      da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:] += W[:,:,:,c] * dZ[i, h, w, c]
      dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
      db[:,:,:,c] += dZ[i,h,w,c]
      ```

  - 池化模块

    池化层会减少输入的宽度和高度，这样它会较少计算量的同时也使特征检测器对其在输入中的位置更加稳定。

    - 前向池化

      分为两种：最大池化、平均值池化

      实现类似于前向卷积，最后一步不同而已。

      ![]()

    - 创建掩码

      最大池化：从输入矩阵中创建掩码，以保存最大值的矩阵的位置。（mask中true位置是最大值位置）

      ![]()

    - 值分配

      平均池化：给定一个值，为按矩阵大小平均分配到每一个矩阵位置中。

      ![]()

    - 反向池化

      ```python
      #选择反向传播的计算方式
      if mode == "max":
      #开始切片
      	a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
      #创建掩码
      	mask = create_mask_from_window(a_prev_slice)
      ##计算dA_prev
      	dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]+= np.multiply(mask,dA[i,h,w,c])
          
      elif mode == "average":
      #获取dA的值
      	da = dA[i,h,w,c]
      #定义过滤器大小
      	shape = (f,f)
      #平均分配
      	dA_prev[i,vert_start:vert_end, horiz_start:horiz_end ,c] += distribute_value(da,shape)
      ```

- ##### CNN简单应用