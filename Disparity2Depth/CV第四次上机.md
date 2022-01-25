# CV第四次上机——利用双目图像计算深度

<center><font face="微软雅黑"><font size="4">19030100075 唐子辰</font></font></center>

## 一、实验目的：

- **利用双目图像计算深度**

## 二、实验环境：

- **Win 10 + Matlab R2018a**

## 三、实验理论——双目立体匹配获取深度图

&emsp;&emsp;双目立体匹配一直是双目视觉的研究热点，双目相机拍摄同一场景的左、右两幅视点图像，运用立体匹配匹配算法获取视差图，进而获取深度图。而深度图的应用范围非常广泛，由于其能够记录场景中物体距离摄像机的距离，可以用以测量、三维重建、以及虚拟视点的合成等。

&emsp;&emsp;若想自己尝试拍摄双目图片进行立体匹配，获取深度图，进而进行三维重建等操作，要做的工作可以使用以下几个步骤简要概括：

- **相机标定（内参、外参）**
- **双目图像的校正（畸变校正、立体校正）**
- **利用立体匹配算法获取视差图**
- **利用视差图求解深度图**
- **利用视差图或者深度图进行虚拟视点的合成**

&emsp;&emsp;对于本实验而言，我们已经获得了老师给出的相机标定后的相关参数，且双目图像已经经过了矫正，因此我们可以直接在这两幅图像的基础上调用Matlab的disparity()函数计算视差图，最后利用课上讲的视差图转深度图的公式求解出深度图。

**&emsp;&emsp;通过视差图计算深度图的方法如下：**

<center>
    <img src="https://s3.bmp.ovh/imgs/2021/12/63530dfbff819c9c.png"width=550/>
</center>

&emsp;&emsp;在$△PO_LO_R$中，由三角形相似定理可得：
$$
\frac{b}{Z} = \frac{\overline{P_LP_R}}{Z-f}
$$
&emsp;&emsp;不难看出：
$$
\overline{P_LP_R} = b - [(X_L - \frac{L}{2}) + (\frac{L}{2} - X_R)]
$$
&emsp;&emsp;因此有：
$$
\frac{b}{Z} = \frac{b - [(X_L - \frac{L}{2}) + (\frac{L}{2} - X_R)]}{Z-f}
$$
&emsp;&emsp;解之，得：
$$
Z = \frac{f · b}{X_L-X_R}=\frac{f · b}{D}
$$
&emsp;&emsp;其中$D=X_L-X_R$为视差(Disparity)。

## 四、实验步骤

&emsp;&emsp;1. 在Matlab中导入原始图像；

&emsp;&emsp;2. 调用img2gray()函数求原始图像的灰度图；

&emsp;&emsp;3. 调用disparity()函数，输入两幅图像的灰度图以及各类参数值，输出为两幅图像的视差图（在此过程中要注意手动调节各参数，保证输出的视差图达到一个比较好的效果）；

&emsp;&emsp;4. 利用color jet将视差图映射到彩色视觉空间进行可视化；

&emsp;&emsp;5. 根据给出的相机标定参数求出归一化焦距$f_x=\frac{f}{d_x}$（$f$单位为毫米，$d_x$是$x$方向上的像素长度）和基线距离$b$（单位为毫米）；

&emsp;&emsp;6. 根据公式$Z =\frac{f · b}{D}$求解深度图，此时要注意将深度图的精度转换为uint16，即16位深度格式（uint8只有8位，最大只能达到255，不足以反映出本次实验图像的深度差异，而uint16最大值是65535，单位是毫米，则最多能反映出65米左右的深度信息）。

## 五、实验结果

&emsp;&emsp;1. disparity()函数的调参结果：

&emsp;&emsp;disp_img = disparity(img1_gray, img2_gray, 'BlockSize', 15, 'DisparityRange', [0,80], 'ContrastThreshold', 0.5, 'UniquenessThreshold', 45);

**&emsp;&emsp;参数值设定如下：**

**&emsp;&emsp;Method =  'SemiGlobal'**，即采用半全局匹配算法。

**&emsp;&emsp;BlockSize =  15**，块大小对结果似乎没有影响。

**&emsp;&emsp;DisparityRange = [0,80]**，视差范围区间的上界值取64或80较为合适。

**&emsp;&emsp;ContrastThreshold = 0.5**，对比度阈值似乎对结果没有影响。

**&emsp;&emsp;UniquenessThreshold = 45**，唯一性的最小值。

**&emsp;&emsp;DistanceThreshold = 400**，从左到右图像检查的最大距离，对结果无影响。

&emsp;&emsp;2. 视差图求解结果：

<center>
    <img src="https://s3.bmp.ovh/imgs/2021/12/394b00383afa2a55.bmp"/>
</center>

&emsp;&emsp;3. 深度图求解结果：

<center>
    <img src="https://s3.bmp.ovh/imgs/2021/12/48b5ac99d884fef9.png" width=500/>
</center>

&emsp;&emsp;上述结果是采用SemiGlobal半全局匹配算法求解出来的视差图，所有参数都基本上调整到该算法下的最优了，但是可以发现，虽然道路边缘、道路中间的白线以及两辆车的深度信息都被提取出来了，但是道路中间包括两侧还是有很多杂乱的点块状区域出现在深度图中，而且远处的山脉的轮廓也丢失了，仅显示一片黑，效果不是太好。

------

**&emsp;&emsp;为了对实验结果进行进一步优化改进，我尝试采用disparity()函数中允许使用的另一个匹配算法，即BlockMatching算法，并在此基础上重新反复调整disparity()函数的其他相关参数，最终求出的深度图可以达到一个比较好的效果。**

**&emsp;&emsp;新的函数参数设定如下：**

**&emsp;&emsp;Method =  'BlockMatching'**，即采用块匹配算法。

**&emsp;&emsp;BlockSize =  255**，块大小若太小，会导致视差图及深度图中出现一些噪声点块，该值越大，噪声点块越少。

**&emsp;&emsp;DisparityRange = [0,80]**，视差范围区间的上界值取64或80较为合适。

**&emsp;&emsp;ContrastThreshold = 1**，对比度阈值若太小（比如0.2）会导致深度图有大片白色，无法显示深度信息，该值越大，效果越好。

**&emsp;&emsp;UniquenessThreshold = 55**，唯一性的最小值。该值太小会导致图像两侧一些非重点区域连在一起，太大会导致丢失深度信息。该值比较合适的取值区间为40~60。

**&emsp;&emsp;DistanceThreshold = 400**，从左到右图像检查的最大距离，对结果无影响。

**&emsp;&emsp;优化后新的视差图：**

<center>
    <img src="https://s3.bmp.ovh/imgs/2021/12/c242a95fc086b897.bmp"/>
</center>
**&emsp;&emsp;优化后新的深度图：**

<center>
    <img src="https://s3.bmp.ovh/imgs/2021/12/6cec4405f14bf387.png"width=500/>
</center>


&emsp;&emsp;观察上图可见，当我把disparity()函数采用的匹配算法更换为块匹配BlockMatching并在此基础上重新调参后，视差图及深度图的视觉效果可以达到一个比较好的状态。道路的轮廓线、车辆的深度信息都可以被较好地提取出来，离相机较远的背景山脉的轮廓也可以被检测出来。除此之外，相比上面那张旧的深度图而言，优化后的深度图的道路两侧及中间的噪声点块区域均可以被基本消除，仅保留图像重要区域的深度信息，这也是我们所需要的信息。

## 六、心得体会

&emsp;&emsp;通过本次实验，我们熟悉了利用双目图像计算深度的方法，并通过编程实现了利用实际拍摄的双目图片求解深度图，加深了我们对于计算机视觉及双目立体视觉理论知识中的视差图及深度图的理解。在本次实验中，我一开始在求解视差图时采用了SemiGlobal的匹配算法，后来发现无论怎么调参，似乎深度图上都会带有一些杂质区域，效果不是太好；后来我把SemiGlobal改成了块匹配BlockMatching并在BlockMatching的基础上重新调参，最终才使深度图的视觉效果达到了一个比较好的状态。

