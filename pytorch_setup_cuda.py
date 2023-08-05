from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#即可以使用pytorch setup.py install 也可以使用pytorch setup.py develop,前者会把库放到site-packages内，后者会把库放在
#CUDAExtension的name参数对应ed文件夹内，比如输入是packages.ops.iou3d_nms.iou3d_nms_cuda
#则会在packages/ops/iou3d_nms文件夹下创建一个iou3d_nms_cuda_xxxxx.so的库，该库则包含了定义好的所有函数。

setup(
	#1. 包的名字
	name='loopid',
	#2. 固定使用，这么写就行
	cmdclass={
		'build_ext': BuildExtension,
	},
	#3. 列表内放置自定义算子包
	ext_modules=[
		CUDAExtension(name='packages.ops.iou3d_nms.iou3d_nms_cuda',
		              #sources内填入所有的源文件路径
		              sources=['packages/ops/iou3d_nms/src/iou3d_cpu.cpp',
		                       'packages/ops/iou3d_nms/src/iou3d_api.cpp',
		                       'packages/ops/iou3d_nms/src/iou3d_nms.cpp',
		                       'packages/ops/iou3d_nms/src/iou3d_nms_kernel.cu'],
		              #编译选项，'-gencode=arch=compute_61,code=sm_61'根据硬件版本设置即可，也可以调用函数自行获取
		              extra_compile_args={'cxx': [], 'nvcc': ['-D__CUDA_NO_HALF_OPERATORS__',
		                                                      '-D__CUDA_NO_HALF_CONVERSIONS__',
		                                                      '-D__CUDA_NO_HALF2_OPERATORS__',
		                                                      '-gencode=arch=compute_61,code=sm_61']},
		              include_dirs=[],
		              define_macros=[],
		              ),

		#若有第二个包，则在此继续添加一个CUDAExtension类对象，传入的参数与第一个类似
	],
)

#src 三段式
#iou_nms.h iou_nms.cpp iou_nms_kernel.cu
#iou_nms.h 仅存在函数声明，暴露接口
#iou_nms.cpp 存在launcher函数声明，并定义iou_nms.h内的接口函数
#iou_nms_kernel.cu 定义kernel函数，定义存在于iou_nms.cpp声明的launcher函数
#至此在写一个iou_nms_api.cpp，包含iou_nms.h， 写入PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)，即可完成自定义cuda算子绑定。
