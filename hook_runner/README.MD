runner hook学习笔记
1. 定义Hook虚拟基类，各个phase的纯虚函数定义好，每个函数体都是直接pass
2. 定义具体Hook类，构造函数统一传入一个字典接受配置，重写如before_run / after_run等阶段执行函数
3. 通过配置文件，将Hook注册进入runner
4. runner调用call_hooks函数执行hook，使用到了反射技术通过字符串寻找hook对应的方法
5. hook执行时可能需要runner内含有的一些东西，如model optim等，因此在hook before_run / after_run函数内应该传入runner，可以使用传入self来传入runner对象