# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch
import numpy as np

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 ⌘F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    print(torch.cuda.is_available())
    print(torch.__version__)
    t = torch.tensor([1,2])
    print(t)
    t1 = torch.tensor((1,2))
    print(t1)
    a = np.array((1,2,3))
    t2 = torch.tensor(a)
    print(t2)
    print(t2.dtype)
    print(t1.dtype)
    print(a.dtype)
    print(np.array((1,2)).dtype)
    print(torch.tensor([1.1,2.2]).dtype)
    print(torch.tensor([1,2]).dtype)
    print(torch.tensor([1.11, 2.22]).dtype)
    print(torch.tensor(np.array([1.11, 2.22]),dtype=torch.int16))
    print(np.array((1.1, 2.2)).dtype)

    print(torch.tensor(([1,2,3],[2,3,3],[4,5,5])).ndim)
    print(torch.tensor(([1, 2, 3], [2, 3, 3], [4, 5, 5])).shape)
    print(torch.tensor(([[1, 2, 3], [2, 3, 3], [4, 5, 5]],[[1, 2, 3], [2, 3, 3], [4, 5, 5]])).ndim)
    print(torch.tensor(([[1, 2, 3], [2, 3, 3], [4, 5, 5]], [[1, 2, 3], [2, 3, 3], [4, 5, 5]])).shape)
    print(torch.tensor(([[1, 2, 3], [2, 3, 3], [4, 5, 5]], [[1, 2, 3], [2, 3, 3], [4, 5, 5]])).numel())


    print(torch.tensor(1).ndim)
    print(torch.tensor(1).shape)

    print(torch.tensor((1,2)).ndim)

    t10 = torch.tensor(([[1, 2, 3], [2, 3, 3], [4, 5, 5]], [[1, 2, 3], [2, 3, 3], [4, 5, 5]]))

    print(t10)
    print(t10.flatten())
    print(t10.shape)
    print(t10.reshape(3,3,2))
    print(t10.shape)
    print(t10.reshape(9,2))
    print(t10.shape)

    print(torch.eye(10))

    print(torch.arange(8))

    print(torch.linspace(1,20,11))
    print(t10)
    print(t10.numpy())
    print(t10)
    t11 = torch.tensor(2)
    print(t11.item())
    print("sqlrush test1")

