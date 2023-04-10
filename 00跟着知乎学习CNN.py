"""
1. 对数据集进行标签分类
要想搭建神经网络进行分类，或者预测的话，主要是先把数据标签搞定
数据标签怎么贴？哪些作为————对全部数据进行判断，如果存在超速、急变速、急转向的两种或两种以上情况，则定义为危险骑行

2. 构建CNN（convolutional neural network）
(1) CNNb本身就是来自于普通的神经网络，CNN的filter相当于线性的weight(w)，也就是w1*feature1 + w2*feature2+...
图像的每一个pixel就是一个feature

(2)
"""
import numpy as np
import mnist

"""
1. 卷积层
手撸CNN 参考链接 https://zhuanlan.zhihu.com/p/102119808
"""
class Conv3x3:
    # 构建一个过滤器(filter)为3 x 3 的卷积层
    # 卷积层的方向传播是CNN模型训练的核心
    def __init__(self, num_filters):
        """
        Conv3x3 类只需要一个参数：filter 个数,
        之所以在初始化的时候除以 9 是因为对于初始化的值不能太大也不能太小
        :param num_filters:
        """
        self.num_filters = num_filters  # 定义filters的数量

        # filters 是一个大小为(num_filters,3,3)的3维数组
        # 我们除于9来减小初始化值的变化情况
        # 通过randn函数生成一个num_filters * 3 * 3的随机3d数组
        self.filters = np.random.randn(num_filters, 3, 3)/9   # 定义filters

    def iterate_regions(self, image):   # 确实叫做迭代区域更好
        """
        实现具体的卷积层：通过有效填充法生成所有可能的3x3图像区域
        其中: 图像是一个二维的numpy数组
        :param image:   图像其实就是一个n x n的数据
        :return:
        """
        h, w = image.shape   # 图像的高和宽，那么图像的数据量就为 h x w
        for i in range(0, h-2):
            for j in range(w-2):   # range(w-2)  <==> range(0, w-2)
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j   # 将imregion, i, j以tuple行驶存储到迭代器中，以便以后使用

    def forward(self, input):
        """
        使用给定的输入，执行卷积层的前向传递
        返回一个尺寸为(h, w, numfilters)的3维数组
        :param input:  input为一个二维Numpy数组
        :return:
        """
        # input为image, 记输入数据
        # output为输出框架， 默认都为0， 都为1也行， 后面会覆盖？
        # input: 28x28
        # output: 26x26x8
        self.last_input = input  # 28 x 28
        h, w = input.shape    # 通过这种方式来获取h,w 也就意味着我的数据也可以输入进行
        output = np.zeros((h-2, w-2, self.num_filters))   # 生成一个0数列

        for im_region, i, j in self.iterate_regions(input): # yield关键字是一个return 一个生成器
            # 卷积运算，点乘再相加， output[i, j]为向量， 8层
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))   # 图像区域 * n个卷积核

        # 通过return 返回卷积层的输出
        return output
    def backprop(self, d_L_d_out, learn_rate):
        """
        卷积层主要是为了更新filter的权重，已经有了loss对out的梯度， 还需要获取out对于filter的权重
        卷积层的反向传播
        :param d_L_d_out: 这一层输出的损失函数的梯度
        :param learn_rate: 学习率
        :return:
        """
        # 初始化一组为0的gradient， 3x3x8
        d_L_d_filters = np.zeros(self.filters.shape)

        # im_region, 一个个3x3的小矩阵
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # 按f分层计算， 一次算一层，然后累加起来
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新filters
        self.filters -= learn_rate * d_L_d_filters

        """
        因为我们使Con3x3作为我们CNN的第一层，所以不需要返回什么东西。
        因为反向传播至此结束了
        如果卷积层上面还有其他层的话，则需要再返回一个gradient
        """
        return None




"""
2. 池化层
池化(pooling)
图片的相邻像素具有相似的值，因此卷积层中很多信息是冗余的。通过池化来减少这个影响，包含max, min, or average这几种
"""
class MaxPool2:   # 该作者采用的是最大池化层
    # 用2个大小池化来构建一个最大化池化层
    def iterate_regions(self, image):   # 迭代区域为h/2 w/2 8   这里的h,w为卷积层的输出26 x 26
        """
        生成不重叠的2x2大小的图像区域到池中
        池化层不需要训练， 因为它里面不存在任何weights，但是为了计算gradient我们仍然需要实现一个backprop()方法
        :param image: 这里的image为卷积层的输出
        :return:
        """
        # 存入上一层的一些信息
        # 存储池化层的输入参数， 26x26x8

        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2 + 2), (j*2):(j*2 + 2)]   # 池的大小是2，所以是按2来变，变化等到图像区域
                yield im_region, i, j

    def forward(self, input):
        """
        用给定的输入，来执行最大池化层的前向传播
        返回一个大小为(h/2, w/2, num_filters)的3d numpy数组
        :param input:  - 输入为大小为（h, w, numfilters）的三维数组。 卷积层的输出，池化层的输入
        :return:
        """
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0,1))   # 得到im_region区域中最大的一个数

        return output

    def backprop(self, d_L_d_out):
        """
        对于池化层来说，当input为最大时，梯度为loss对output的梯度， 当input不是最大则为0
        执行池化层的反向传播
        为这个层的输入范围一个损失函数梯度
        :param d_L_d_out: 这个层输出的损失函数梯度
        :return:
        """
        # 池化层输入数据， 26x26x8， 默认初始化为0
        d_L_d_input = np.zeros(self.last_input.shape)

        # 每一个im_region都是一个3x3x8的8层小矩阵
        # 修改max的部分，首先查找max
        for im_region, i, j in self.iterate_regions(self.last_input):   # last_input 26x26x8
            h, w, f = im_region.shape   # 3 x 3 x 8
            # 获取im_region里面最大值的索引向量， 一叠的感觉？
            amax = np.amax(im_region, axis=(0, 1))

            # 遍历整个im_region, 对于传递下去的像素点，修改gradient 为loss 对output的gradient
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # 如果这个这像素是最大值，则复制梯度给它
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
"""
3. 全连接层
激活函数 —— softmax
为了完成CNN， 还需要进行具体的预测。 通过softmax来实现， 将一组数字转换为一组概率，总和为1
softmax ---- 激活函数

这里作者用了包含10个节点的softmax层， 分别代表相应的数字，作为CNN的最后一层

最后一层为一个全连接层，只是激活函数为softmax, 经过softmax的变换，数字就是具有最高概率的节点

# 这里作者使用了交叉熵损失函数作为CNN的损失函数
"""
class Softmax:
    # 一个带有softmax激活函数的标准全连接层
    def __init__(self, input_len, nodes):
        """
        构建权重矩阵，初始化随机数不能太大，除以输入长度来减小变量初始化的值
        :param input_len: 输入层的节点个数，池化层输出拉之后
        :param nodes: 输出层的节点个数， 这里为10——数字个数， 如果是二分类就是2
        """
        # 对权重和偏置进行初始化
        self.weights = np.random.randn(input_len, nodes) / input_len   # 权重
        self.biases = np.zeros(nodes)   # 偏置

    def forward(self, input):
        """
        用给定的输入进行全连接层的前向传递
        返回一个包含各自概率值的一维数组
        :param input: 任意尺寸的任意数组
        :return:
        """
        # 记忆上一个输入的尺寸大小
        self.last_input_shape = input.shape   # 13 x 13 x 8

        # 3d ---> 1d，用来构建全链接网络
        input = input.flatten()   # 将池化层拉平

        # 记忆上一个输入的向量， 1352
        self.last_input = input

        input_len, nodes = self.weights.shape   # 获取输入长度和节点数
        # input: 13 x 13 x 8 = 1352
        # self.weights: (1352, 10)
        # 以上叉乘之后为向量， 1352个节点与对应的权重相乘，再加上bias得到输出的节点
        # totals: 向量， 10  应该是10个概率，反正对应10类数字
        totals = np.dot(input, self.weights) + self.biases

        # 记忆上一个的总的概率
        self.last_totals = totals
        # exp : 向量， 10 这个是进行e次操作的
        exp = np.exp(totals)
        return exp/np.sum(exp, axis=0)   # 得到一个概率

    def backprop(self, d_L_d_out, learn_rate):   # 反向传播是神经网络的核心，卡了hiton20年，所以要认真学
        """
        执行全连接层的反向传播
        返回这一层输入的一个损失梯度
        :param d_L_d_out:  这一层输出的损失梯度
        :param learn_rate  学习率是一个浮点数，学习率越大学习的越快，但是学习经度不高，学习率越小则学的慢但不容易发散
        :return:
        """
        # 我们知道只有d_L_d_out的单元只有一个是不会为0的
        for i, gradient in enumerate(d_L_d_out):
            # 找到label的值， 即梯度不为0的值
            if gradient == 0:
                continue

            # e^totals e的totals平方
            t_exp =  np.exp(self.last_totals)

            # 计算所有e^totals的和
            S = np.sum(t_exp)

            # out[i]对totals的梯度
            # 初始化都设置为非c的值， 再单独修改c的值
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # 计算totals对于weights/ biases / input的梯度
            # d_t_d_w的结果是softmax层的输入数据， 1352个元素的向量
            # 不是最终的结果，最终结果是2d矩阵，1352 x 10
            d_t_d_w = self.last_input
            d_t_d_b = 1
            # 不是最终的结果，最终结果是2d矩阵, 1352x10
            d_t_d_inputs = self.weights

            # 计算loss 对于 totals的梯度
            # 向量， size = 10
            d_L_d_t = gradient * d_out_d_t

            # 计算loss 对于 weights/biases/input的梯度
            # np.newaxis可以帮助一维向量变成二维矩阵
            # (1352, 1) @ (1, 10) to (1352, 10)
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]   # size = 1352 x 10
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1352， 10) @ (10, 1) to (1352, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t   # siez 为(1352, 1)

            """
            计算出gradient之后，剩下的就是训练softmax层。我们通过SGD来更新weights 和 bias
            这里一步，新增一个叫做学习率的参数  learn rate
            学习率确实像人一样，学的越快则可能就没有那么注意细节， 学的越慢则学的越细
            """
            # 更新权重 / 偏置
            # 通过learn_rate 来控制权重更新的快慢，似乎也好理解，越大更新的越快
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            # 将矩阵从1d 转为 3d   因为对input进行了flatten操作，反向传播时需要保证与原始输入具有相同结构
            # 1352 to 13 x 13 x 8
            return d_L_d_inputs.reshape(self.last_input_shape)




def forward(image, label):
    """
    前向传播的操作
    完成一个CNN和计算准确、交叉损失熵的前向传递
    :param image: 2维numpy数组
    :param label: 一个数字，代表类别
    :return:
    """
    # 将图片转化从[0,255] 转化为 [-0.5, 0.5]来化简计算
    # 事实上这是标准化的操作
    # out 为卷积层的输出， 26x26x8
    out = conv.forward((image/255) - 0.5)
    # out 为池化层输出, 13 x 13 x 8
    out = pool.forward(out)
    # out为softmax的输出，10
    out = softmax.forward(out)

    # 计算交叉熵损失函数，与准确率；np.log()是自然对数
    loss = -np.log(out[label])
    # 如果softmax输出的最大值就是label的值，表示正确， 否则错误
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

    """
    上面的值在相当于随机采集，猜中每个数的概率为10%

    因此，下面将进行训练操作
    """


    """
    训练神经网络
    训练神经网络一般包含两个阶段:
    (1) forward phase: 输入参数传递通过整个网络。
    (2) backward phase: 反向传播更新gradient 和 weight

    训练方法:
    (1) 在forward phase阶段中， 每一层都需要存储一些数据（例如输入数据，中间值等）。这些数据将会在backward phase中用到
    因此每一个backward phase 都需要在相应的forward phase之后运行

    (2) 在backward phase中，每一层都要获取gradient并且也返回gradient。获取的是loss对于该层输出的gradient梯度
    返回的是loss对于该层输入的gradient梯度（求导）
    """
    # 反馈前向传播
    # image 为输入层， 28x28
    # out 为卷积层输出， 26x26x8
    # out = conv.forward((image/255)-0.5)

def train(im, label, lr=.005):
    """
    在给定的图像和标签上完成一个完整的训练步骤
    :param im: image 一个2d的numpy array
    :param label:  标签，一个分类
    :param lr: 学习率
    :return:
    """
    # forward 前向传播
    out, loss, acc = forward(im, label)

    # 计算初始化的梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # 反向传播 ( 全连接层 - 池化层 - 卷积层
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc



if __name__ == "__main__":
    # output = np.zeros((14, 4, 8))
    # print(np.random.randn(3,1,3))
    # print(output.shape)
    # print(output.flatten())
    # 我们只用第一千个测试样例（总共有1w个）
    # 如果你感兴趣的话，可以更改使用
    """
    这里是没有搭建反向传播层的预测代码，及意味着没有训练
    """
    # test_image = mnist.test_images()[:1000]
    # test_labels = mnist.test_labels()[:1000]
    #
    # # 这里相当于实例化每一个类
    # # 每次输入输出顺序都很有意思，上一个的输出是下一个的输入
    # conv = Conv3x3(8)   # 8个卷积核   # 28 x 28 x 1 --->  26 x 26 x 8
    # pool = MaxPool2()               # 26 x 26 x 8  ---> 13 x 13 x 8
    # softmax = Softmax(13 * 13 * 8, 10)   # 13 x 13 x8 ---> 10
    # print('MNIST CNN initialized!')
    #
    # loss = 0
    # num_correct = 0
    # # enumerate 函数用来增加索引值
    # for i, (im,label) in enumerate(zip(test_image, test_labels)):   # 挨个遍历
    #     # 做一次前向传播
    #     _, l, acc = forward(im, label)
    #     loss += 1
    #     num_correct += acc
    #
    #     # 输出每一百次迭代的状态信息
    #     # 损失率也要搞为0-1
    #     # 迭代一百次，超过又重新计算？
    #     if i%100 == 99:
    #         print(
    #             '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy : %d%%' %
    #             (i + 1, loss/100, num_correct)
    #         )
    #         loss = 0
    #         num_correct = 0

    """
    训练CNN
    我们只用第一千个测试样例（总共有1w个）
    如果你感兴趣的话，可以更改使用
    """
    train_images = mnist.train_images()[:1000]
    train_labels = mnist.train_labels()[:1000]
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]

    # 实例化三个层
    conv = Conv3x3(8)
    pool = MaxPool2()
    softmax = Softmax(13 * 13 * 8, 10)   # 13* 13 * 8 -> 10

    # 训练3轮
    print("MNIST CNN initialized!")

    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))

        # 打乱训练数据
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        # 训练
        loss = 0
        num_correct = 0

        # i: index
        # im: image
        # label: label
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i>0 and i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy : %d%%' %
                    (i + 1, loss/100, num_correct)
                )
                loss = 0
                num_correct = 0

            l, acc = train(im, label)
            loss += l
            num_correct += acc

# Test the CNN
print('\n---Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct +=acc

num_tests = len(test_images)
print('Test Loss', loss / num_tests)
print('Test Accuracy', num_correct / num_tests)