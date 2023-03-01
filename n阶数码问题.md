<div class="cover" style="page-break-after:always;font-family:方正公文仿宋;width:100%;height:100%;border:none;margin: 0 auto;text-align:center;">
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:20%;">
        </br>
        <img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/%E5%8D%97%E5%BC%80%E5%A4%A7%E5%AD%A6logo.jpeg" alt="校名" style="width:80%;"/>
    </div>
    </br></br></br></br></br>
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:30%;">
        <img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/%E5%8D%97%E5%BC%80%E5%A4%A7%E5%AD%A6%E6%A0%A1%E5%BE%BDlogo.jpg" alt="校徽" style="width:50%;"/>
	</div>
    </br></br></br></br></br></br></br></br>
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">《n阶数码问题》</span>
    <p style="text-align:center;font-size:14pt;margin: 0 auto">人工智能导论</p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 基于搜索策略的八数码问题（拓展至n）</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 周三下午</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">授课教师</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">张玉志 </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">姓　　名</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 张怡桢</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">学　　号</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">2013747 </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">年　　级</td>
    		<td style="width:%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 2020级本科生</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">日　　期</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">2022/10/30</td>     </tr>
    </tbody>              
    </table>
</div>



<!-- 注释语句：导出PDF时会在这里分页 -->

# 基于搜索策略的八数码问题（拓展至n）



<center><div style='height:2mm;'></div><div style="font-family:华文楷体;font-size:14pt;">张怡桢，2013747</div></center>
<center><span style="font-family:华文楷体;font-size:9pt;line-height:9mm">南开大学软件学院</span>
</center>
<div>
<div style="width:52px;float:left; font-family:方正公文黑体;">摘　要：</div> 
<div style="overflow:hidden; font-family:华文楷体;">  八数码问题也称为九宫问题。在3×3的棋盘，摆有八个棋子，每个棋子上标有1至8的某一数字，不同棋子上标的数字不相同。棋盘上还有一个空格，与空格相邻的棋子可以移到空格中。要求解决的问题是：给出一个初始状态和一个目标状态，找出一种从初始转变成目标状态的移动棋子步数最少的移动步骤。
所谓问题的一个状态就是棋子在棋盘上的一种摆法。棋子移动后，状态就会发生改变。解八数码问题实际上就是找出从初始状态到达目标状态所经过的一系列中间过渡状态。
八数码问题一般使用搜索法来解。
  </div>
</div>

<div>
<div style="width:52px;float:left; font-family:方正公文黑体;">关键词：</div> 
<div style="overflow:hidden; font-family:华文楷体;">搜索 算法 八数码问题</div>
</div>


## 实验介绍

### 实验要求

1. 动手编程，实现一个搜索解决8数码（15数码问题）
2. 显示搜索了多少节点，搜索了多少时间，可以显示内存占用的情况
3. 基本摸清楚，不同的策略的差异
4. 先从8数码做起，考虑要扩展到15数码，程序可扩展性易改造的问题
5. 找出八数码问题最长的一个搜索链

### 实验内容

1. 宽度优先搜索
2. 深度优先搜索
3. 启发式搜索，希望给出至少两个启发式函数

## n数码有无解

八数码问题的有解无解的结论：

一个状态表示成一维的形式，求出除0之外所有数字的逆序数之和，也就是每个数字前面比它大的数字的个数的和，称为这个状态的逆序。

若两个状态的逆序奇偶性相同，则可相互到达，否则不可相互到达。

由于原始状态的逆序为0（偶数），则逆序为偶数的状态有解。

也就是说，逆序的奇偶将所有的状态分为了两个等价类，同一个等价类中的状态都可相互到达。

简要说明一下：当左右移动空格时，逆序不变。当上下移动空格时，相当于将一个数字向前（或向后）移动两格，跳过的这两个数字要么都比它大（小），逆序可能±2；要么一个较大一个较小，逆序不变。所以可得结论：只要是相互可达的两个状态，它们的逆序奇偶性相同。

N×N的棋盘，N为奇数时，与八数码问题相同。逆序奇偶同性可互达

N为偶数时，空格每上下移动一次，奇偶性改变。称空格位置所在的行到目标空格所在的行步数为空格的距离（不计左右距离），若两个状态的可相互到达，则有，两个状态的逆序奇偶性相同且空格距离为偶数，或者，逆序奇偶性不同且空格距离为奇数数。否则不能。

也就是说，当此表达式成立时，两个状态可相互到达：(状态1奇偶性 == 状态2奇偶性) == (空格距离%2==0)。

**结论**

在算N数码的逆序数时，不把0算入在内；

当N为奇数时， 当两个N数码的逆序数奇偶性相同时，可以互达，否则不行；

当N为偶数时，当两个N数码的逆序数奇偶性相同的话，那么两个N数码中的0所在行的差值 k，k也必须是偶数时，才能互达；当两个N数码的逆序数奇偶性不同时，那么两个N数码中的0所在行的差值 k，k也必须是奇数时，才能互达；

相关的判断代码如下

```python
# 计算逆序数之和
def N(nums):
    N = 0
    temp = sum(nums, [])
    # print(temp)
    for i in range(len(temp)):
        if (temp[i] != 0):
            for j in range(i):
                if (temp[j] > temp[i]):
                    N += 1
    return N


# 根据逆序数之和判断所给八数码是否可解
def judge(src, target):
    N1 = N(src)
    N2 = N(target)
    # n 对于n阶的数码问题，判断是否有解
    if n % 2 == 1:
        if N1 % 2 == N2 % 2:
            return True
        else:
            return False
    else:
        if N1 % 2 == N2 % 2:
            return abs(g.zero[0] - g.zero_target[0]) % 2 == 0
        else:
            return abs(g.zero[0] - g.zero_target[0]) % 2 == 1

```



## 模型的实现

### n数码模型抽象棋盘类

设计一个类，接受n阶像棋盘一样的矩阵

定义了移动move，分为4个方向up，down，left，right。

定义了一些功能型函数，可以找到对应元素在表格中的位置，存储找到的解的全程变化状态等。

```python
# 棋盘的类，实现移动和扩展状态
class grid:
    def __init__(self, stat, target):
        self.pre = None
        # 目标状态
        self.target = target
        # stat是一个二维列表
        self.stat = stat
        self.find0_stat()
        self.find0_target()
        self.update()


    # 以三行三列的形式输出当前状态
    def see(self):
        for i in range(n):
            print(self.stat[i])
        print("F=", self.F, "G=", self.G, "H=", self.H)
        print("-" * 10)

    def seeCommon(self):
        print("depth:", self.G)
        for i in range(n):
            print(self.stat[i])
        print("-" * 10)

    # 查看找到的解是如何从头移动的
    def seeAns(self):
        ans = []
        ans.append(self)
        p = self.pre
        while (p):
            ans.append(p)
            p = p.pre
        ans.reverse()
        for i in ans:
            i.see()

    def seeAnsCommon(self):
        ans = []
        ans.append(self)
        p = self.pre
        while (p):
            ans.append(p)
            p = p.pre
        ans.reverse()
        for i in ans:
            i.seeCommon()

    # 找到数字x的位置
    def findx_stat(self, x):
        for i in range(n):
            if x in self.stat[i]:
                j = self.stat[i].index(x)
                return [i, j]

    def findx_target(self, x):
        for i in range(n):
            if x in self.target[i]:
                j = self.target[i].index(x)
                return [i, j]

    # 找到0，也就是空白格的位置
    def find0_stat(self):
        self.zero = self.findx_stat(0)

    def find0_target(self):
        self.zero_target = self.findx_target(0)

    # 扩展当前状态，也就是上下左右移动。返回的是一个状态列表，也就是包含stat的列表
    def expand(self):
        i = self.zero[0]
        j = self.zero[1]
        gridList = []
        if j > 0:
            gridList.append(self.left())
        if i > 0:
            gridList.append(self.up())
        if i < n - 1:
            gridList.append(self.down())
        if j < n - 1:
            gridList.append(self.right())
        return gridList

    # deepcopy多维列表的复制，防止指针赋值将原列表改变
    # move只能移动行或列，即row和col必有一个为0
    # 向某个方向移动
    def move(self, row, col):
        newStat = copy.deepcopy(self.stat)
        tmp = self.stat[self.zero[0] + row][self.zero[1] + col]
        newStat[self.zero[0]][self.zero[1]] = tmp
        newStat[self.zero[0] + row][self.zero[1] + col] = 0
        return newStat

    def up(self):
        return self.move(-1, 0)

    def down(self):
        return self.move(1, 0)

    def left(self):
        return self.move(0, -1)

    def right(self):
        return self.move(0, 1)


```



### 15数码拓展

包含从8数码拓展到n数码的全局变量

我在grid类中实现的是n阶棋盘格的数码问题，不仅仅是针对8数码的，可以根据矩阵的阶n进行拓展，自己设置stat以及目标形式target。

8数码的全局设置：

```python
n = 3
stat = [[2, 8, 3],
        [1, 0, 4], 
        [7, 6, 5]]
target = [[1, 2, 3],
          [8, 0, 4], 
          [7, 6, 5]]
g = grid(stat, target)
```



15数码的全局设置：

```python
n = 4
stat = [[0, 2, 3, 4],
        [1, 6, 7, 8],
        [5, 10, 11, 12],
        [9, 13, 14, 15]]
target = [[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 0]]
g = grid(stat, target)
```



## 算法

### BFS（宽度优先搜索）：

广度优先搜索算法（英语：Breadth-First-Search，缩写为BFS），是一种图形搜索算法。简单的说，BFS是从根节点开始，沿着树的宽度遍历树的节点。如果所有节点均被访问，则算法中止。BFS是一种盲目搜索法，目的是系统地展开并检查图中的所有节点，以找寻结果。

1. BFS会先访问根节点的所有邻居节点，然后再依次访问邻居节点的邻居节点，直到所有节点都访问完毕。
2. 在具体的实现中，使用open和closed两个表，
   1. open是一个队列，每次对open进行一次出队操作（并放入closed中），并将其邻居节点进行入队操作。直到队列为空时即完成了所有节点的遍历。
   2. closed表在遍历树时其实没有用，因为子节点只能从父节点到达。但在进行图的遍历时，一个节点可能会由多个节点到达，所以此时为了防止重复遍历应该每次都检查下一个节点是否已经在closed中了。

由于BFS是一层一层找的，所以一定能找到解，并且是最优解。虽然能找到最优解，但它的盲目性依然是一个很大的缺点。从上面的遍历树状图中，每一层都比上一层元素更多，且是近似于指数型的增长。也就是说，深度每增加一，这一层的搜索速度就要增加很多。

```python
def BFS():
    if (judge(g.stat, g.target) != True):
        print("所给八数码无解，请检查输入")
        exit(1)

    visited = []
    queue = [g]
    time = 0
    while (queue):
        time += 1
        v = queue.pop(0)
        # 判断是否找到解
        if (v.H == 0):
            print("found and times:", time, "moves:", v.G)
            # 查看找到的解是如何从头移动的
            v.seeAnsCommon()
            break
        else:
            # 对当前状态进行扩展
            visited.append(v.stat)
            expandStats = v.expand()
            for stat in expandStats:
                tmpG = grid(stat, target)
                tmpG.pre = v
                tmpG.update()
                if (stat not in visited):
                    queue.append(tmpG)

```



### DFS（深度优先搜索）：

深度优先搜索算法（Depth-First-Search，DFS）是一种用于遍历或搜索树或图的算法。沿着树的深度遍历树的节点，尽可能深的搜索树的分支。当节点v的所在边都己被探寻过，搜索将回溯到发现节点v的那条边的起始节点。这一过程一直进行到已发现从源节点可达的所有节点为止。如果还存在未被发现的节点，则选择其中一个作为源节点并重复以上过程，整个进程反复进行直到所有节点都被访问为止。属于盲目搜索。

在该算法我限制了深度优先搜索的层数界限，为什么要设置深度界限？

因为理论上我们只需要一条路就可以找到解，只要不停地向下扩展就可以了。而这样做的缺点是会绕远路，也许第一条路找到第100层才找到解，但第二条路找两层就能找到解。从DFS的原理出发，我们不难看出这一点。还有一个问题是其状态数太多了，在不设置深度界限的情况下经常出现即使程序的栈满了依然没有找到解的情况。所以理论只是理论，在坚持"一条道走到黑"时，很可能因为程序"爆栈"而走到了黑还是没有找到解。

由于需要设置深度界限，每条路都会在深度界限处截至，而如果所给的八数码的最优解大于深度界限，就会出现遍历完所有情况都找不解。而在事先不知道最优解的深度的情况下这个深度界限很难确定，设置大了会增大搜索时间，设置小了会找不到解。这也是DFS的一个缺点。

DFS不一定能找到最优解。因为深度界限的原因，找到的解可能在最优解和深度界限之间。

```python
def DFS():
    # 判断所给的八数码受否有解
    if (judge(g.stat, g.target) != True):
        print("所给八数码无解，请检查输入")
        exit(1)
    # visited储存的是已经扩展过的节点
    visited = []

    # 用递归的方式进行DFS遍历
    def DFSUtil(v, visited):
        global time
        # 判断是否达到深度界限
        if (v.G > 7):
            return
        time += 1
        # 判断是否已经找到解
        if (v.H == 0):
            print("found and times", time, "moves:", v.G)
            v.seeAnsCommon()
            exit(1)

        # 对当前节点进行扩展
        visited.append(v.stat)
        expandStats = v.expand()
        w = []
        for stat in expandStats:
            tmpG = grid(stat, target)
            tmpG.pre = v
            tmpG.update()
            if (stat not in visited):
                w.append(tmpG)
        for vadj in w:
            DFSUtil(vadj, visited)
        # visited查重只对一条路，不是全局的，每条路开始时都为空
        # 因为如果全局查重，会导致例如某条路在第100层找到的状态，在另一条路是第2层找到也会被当做重复
        # 进而导致明明可能会找到解的路被放弃
        visited.pop()

    DFSUtil(g, visited)
    # 如果找到解程序会在中途退出，走到下面这一步证明没有找到解
    print("在当前深度下没有找到解，请尝试增加搜索深度")

```



### 启发式搜索A*算法：

启发式搜索（Heuristic Search）也被称为有信息搜索。和无信息搜索相反，这类的搜索算法的决策依赖于一定的信息，利用问题拥有的启发信息来引导搜索，达到减少搜索范围、降低问题复杂度的目的。贪心最优搜索和A/A*搜索都是启发式搜索。

Astar算法是一种求解最短路径最有效的直接搜索方法，也是许多其他问题的常用启发式算法。

1. 它的启发函数为$$f(n)=g(n)+h(n)$$,其中，
   1. $$f(n)$$ 是从初始状态经由状态n到目标状态的代价估计，
   2. $$g(n)$$ 是在状态空间中从初始状态到状态n的实际代价，
   3. $$h(n) $$是从状态n到目标状态的最佳路径的估计代价。

h(n)是启发函数中很重要的一项，它是对当前状态到目标状态的最小代价h*(n)的一种估计，且需要满足

​															$$h(n)<=h*(n) $$

也就是说$$h(n)$$是$$h*(n)$$的下界，这一要求保证了Astar算法能够找到最优解。

这一点很容易想清楚，因为满足了这一条件后，启发函数的值总是小于等于最优解的代价值，也就是说寻找过程是在朝着一个可能是最优解的方向或者是比最优解更小的方向移动，如果启发函数值恰好等于实际最优解代价值，那么搜索算法在一直尝试逼近最优解的过程中会找到最优解；如果启发函数值比最优解的代价要低，虽然无法达到，但是因为方向一致，会在搜索过程中发现最优解。

h是由我们自己设计的，h函数设计的好坏决定了Astar算法的效率。h值越大，算法运行越快。但是在设计评估函数时，需要注意一个很重要的性质：评估函数的值一定要小于等于实际当前状态到目标状态的代价。否则虽然程序运行速度加快，但是可能在搜索过程中漏掉了最优解。相对的，只要评估函数的值小于等于实际当前状态到目标状态的代价，就一定能找到最优解。

### A star的实现

```python
# Astar算法的函数
def Astar():
    # open和closed存的是grid对象
    open = []
    closed = []
    # 初始化状态

    # 检查是否有解
    if (judge(g.stat, g.target) != True):
        print("所给八数码无解，请检查输入")
        exit(1)

    open.append(g)
    # time变量用于记录遍历次数
    time = 0
    # 当open表非空时进行遍历
    while (open):
        # 根据启发函数值对open进行排序，默认升序
        open.sort(key=lambda G: G.F)
        # 找出启发函数值最小的进行扩展
        minFStat = open[0]
        # 检查是否找到解，如果找到则从头输出移动步骤
        if (minFStat.H == 0):
            print("found and times:", time, "moves:", minFStat.G)
            minFStat.seeAns()
            break

        # 走到这里证明还没有找到解，对启发函数值最小的进行扩展
        open.pop(0)
        closed.append(minFStat)
        expandStats = minFStat.expand()
        # 遍历扩展出来的状态
        for stat in expandStats:
            # 将扩展出来的状态（二维列表）实例化为grid对象
            tmpG = grid(stat, target)
            # 指针指向父节点
            tmpG.pre = minFStat
            # 初始化时没有pre，所以G初始化时都是0
            # 在设置pre之后应该更新G和F
            tmpG.update()
            # 查看扩展出的状态是否已经存在与open或closed中
            findstat = isin(tmpG, open)
            findstat2 = isin(tmpG, closed)
            # 在closed中,判断是否更新
            if (findstat2[0] == True and tmpG.F < closed[findstat2[1]].F):
                closed[findstat2[1]] = tmpG
                open.append(tmpG)
                time += 1
            # 在open中，判断是否更新
            if (findstat[0] == True and tmpG.F < open[findstat[1]].F):
                open[findstat[1]] = tmpG
                time += 1
            # tmpG状态不在open中，也不在closed中
            if (findstat[0] == False and findstat2[0] == False):
                open.append(tmpG)
                time += 1

```



### 启发函数

在该算法中，我们在grid类中定义启发函数，并在更新位置信息时更新启发函数的值。

对于启发类的算法而言：H是和目标状态距离之和，G是深度，也就是走的步数，F是启发函数，F=G+H。

其中我们在fH定义启发函数的类型。

```python
# 更新启发函数的相关信息
    def update(self):
        self.fH()
        self.fG()
        self.fF()

    # G是深度，也就是走的步数
    def fG(self):
        if (self.pre != None):
            self.G = self.pre.G + 1
        else:
            self.G = 0

    # H是和目标状态距离之和
    def fH(self):
        pass

    # F是启发函数，F=G+H
    def fF(self):
        self.F = self.G + self.H
```

对于上面的fH函数，我们可以使用不同的距离定义，得到不同的启发函数。

#### 哈曼顿距离之和

```python
# H是和目标状态距离之和
  def fH(self):
        self.H = 0
        for i in range(n):
            for j in range(n):
                targetX = self.target[i][j]
                nowP = self.findx_stat(targetX)
                # 曼哈顿距离之和
                self.H += abs(nowP[0] - i) + abs(nowP[1] - j)
```



#### 欧氏距离

```python
# H是和目标状态距离之和
    def fH(self):
        self.H = 0
        for i in range(n):
            for j in range(n):
                targetX = self.target[i][j]
                nowP = self.findx_stat(targetX)
                # 欧式距离之和
                self.H += sqrt((nowP[0] - i) ** 2 + (nowP[1] - j) ** 2)
```

## 实现结果

### 8数码

原形式为stat，目标形式为target

```python
n = 3
stat = [[0, 2, 3],
        [1, 4, 6], 
        [7, 5, 8]]
target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
```



#### DFS

```python
found and times 56 moves: 4
depth: 0
[0, 2, 3]
[1, 4, 6]
[7, 5, 8]
----------
depth: 1
[1, 2, 3]
[0, 4, 6]
[7, 5, 8]
----------
depth: 2
[1, 2, 3]
[4, 0, 6]
[7, 5, 8]
----------
depth: 3
[1, 2, 3]
[4, 5, 6]
[7, 0, 8]
----------
depth: 4
[1, 2, 3]
[4, 5, 6]
[7, 8, 0]
----------
DFS运行时间：9.549856185913086毫秒
```



#### BFS

```python
found and times: 21 moves: 4
depth: 0
[0, 2, 3]
[1, 4, 6]
[7, 5, 8]
----------
depth: 1
[1, 2, 3]
[0, 4, 6]
[7, 5, 8]
----------
depth: 2
[1, 2, 3]
[4, 0, 6]
[7, 5, 8]
----------
depth: 3
[1, 2, 3]
[4, 5, 6]
[7, 0, 8]
----------
depth: 4
[1, 2, 3]
[4, 5, 6]
[7, 8, 0]
----------
BFS运行时间：1.56402587890625毫秒
```



#### Astar_哈曼顿距离之和

```python
found and times: 9 moves: 4
[0, 2, 3]
[1, 4, 6]
[7, 5, 8]
F= 8 G= 0 H= 8
----------
[1, 2, 3]
[0, 4, 6]
[7, 5, 8]
F= 7 G= 1 H= 6
----------
[1, 2, 3]
[4, 0, 6]
[7, 5, 8]
F= 6 G= 2 H= 4
----------
[1, 2, 3]
[4, 5, 6]
[7, 0, 8]
F= 5 G= 3 H= 2
----------
[1, 2, 3]
[4, 5, 6]
[7, 8, 0]
F= 4 G= 4 H= 0
----------
Astar运行时间：0.46515464782714844毫秒

```

#### Astar_欧式距离

```python
found and times: 9 moves: 4
[0, 2, 3]
[1, 4, 6]
[7, 5, 8]
F= 6.82842712474619 G= 0 H= 6.82842712474619
----------
[1, 2, 3]
[0, 4, 6]
[7, 5, 8]
F= 6.23606797749979 G= 1 H= 5.23606797749979
----------
[1, 2, 3]
[4, 0, 6]
[7, 5, 8]
F= 5.414213562373095 G= 2 H= 3.414213562373095
----------
[1, 2, 3]
[4, 5, 6]
[7, 0, 8]
F= 5.0 G= 3 H= 2.0
----------
[1, 2, 3]
[4, 5, 6]
[7, 8, 0]
F= 4.0 G= 4 H= 0.0
----------
Astar运行时间：0.4398822784423828毫秒

```



#### 分析

| 方法                 | 所创建的结点数 | stat到target的路径结点数 | 时间/ms     |
| -------------------- | -------------- | ------------------------ | ----------- |
| DFS                  | 54             | 4                        | 9.549856186 |
| BFS                  | 21             | 4                        | 1.564025879 |
| Astar_哈曼顿距离之和 | 9              | 4                        | 0.465154648 |
| Astar_欧式距离       | 9              | 4                        | 0.439882278 |

如表格所示，对于这个简单的8数码的例子而言，从三个算法的遍历次数可以看出Astar算法更加优秀，能够更快的找到解，我选择的两种启发函数在这个例子表现差不多。



使用更加复杂的stat与target

```python
n = 3
stat = [[2, 5, 3], [7, 0, 6], [1, 8, 4]]
target = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
```



结果如下

| 方法                 | 所创建的结点数 | stat到target的路径结点数 | 时间/ms     |
| -------------------- | -------------- | ------------------------ | ----------- |
| DFS                  | 1639           | 14                       | 108.8061333 |
| BFS                  | 4599           | 14                       | 1041.503906 |
| Astar_哈曼顿距离之和 | 97             | 14                       | 5.168914795 |
| Astar_欧式距离       | 129            | 14                       | 8.717060089 |



更加复杂的模型体现启发式搜索相较于盲目搜索的优越性。





### 十五数码

| 方法                 | 所创建的结点数 | stat到target的路径结点数 | 时间/ms     |
| -------------------- | -------------- | ------------------------ | ----------- |
| DFS                  | 3953           | 6                        | 359.7319126 |
| BFS                  | 100            | 6                        | 11.08980179 |
| Astar_哈曼顿距离之和 | 11             | 6                        | 0.766038895 |
| Astar_欧式距离       | 11             | 6                        | 1.093149185 |

```python
n = 4
stat = [[0, 2, 3, 4],
        [1, 6, 7, 8],
        [5, 10, 11, 12],
        [9, 13, 14, 15]]
target = [[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 0]]
```

　　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/image-20221130214308779.png" ></td><td><img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/image-20221130214225511.png" ></td>
		</tr>
        <tr><td><strong>图   Astar_欧式距离</strong></td><td><strong>图   Astar_欧式距离</strong></td></tr>
	</tbody>
</table>



　　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/image-20221130214549551.png" ></td><td><img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/image-20221130214528812.png" ></td>
		</tr>
        <tr><td><strong>图   BFS</strong></td><td><strong>图   BFS</strong></td></tr>
	</tbody>
</table>

　　<table style="border:none;text-align:center;width:auto;margin: 0 auto;">
	<tbody>
		<tr>
			<td style="padding: 6px"><img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/image-20221130214507520.png" ></td><td><img src="https://zyzstc-1303973796.cos.ap-beijing.myqcloud.com/uPic/image-20221130214254601.png" ></td>
		</tr>
        <tr><td><strong>图   DFS</strong></td><td><strong>图   DFS</strong></td></tr>
	</tbody>
</table>





## 完整代码

```python
import copy
import math
import time


# 棋盘的类，实现移动和扩展状态
class grid:
    def __init__(self, stat, target):
        self.pre = None
        # 目标状态
        self.target = target
        # stat是一个二维列表
        self.stat = stat
        self.find0_stat()
        self.find0_target()
        self.update()

    # 更新启发函数的相关信息
    def update(self):
        self.fH()
        self.fG()
        self.fF()

    # G是深度，也就是走的步数
    def fG(self):
        if (self.pre != None):
            self.G = self.pre.G + 1
        else:
            self.G = 0

    # H是和目标状态距离之和
    def fH(self):
        self.H = 0
        for i in range(n):
            for j in range(n):
                targetX = self.target[i][j]
                nowP = self.findx_stat(targetX)
                # # 曼哈顿距离之和
                # self.H += abs(nowP[0] - i) + abs(nowP[1] - j)
                # 欧式距离之和
                self.H += math.sqrt((nowP[0] - i) ** 2 + (nowP[1] - j) ** 2)

    # F是启发函数，F=G+H
    def fF(self):
        self.F = self.G + self.H

    # 以三行三列的形式输出当前状态
    def see(self):
        for i in range(n):
            print(self.stat[i])
        print("F=", self.F, "G=", self.G, "H=", self.H)
        print("-" * 10)

    def seeCommon(self):
        print("depth:", self.G)
        for i in range(n):
            print(self.stat[i])
        print("-" * 10)

    # 查看找到的解是如何从头移动的
    def seeAns(self):
        ans = []
        ans.append(self)
        p = self.pre
        while (p):
            ans.append(p)
            p = p.pre
        ans.reverse()
        for i in ans:
            i.see()

    def seeAnsCommon(self):
        ans = []
        ans.append(self)
        p = self.pre
        while (p):
            ans.append(p)
            p = p.pre
        ans.reverse()
        for i in ans:
            i.seeCommon()

    # 找到数字x的位置
    def findx_stat(self, x):
        for i in range(n):
            if x in self.stat[i]:
                j = self.stat[i].index(x)
                return [i, j]

    def findx_target(self, x):
        for i in range(n):
            if x in self.target[i]:
                j = self.target[i].index(x)
                return [i, j]

    # 找到0，也就是空白格的位置
    def find0_stat(self):
        self.zero = self.findx_stat(0)

    def find0_target(self):
        self.zero_target = self.findx_target(0)

    # 扩展当前状态，也就是上下左右移动。返回的是一个状态列表，也就是包含stat的列表
    def expand(self):
        i = self.zero[0]
        j = self.zero[1]
        gridList = []
        if j > 0:
            gridList.append(self.left())
        if i > 0:
            gridList.append(self.up())
        if i < n - 1:
            gridList.append(self.down())
        if j < n - 1:
            gridList.append(self.right())
        return gridList

    # deepcopy多维列表的复制，防止指针赋值将原列表改变
    # move只能移动行或列，即row和col必有一个为0
    # 向某个方向移动
    def move(self, row, col):
        newStat = copy.deepcopy(self.stat)
        tmp = self.stat[self.zero[0] + row][self.zero[1] + col]
        newStat[self.zero[0]][self.zero[1]] = tmp
        newStat[self.zero[0] + row][self.zero[1] + col] = 0
        return newStat

    def up(self):
        return self.move(-1, 0)

    def down(self):
        return self.move(1, 0)

    def left(self):
        return self.move(0, -1)

    def right(self):
        return self.move(0, 1)


# 判断状态g是否在状态集合中，g是对象，gList是对象列表
# 返回的结果是一个列表，第一个值是真假，如果是真则第二个值是g在gList中的位置索引
def isin(g, gList):
    gstat = g.stat
    statList = []
    for i in gList:
        statList.append(i.stat)
    if (gstat in statList):
        res = [True, statList.index(gstat)]
    else:
        res = [False, 0]
    return res


# 计算逆序数之和
def N(nums):
    N = 0
    temp = sum(nums, [])
    # print(temp)
    for i in range(len(temp)):
        if (temp[i] != 0):
            for j in range(i):
                if (temp[j] > temp[i]):
                    N += 1
    return N


# 根据逆序数之和判断所给八数码是否可解
def judge(src, target):
    N1 = N(src)
    N2 = N(target)
    # n 对于n阶的数码问题，判断是否有解
    if n % 2 == 1:
        if N1 % 2 == N2 % 2:
            return True
        else:
            return False
    else:
        if N1 % 2 == N2 % 2:
            return abs(g.zero[0] - g.zero_target[0]) % 2 == 0
        else:
            return abs(g.zero[0] - g.zero_target[0]) % 2 == 1


# Astar算法的函数
def Astar():
    # open和closed存的是grid对象
    open = []
    closed = []
    # 初始化状态

    # 检查是否有解
    if (judge(g.stat, g.target) != True):
        print("所给八数码无解，请检查输入")
        exit(1)

    open.append(g)
    # time变量用于记录遍历次数
    founds = 0
    # 当open表非空时进行遍历
    while (open):
        # 根据启发函数值对open进行排序，默认升序
        open.sort(key=lambda G: G.F)
        # 找出启发函数值最小的进行扩展
        minFStat = open[0]
        # 检查是否找到解，如果找到则从头输出移动步骤
        if (minFStat.H == 0):
            print("found and times:", founds, "moves:", minFStat.G)
            minFStat.seeAns()
            break

        # 走到这里证明还没有找到解，对启发函数值最小的进行扩展
        open.pop(0)
        closed.append(minFStat)
        expandStats = minFStat.expand()
        # 遍历扩展出来的状态
        for stat in expandStats:
            # 将扩展出来的状态（二维列表）实例化为grid对象
            tmpG = grid(stat, target)
            # 指针指向父节点
            tmpG.pre = minFStat
            # 初始化时没有pre，所以G初始化时都是0
            # 在设置pre之后应该更新G和F
            tmpG.update()
            # 查看扩展出的状态是否已经存在与open或closed中
            findstat = isin(tmpG, open)
            findstat2 = isin(tmpG, closed)
            # 在closed中,判断是否更新
            if (findstat2[0] == True and tmpG.F < closed[findstat2[1]].F):
                closed[findstat2[1]] = tmpG
                open.append(tmpG)
                founds += 1
            # 在open中，判断是否更新
            if (findstat[0] == True and tmpG.F < open[findstat[1]].F):
                open[findstat[1]] = tmpG
                founds += 1
            # tmpG状态不在open中，也不在closed中
            if (findstat[0] == False and findstat2[0] == False):
                open.append(tmpG)
                founds += 1


def BFS():
    if (judge(g.stat, g.target) != True):
        print("所给八数码无解，请检查输入")
        exit(1)

    visited = []
    queue = [g]
    founds = 0
    while (queue):
        founds += 1
        v = queue.pop(0)
        # 判断是否找到解
        if (v.H == 0):
            print("found and times:", founds, "moves:", v.G)
            # 查看找到的解是如何从头移动的
            v.seeAnsCommon()
            break
        else:
            # 对当前状态进行扩展
            visited.append(v.stat)
            expandStats = v.expand()
            for stat in expandStats:
                tmpG = grid(stat, target)
                tmpG.pre = v
                tmpG.update()
                if (stat not in visited):
                    queue.append(tmpG)


founds = 0


# 用递归的方式进行DFS遍历
def DFSUtil(v, visited):
    global founds
    # 判断是否达到深度界限
    if v.G > 14:
        return False
    founds += 1
    # 判断是否已经找到解
    if v.H == 0:
        print("found and times", founds, "moves:", v.G)
        v.seeAnsCommon()

        return True

    # 对当前节点进行扩展
    visited.append(v.stat)
    expandStats = v.expand()
    w = []
    for stat in expandStats:
        tmpG = grid(stat, target)
        tmpG.pre = v
        tmpG.update()
        if stat not in visited:
            w.append(tmpG)
    x = False
    for vadj in w:
        x = x or DFSUtil(vadj, visited)
    # visited查重只对一条路，不是全局的，每条路开始时都为空
    # 因为如果全局查重，会导致例如某条路在第100层找到的状态，在另一条路是第2层找到也会被当做重复
    # 进而导致明明可能会找到解的路被放弃
    visited.pop()
    return x


def DFS():
    # 判断所给的八数码受否有解
    if (judge(g.stat, g.target) != True):
        print("所给八数码无解，请检查输入")
        exit(1)
    # visited储存的是已经扩展过的节点
    visited = []
    founds = 0
    # 用递归的方式进行DFS遍历

    if not DFSUtil(g, visited):
        print("在当前深度下没有找到解，请尝试增加搜索深度")


# n = 3
# stat = [[2, 5, 3], [7, 0, 6], [1, 8, 4]]
# target = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

n = 4
stat = [[0, 2, 3, 4],
        [1, 6, 7, 8],
        [5, 10, 11, 12],
        [9, 13, 14, 15]]
target = [[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
          [13, 14, 15, 0]]
g = grid(stat, target)
time_start_1 = time.time()
Astar()
time_end_1 = time.time()
print("Astar运行时间：" + str((time_end_1 - time_start_1) * 1000) + "毫秒")
time_start_2 = time.time()
BFS()
time_end_2 = time.time()
print("BFS运行时间：" + str((time_end_2 - time_start_2) * 1000) + "毫秒")

time_start_3 = time.time()
DFS()
time_end_3 = time.time()
print("DFS运行时间：" + str((time_end_3 - time_start_3) * 1000) + "毫秒")

```

