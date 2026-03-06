## 目录

- 为什么要聊 SQL 注入攻击？
- 什么是 SQL 注入攻击？
- 如何进行 SQL 注入攻击？
- 如何防范？
- 常见面试题
- 瞎比比

## 为什么要聊 SQL 注入攻击？

我这人有个想法，就是不管自己跳不跳槽，每年都会调个时间去面试一下，一来可以摸摸自己的底，知道自己的价值，二来也可以知道市场的环境局势。可以更好地为自己定位，能及时查缺补漏。所以半年前我也执行了这个想法，去参加了面试。我当时就被问到了 SQL 注入攻击，你说不知道 SQL 注入吧，我又听说过，但你叫我说清楚吧，我又说不清楚，于是场面一度很尴尬。也是后面结束面试之后，查资料才搞清楚的。那么今天我们就来聊聊 SQL 注入攻击。

## 什么是 SQL 注入攻击？

首先我们得知道什么是 SQL 注入攻击，官方一点的说法是这样的：

> 所谓SQL注入，就是通过把SQL命令插入到Web表单提交或输入域名或页面请求的查询字符串，最终达到欺骗服务器执行恶意的SQL命令。

那通俗一点呢？这么来说吧，一般我们提交的表单数据（未经过滤的情况下）都会拼接到 SQL 查询语句中的，就例如：

```go
SELECT * FROM users WHERE name='zone'
```

其中 name 参数 zone 就是从表单中传过来的数据，如果传的参数不是 zone，而是一条 SQL 语句，那么就可能骗过了 SQL 数据库，从而执行了一段恶意的代码。达到了我们（程序员）意料之外的结果。

## 如何进行 SQL 注入攻击？

说了那么多，那究竟是怎么进行攻击的呢？我自己能够攻击一下自己，测试一下吗？别急，现在就来试试看。

![img](https://i-blog.csdnimg.cn/blog_migrate/c3adab6990d3b588685008e8791d2a0a.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

普通查询 - 图1



![img](https://i-blog.csdnimg.cn/blog_migrate/5beea75c3f4227ce773c2c8b9569c3d4.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

恶意查询 - 图2

普通查询中，我们传入的查询数据为 1，恶意查询中，我们传入的数据为：

```go
图 2 参数：-1 OR 1=1
```

这各个语句中 id=-1 一般为 False，而 1=1 却恒为 true，所以这个查询语句能查询到所有结果，这是与我们编程的初衷相违背的。

![img](https://i-blog.csdnimg.cn/blog_migrate/fd9c4ff08b0c53f4c8f0364f975e66ff.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

普通查询 - 图3



![img](https://i-blog.csdnimg.cn/blog_migrate/111b6022ca1759e5bf6675420d3a9bce.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

恶意查询 - 图4



![img](https://i-blog.csdnimg.cn/blog_migrate/89aeefd435e9f3397ed2334f4924d1c3.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

恶意查询 - 图5

一般我们用户登录都需要用户名和密码，如图 3，需要用户名：zone 和密码：123 方可查询相关信息。但如图 4、图 5，我的用户名参数为如下：

```go
图 4 参数：zone'#
图 5 参数（--后面有一空格）：zone'-- 
```

这两个参数，在我输入的密码不正确的情况下，也查询出了结果，是因为 SQL 语句中有两种注释，一种是： # ，另一种是：-- ，这两条查询语句正式利用了这个，导致 SQL 认为后面的语句是注释。从而，不管你输入的密码是否正确，都可以登录。这个也违背了我们编程的初衷。
 当然，SQL 注入攻击不止这些，我这里只是列举了其中的一些操作，这些操作简单，你也可以自行测试。更多的注入方式，可以自行到网络上搜索，也可以看看这篇文章：

```go
https://www.jianshu.com/p/078df7a35671
```

## 如何防范？

防范 SQL 注入攻击，我在网络上搜寻了一些方法，仅供参考建议。

- 把应用服务器的数据库权限降至最低，尽可能地减少 SQL 注入攻击带来的危害
- 避免网站打印出SQL错误信息，比如类型错误、字段不匹配等，把代码里的SQL语句暴露出来，以防止攻击者利用这些错误信息进行SQL注入。
- 对进入数据库的特殊字符（'"\尖括号&*;等）进行转义处理，或编码转换。
- 所有的查询语句建议使用数据库提供的参数化查询接口，参数化的语句使用参数而不是将用户输入变量嵌入到SQL语句中，即不要直接拼接SQL语句。
- 在测试阶段，建议使用专门的 SQL 注入检测工具进行检测。网上有很多这方面的开源工具，例如sqlmap、SQLninja等。
- 善用数据库操作库，有些库包可能已经做好了相关的防护，我们只需阅读其文档，看是否支持相应的功能即可。

## 常见面试题

- 说说什么是 SQL 注入？
- 说说 SQL 注入的危害？
- 举个 SQL 注入的栗子？
- 如何猜测、确认数据库表名？

这些问题基本在本文中都能找到答案，我也就不一一再写一般了。

## 瞎比比

ok，关于 SQL 注入攻击就说到这里吧，这个也是面试官经常会问到问题。如果你还不会，建议尽快掌握。如果哪天面试官问到你，有用到了我文章中的知识，要回来报喜呀！

**往期推荐：**

[介绍几款 Python 类型检查工具](http://mp.weixin.qq.com/s?__biz=MzAwMDI3OTc5NA%3D%3D&chksm=8d448aaeba3303b8b1a3cd6997745b6be04d1786d8e60bb86d30745fc342b11831161cc800e4&idx=1&mid=2455469875&scene=21&sn=5ed1a107423a2a888e5b30e00813fa01#wechat_redirect)

[我最近为什么不写原创](http://mp.weixin.qq.com/s?__biz=MzAwMDI3OTc5NA%3D%3D&chksm=8d448adfba3303c9d3af2e365d9fbfa5501269e8750591ef786aaf31187cd5985edd1b443a6b&idx=1&mid=2455469890&scene=21&sn=ea1583e62b7113d36b0b18da1d2c85c1#wechat_redirect)

[复联4即将上映，用Python来分析漫威中谁是赢家？](http://mp.weixin.qq.com/s?__biz=MzAwMDI3OTc5NA%3D%3D&chksm=8d448ad7ba3303c19ad78bf2c7396777d3e5ae6bafa9263e52db9ac38a68c0e395df7d5d0a16&idx=1&mid=2455469898&scene=21&sn=4e2643fcb8d6817e4e6fce0772b30dc2#wechat_redirect)

------

![img](https://i-blog.csdnimg.cn/blog_migrate/cf05370a4bdf770b4d5d54de3524164b.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑