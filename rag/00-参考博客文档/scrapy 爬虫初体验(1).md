## 目录
- 前言
- scrapy 数据流
- scrapy 组件
- 爬取豆瓣电影 Top250
- 后记
- 送书后话
## 前言
为什么要学 scrapy 呢？看下图，就清楚了。很多招聘要求都有 scrapy，主要是因为 scrapy 确实很强。那到底强在哪里呢？请在文中找答案。
![](https://i-blog.csdnimg.cn/img_convert/f01da2d48787aadb9f6b0148166eaa89.webp?x-oss-process=image/format,png)

![](https://i-blog.csdnimg.cn/img_convert/44d677f4ff0a7aa645ae5fc758be0eed.webp?x-oss-process=image/format,png)
## scrapy 数据流
首先我们先来学习一下 scrapy 的工作流程。[scrapy 文档地址](https://doc.scrapy.org/en/master/topics/architecture.html#data-flow)
![](https://i-blog.csdnimg.cn/img_convert/d393e6aa0c5bab30447428965334f973.webp?x-oss-process=image/format,png)


1、爬虫引擎获得初始请求开始抓取。 
2、爬虫引擎开始请求调度程序，并准备对下一次的请求进行抓取。 
3、爬虫调度器返回下一个请求给爬虫引擎。 
4、引擎请求发送到下载器，通过下载中间件下载网络数据。 
5、一旦下载器完成页面下载，将下载结果返回给爬虫引擎。 
6、引擎将下载器的响应通过中间件返回给爬虫进行处理。 
7、爬虫处理响应，并通过中间件返回处理后的items，以及新的请求给引擎。 
8、引擎发送处理后的items到项目管道，然后把处理结果返回给调度器，调度器计划处理下一个请求抓取。 
9、重复该过程（继续步骤1），直到爬取完所有的 url 请求。

## scrapy 组件
#### 爬虫引擎
爬虫引擎负责控制各个组件之间的数据流，当某些操作触发事件后都是通过engine来处理。

#### 调度器 
调度接收来engine的请求并将请求放入队列中，并通过事件返回给engine。

#### 下载器 
通过engine请求下载网络数据并将结果响应给engine。

#### Spider 
Spider发出请求，并处理engine返回给它下载器响应数据，以items和规则内的数据请求(urls)返回给engine。

#### item pipeline
负责处理engine返回spider解析后的数据，并且将数据持久化，例如将数据存入数据库或者文件。

#### download middleware
下载中间件是engine和下载器交互组件，以钩子(插件)的形式存在，可以代替接收请求、处理数据的下载以及将结果响应给engine。

#### spider middleware 
spider中间件是engine和spider之间的交互组件，以钩子(插件)的形式存在，可以代替处理response以及返回给engine items及新的请求集。

## 爬取豆瓣电影 Top250
#### 安装
```
pip install scrapy
```
#### 初始化爬虫
```
scrapy startproject doubanTop250（项目名称）
```
目录架构如下，其中 douban_spider.py 为手动创建。
![](https://i-blog.csdnimg.cn/img_convert/d790cd6d728900edb7a566bd2d859d37.webp?x-oss-process=image/format,png)

#### 启动爬虫
```
scrapy crawl douban（后面会解释，这个 dougban 是从哪里来的，此处先留一个小坑）
```
## spider
以下代码为 douban_spider.py ，里面都有相应的注释，以方便理解
```
class RecruitSpider(scrapy.spiders.Spider):
    # 此处为上面留下的小坑，即是设置爬虫名称
    name = "douban"
    # 设置允许爬取的域名
    allowed_domains = ["douban.com"]
    # 设置起始 url
    start_urls = ["https://movie.douban.com/top250"]
    
    # 每当网页数据 download 下来，就会发送到这里进行解析
    # 然后返回一个新的链接，加入 request 队列
    def parse(self, response):
        item = Doubantop250Item()
        selector = Selector(response)
        Movies = selector.xpath('//div[@class="info"]')
        for eachMovie in Movies:
            title = eachMovie.xpath('div[@class="hd"]/a/span/text()').extract()  # 多个span标签
            fullTitle = "".join(title)
            movieInfo = eachMovie.xpath('div[@class="bd"]/p/text()').extract()
            star = eachMovie.xpath('div[@class="bd"]/div[@class="star"]/span/text()').extract()[0]
            quote = eachMovie.xpath('div[@class="bd"]/p[@class="quote"]/span/text()').extract()
            # quote 可能为空，这里进行判断一下
            if quote:
                quote = quote[0]
            else:
                quote = ''
            item['title'] = fullTitle
            item['movieInfo'] = ';'.join(movieInfo)
            item['star'] = star
            item['quote'] = quote
            yield item
        nextLink = selector.xpath('//span[@class="next"]/link/@href').extract()
        # 第10页是最后一页，没有下一页的链接
        if nextLink:
            nextLink = nextLink[0]
            yield Request(urljoin(response.url, nextLink), callback=self.parse)
```
## pipelines
每当 spider 分析完 HTML 之后，变会返回 item，传递给 item pipelines。上面代码中：
```
yield item
```
便是返回的数据。
一般 pipelines 常用于：
- 检查是否有某些字段
- 将数据存进数据库
- 数据查重
由于只是初步尝试一下 scrapy 爬虫，所以这里我没有进行修改
```
class Doubantop250Pipeline(object):
    def process_item(self, item, spider):
        return item
```
## items
定义我们需要获取的字段
```
class Doubantop250Item(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()  # 电影名字
    movieInfo = scrapy.Field()  # 电影的描述信息，包括导演、主演、电影类型等等
    star = scrapy.Field()  # 电影评分
    quote = scrapy.Field()  # 脍炙人口的一句话
    pass
```
## setting
settings.py 定义我们爬虫的各种配置，由于这里是初步了解 scrapy 故相应的介绍会在后面。
#### 启动爬虫
```
scrapy crawl douban
```
![这里没有进行详细的解析，只是展示大概数据](https://i-blog.csdnimg.cn/img_convert/b3a90c221cc254894dcd8245162748bd.webp?x-oss-process=image/format,png)


## 后记
关于豆瓣电影的小爬虫就下完了，后面会深入解析一下 scrapy 的高级用法。