---
layout: post
title: Converting ipynb to .md
subtitle : nbconvert
tags: [Google_colab, Python]
author: Huey Kim
comments : False
---


As I am working on Machine Learning by using Google Colab, 
I wanted to know whether there's a way where I can convert what I wrote in Colab file to .md file
so that I can upload them in my Github Blog.
<br><br>
While surfing the web I found about this tool called [nbconvert](https://github.com/jupyter/nbconvert).
<br><br>
However, I am currently serving in military, which prevents me to install programs excluding pre-installed ones.
<br><br>
I needed a way to use nbconvert online and figured out that I could do so by using Google Colab.
<br><br>

<b>I. Create a code block and run statement below.</b>

{% highlight python %}
!pip install nbconvert
{% endhighlight %}

<br>
<b>II. Type below</b>

{% highlight python %}
from google.colab import files
uploaded = files.upload()
{% endhighlight %}

<br>
<b>III. If you run the code block above, an upload feature will appear. Simply select the .ipynb file and upload it.</b>
<br><br>
<b>IV. Replace **YOUR_FILE** to name of your .ipynb file you intend to convert and run the code block.</b>

{% highlight python %}
!jupyter nbconvert --to markdown YOUR_FILE.ipynb
{% endhighlight %}

<br>
<b>V. If you have followed the steps above, .md file will be created. Type below to download it.</b>

{% highlight python %}
from google.colab import files
files.download('YOUR_FILE.md')
{% endhighlight %}

<br>
Done!
