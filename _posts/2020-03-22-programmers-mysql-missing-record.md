---
layout: post
title: MySQL - Missing Record
subtitle : https://programmers.co.kr/learn/courses/30/lessons/59042
tags: [Programmers, MySQL]
author: Huey Kim
comments : False
---



MySQL coding problem from programmers.co.kr
<br>

There is `ANIMAL_INS` table that has structure shown below
<br>

| NAME  | TYPE | NULLABLE |
| :------- | :------ | :------ |
| ANIMAL_ID | VARCHAR(N) | FALSE |
| ANIMAL_TYPE | VARCHAR(N) | FALSE |
| DATETIME | DATETIME | FALSE |
| INTAKE_CONDITION | VARCHAR(N) | FALSE |
| NAME | VARCHAR(N) | TRUE |
| SEX_UPON_INTAKE | VARCHAR(N) | FALSE |

<br>
and `ANIMAL_OUTS` table that has structure shown below.
<br>

| NAME  | TYPE | NULLABLE |
| :------- | :------ | :------ |
| ANIMAL_ID | VARCHAR(N) | FALSE |
| ANIMAL_TYPE | VARCHAR(N) | FALSE |
| DATETIME | DATETIME | FALSE |
| NAME | VARCHAR(N) | TRUE |
| SEX_UPON_OUTCOME | VARCHAR(N) | FALSE |

<br>
The problem states that some data were deleted due to technical problem and requires to find the ID and name of an animal that got adopted, but were never brought into the facility. Animals should be ordered by their ID.
<br>

To start with, we need to find the ID and name of an animal.
<br>

{% highlight html %}
SELECT B.ANIMAL_ID, B.NAME
{% endhighlight %}

<br>
Then we need to specify tables that contain those data.
We need to join two tables because we are comparing `ANIMAL_INS` and `ANIMAL_OUTS` table.
<br>

{% highlight html %}
FROM ANIMAL_OUTS B LEFT JOIN ANIMAL_INS A ON A.ANIMAL_ID = B.ANIMAL_ID
{% endhighlight %}

<br>
Then we need to specify what data we want.
In this case it will be NULL ANIMAL_ID field in `ANIMAL_INS` table.
<br>

{% highlight html %}
WHERE A.ANIMAL_ID IS NULL
{% endhighlight %}

<br>
Finally, we need to order them by their ID.
<br>

{% highlight html %}
ORDER BY ANIMAL_ID
{% endhighlight %}

<br>
which looks like below when all of these statements are written together.
<br>

{% highlight html %}
SELECT 
    B.ANIMAL_ID, 
    B.NAME
FROM ANIMAL_OUTS B LEFT JOIN ANIMAL_INS A ON A.ANIMAL_ID = B.ANIMAL_ID
WHERE A.ANIMAL_ID IS NULL
ORDER BY ANIMAL_ID
{% endhighlight %}
