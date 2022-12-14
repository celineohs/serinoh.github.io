---
layout: page
permalink: /article/
title: article
description: This is a page for the final article.
nav: false
---
<!-- _pages/article.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>

---
