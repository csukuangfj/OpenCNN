#!/bin/bash

# Copyright 2019. All Rights Reserved.
# Author: fangjun.kuang@gmail.com (Fangjun Kuang)

GA_ID=UA-152706513-1
read -d "" ga_str << EOF
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=$GA_ID"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', "$GA_ID");
</script>
EOF

ga_str=$(echo -n $ga_str) # skip new line

doxygen -w html header.html footer.html abc.css

sed -e "s#\(^</body>.*\)#$ga_str\1#" \
		footer.html > footer-ga.html

sed \
		-e "s/^HTML_FOOTER.*=.*/HTML_FOOTER = footer-ga.html/" \
		Doxyfile > Doxyfile-bak

rm -rf ./doxygen

doxygen Doxyfile-bak

rm Doxyfile-bak header.html footer.html abc.css footer-ga.html
