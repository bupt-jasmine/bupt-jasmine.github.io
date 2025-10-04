#!/bin/bash

# 获取当前日期
DATE=$(date "+%Y-%m-%d")
# 获取文章标题
TITLE=$1
# 获取分类（默认使用 "Uncategorized"）
CATEGORY=$2
# 获取标签（可选）
TAGS=$3

# 创建文件名
FILENAME="${DATE}-$(echo $TITLE | tr ' ' '-').md"

# 创建新的 Markdown 文件
echo "---" > $FILENAME
echo "layout: post" >> $FILENAME
echo "title: \"$TITLE\"" >> $FILENAME
echo "date: \"$DATE 12:00:00 +0000\"" >> $FILENAME
echo "categories: [$CATEGORY]" >> $FILENAME
echo "tags: [$TAGS]" >> $FILENAME
echo "---" >> $FILENAME
echo "" >> $FILENAME
echo "# Enter your content here" >> $FILENAME

echo "Created post: $FILENAME"

