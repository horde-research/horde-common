#!/bin/bash

set -e

# Получаем полную версию Google Chrome
chrome_version=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+\.\d+')
echo -e "\n Google Chrome version: $chrome_version\n"

# Ищем ссылку на подходящий chromedriver
driver_url=$(curl -s https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json | \
jq -r --arg ver "$chrome_version" '
  .versions[]
  | select(.version == $ver)
  | .downloads.chromedriver[]
  | select(.platform == "linux64")
  | .url
')

if [ -z "$driver_url" ]; then
    echo "Chromedriver not found for Chrome $chrome_version"
    exit 1
fi

echo "Url for suitable chromedriver: $driver_url"
echo "Downloading..."
curl -s -O "$driver_url"

echo "Unpacking..."
unzip -j chromedriver-linux64.zip "chromedriver-linux64/chromedriver" >/dev/null
chmod +x chromedriver
rm chromedriver-linux64.zip

echo -e "Actual rules:"
ls -l chromedriver
echo ''

target_path=$(which chromedriver || true)

if [ -n "$target_path" ]; then
    if [ -w "$(dirname "$target_path")" ]; then
        echo "Chromedriver updated at $target_path"
        mv -f chromedriver "$target_path"
    else
        echo "Writing in $(dirname "$target_path") is forbidden!"
        echo "please try sudo mv chromedriver $target_path"
    fi
else
    echo "Chromedriver installing in the first time"
    echo "Place chromedriver to /usr/local/bin or in any other on \$PATH:"
    echo "using sudo mv chromedriver /usr/local/bin/"
fi

echo -e "Done!"
