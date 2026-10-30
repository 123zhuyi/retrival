IP=80.254.76.225
export https_proxy="http://<user>:<password>@$IP:8080"
export http_proxy="http://<user>:<password>@$IP:8080"
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

IP=80.251.2.161
export https_proxy="http://<user>:<password>@$IP:8080"
export http_proxy="http://<user>:<password>@$IP:8080"
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

IP=80.253.21.126
export https_proxy="http://<user>:<password>@$IP:8888"
export http_proxy="http://<user>:<password>@$IP:8888"
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

IP=80.253.91.240
export http_proxy=http://$IP:7897
export https_proxy=${http_proxy}


# 输入PROXY_IP，PROXY_HW USER_PASSWD 时 不要加{ }
export PROXY_IP=80.254.84.200
export PROXY_HW=172.19.90.131
export USER_PASSWD=<user>:<password>
export https_proxy=http://${USER_PASSWD}@${PROXY_IP}:8080
export http_proxy=http://${USER_PASSWD}@${PROXY_IP}:8080
export ftp_proxy=ftp://${USER_PASSWD}@${PROXY_IP}:8080
知识卡片与triton相关  GPU NPU体系结构


IP=80.253.92.31
# export http_proxy=http://$IP:7897
# export https_proxy=${http_proxy}
export https_proxy="http://<user>:<password>@$IP:6688"
export http_proxy="http://<user>:<password>@$IP:6688"
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

export http_proxy=http://80.254.84.200:
export https_proxy=${http_proxy}

export http_proxy=http://80.254.84.200:7897
export https_proxy=${http_proxy}


IP=80.254.76.225
export http_proxy=http://80.254.76.225:7897
export https_proxy=${http_proxy}
