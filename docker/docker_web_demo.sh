#!/usr/bin/env bash
# ↑ Shebang 行：指定使用 bash 解释器执行此脚本
# /usr/bin/env bash 会在系统 PATH 中查找 bash，比直接写 /bin/bash 更具可移植性

#
# This script will automatically pull docker image from DockerHub, and start a daemon container to run the Qwen-Chat web-demo.
# ↑ 脚本功能说明：自动从 DockerHub 拉取镜像，并启动一个后台守护容器来运行 Qwen-Chat 网页演示

# 1> 默认配置变量
# Docker 镜像名称，格式为 "仓库名/镜像名:标签"
# qwenllm/qwenvl 是 DockerHub 上的仓库，qwen3vl-cu128 是镜像标签（表示 CUDA 12.8 版本）
IMAGE_NAME=qwenllm/qwenvl:qwen3vl-cu128

# 模型检查点路径，指向 Qwen3-VL 模型文件所在目录
# 默认值是 HuggingFace 上的模型标识符，实际使用时通常需要改为本地路径
QWEN_CHECKPOINT_PATH='Qwen/Qwen3-VL-235B-A22B-Instruct'

# Web 服务对外暴露的端口号，用户通过 http://localhost:8901 访问
PORT=8901

# Docker 容器的名称，用于后续管理容器（如停止、删除、查看日志等）
CONTAINER_NAME=qwen3vl

# 2> 帮助信息函数
# function 关键字定义一个 shell 函数，usage() 是函数名
# 当用户传入 -h 或 --help 参数时，调用此函数显示使用说明
function usage() {
    # echo 命令输出帮助信息，单引号内的内容会原样输出（不解析变量）
    echo '
Usage: bash docker/docker_web_demo.sh [-i IMAGE_NAME] -c [/path/to/Qwen-Instruct] [-n CONTAINER_NAME] [--port PORT]
'
}

# 3> 命令行参数解析
# while 循环：当 $1（第一个参数）不为空时，持续循环处理参数
# [[ "$1" != "" ]] 是 bash 的条件表达式，检查 $1 是否非空
while [[ "$1" != "" ]]; do
    # case 语句：类似其他语言的 switch-case，根据 $1 的值执行不同分支
    case $1 in
        # -i 或 --image-name 参数：指定 Docker 镜像名称
        -i | --image-name )
            shift          # shift 命令：将参数列表左移一位，$2 变成 $1，$3 变成 $2，以此类推
            IMAGE_NAME=$1  # 此时 $1 是参数的值（原来的 $2），赋值给 IMAGE_NAME
            ;;             # ;; 表示当前 case 分支结束
        
        # -c 或 --checkpoint 参数：指定模型检查点路径
        -c | --checkpoint )
            shift
            QWEN_CHECKPOINT_PATH=$1
            ;;
        
        # -n 或 --container-name 参数：指定容器名称
        -n | --container-name )
            shift
            CONTAINER_NAME=$1
            ;;
        
        # --port 参数：指定 Web 服务端口
        --port )
            shift
            PORT=$1
            ;;
        
        # -h 或 --help 参数：显示帮助信息并退出
        -h | --help )
            usage          # 调用 usage 函数显示帮助
            exit 0         # exit 0 表示正常退出，返回码 0 表示成功
            ;;
        
        # * 通配符：匹配所有其他未识别的参数
        * )
            echo "Unknown argument ${1}"  # 输出错误信息，${1} 会被替换为实际参数值
            exit 1                         # exit 1 表示异常退出，返回码 1 表示失败
            ;;
    esac
    shift  # 处理完当前参数后，再次 shift 移动到下一个参数
done

# 4> 检查点路径验证
# if [ ! -e path ] 检查文件/目录是否存在
# -e 测试文件是否存在，! 取反，所以 ! -e 表示"不存在"
# 这里检查模型配置文件 config.json 是否存在，用于验证路径是否正确
if [ ! -e ${QWEN_CHECKPOINT_PATH}/config.json ]; then
    echo "Checkpoint config.json file not found in ${QWEN_CHECKPOINT_PATH}, exit."
    exit 1  # 文件不存在则退出脚本
fi

# 5> 拉取 Docker 镜像
# sudo：以超级用户权限执行命令（Docker 通常需要 root 权限）
# docker pull：从 DockerHub 下载指定镜像到本地
# || { ... }：逻辑或操作符，如果 docker pull 失败（返回非 0），则执行大括号内的命令
# 这是一种错误处理模式：命令失败时执行备选操作
sudo docker pull ${IMAGE_NAME} || {
    echo "Pulling image ${IMAGE_NAME} failed, exit."
    exit 1
}

# 6> 启动 Docker 容器
# docker run：创建并启动一个新容器
# 各参数说明：
#   --gpus all          让容器可以访问宿主机的所有 GPU（需要 nvidia-docker 支持）
#   -d                  detached 模式，容器在后台运行（守护进程模式）
#   --restart always    容器退出时自动重启，即使 Docker 服务重启后也会自动启动
#   --name              为容器指定一个名称，便于后续管理
#   -v                  挂载卷，将宿主机目录映射到容器内
#                       /var/run/docker.sock 是 Docker 守护进程的 Unix 套接字
#   -p                  端口映射，格式为 "宿主机端口:容器端口"
#                       ${PORT}:80 将宿主机的 PORT 端口映射到容器内的 80 端口
#   --mount             绑定挂载，type=bind 将宿主机目录直接映射到容器
#                       source: 宿主机上的源路径（模型检查点目录）
#                       target: 容器内的目标路径
#   -it                 -i 保持标准输入打开，-t 分配伪终端
#
# 容器启动后执行的命令：python web_demo_mm.py ...
#   --server-port 80    Web 服务监听 80 端口（容器内部端口）
#   --server-name 0.0.0.0  监听所有网络接口，允许外部访问
#   -c                  指定模型检查点路径（容器内路径）
#
# && { ... }：逻辑与操作符，如果 docker run 成功，则执行大括号内的成功提示
sudo docker run --gpus all -d --restart always --name ${CONTAINER_NAME} \
    -v /var/run/docker.sock:/var/run/docker.sock -p ${PORT}:80 \
    --mount type=bind,source=${QWEN_CHECKPOINT_PATH},target=/data/shared/Qwen/checkpoint \
    -it ${IMAGE_NAME} \
    python web_demo_mm.py --server-port 80 --server-name 0.0.0.0 -c /data/shared/Qwen/checkpoint/ && {
    # 成功启动后输出提示信息
    # \` 是转义的反引号，在输出中显示为 `
    # docker logs：查看容器输出日志
    # docker rm -f：强制删除容器（-f 表示 force，即使容器正在运行也会停止并删除）
    echo "Successfully started web demo. Open 'http://localhost:${PORT}' to try!
Run \`docker logs ${CONTAINER_NAME}\` to check demo status.
Run \`docker rm -f ${CONTAINER_NAME}\` to stop and remove the demo."
}