#!/bin/bash

# 配置参数
repo_url="http://172.16.10.64/xianyang/test.git"
target_dirs=("assets" "models" "test" "tools")

# 获取脚本所在目录
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 颜色定义
RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m' # No Color

# 解压函数
function extract_tarxz() {
    local dir=$1
    echo -e "${YELLOW}正在解压 $dir 中的.tar.xz文件...${NC}"
    
    find "$dir" -name '*.tar.xz' -type f | while read -r file; do
        echo -e "${GREEN}解压: $file${NC}"
        if tar -xJf "$file" -C "$(dirname "$file")"; then
            echo -e "${GREEN}删除压缩包: $file${NC}"
            rm "$file"
        else
            echo -e "${RED}解压失败: $file${NC}"
        fi
    done
}

# 认证函数
function git_authenticate() {
    # 检查是否已保存凭据
    if [ -f "${script_dir}/.git-credentials" ]; then
        git config --global credential.helper "store --file ${script_dir}/.git-credentials"
        return
    fi

    echo -e "${YELLOW}请输入Git仓库访问凭据（公司内网）${NC}"
    read -p "用户名: " username
    read -s -p "密码: " password
    echo ""

    # 存储凭据（仅本次会话有效）
    echo "http://${username}:${password}@172.16.10.64" > "${script_dir}/.git-credentials"
    chmod 600 "${script_dir}/.git-credentials"
    git config --global credential.helper "store --file ${script_dir}/.git-credentials"
}

# 主同步函数
function sync_repo_dirs() {
    local temp_dir=$(mktemp -d)
    echo -e "${GREEN}正在克隆仓库到临时目录...${NC}"
    
    if ! git clone --depth 1 "$repo_url" "$temp_dir"; then
        echo -e "\n${RED}克隆失败！可能原因：${NC}"
        echo "1. 账号密码错误"
        echo "2. 用户没有仓库访问权限"
        echo "3. 网络不可达：172.16.10.64"
        rm -rf "$temp_dir" "${script_dir}/.git-credentials"
        exit 1
    fi

    cd "$temp_dir" || exit

    for dir in "${target_dirs[@]}"; do
        local target_dir="${script_dir}/${dir}" 
        
        if [ ! -d "$dir" ]; then
            echo -e "${YELLOW}警告: 仓库中不存在目录 '$dir'，跳过${NC}"
            continue
        fi

        mkdir -p "$target_dir" || {
            echo -e "${RED}错误: 无法创建目录 '$target_dir'${NC}"
            continue
        }

        echo -e "${GREEN}同步中: $dir → $target_dir${NC}"

        rsync -av --progress \
              --ignore-existing \
              --exclude='.git/' \
              "$dir/" "$target_dir/" || {
            echo -e "${RED}错误: 同步目录 '$dir' 失败${NC}"
        }

        if [ "$dir" == "tools" ]; then
            extract_tarxz "$target_dir"
        fi
    done

    cd "$script_dir" && rm -rf "$temp_dir" "${script_dir}/.git-credentials"
}

# 主流程
git_authenticate
sync_repo_dirs

printf "\033c"
echo -e "${GREEN}所有操作已完成！${NC}"
