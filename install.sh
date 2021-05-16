#!/usr/bin/env bash

# template from: https://betterdev.blog/minimal-safe-bash-script-template/

#set -Eeuo pipefail
#trap cleanup SIGINT SIGTERM ERR EXIT

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

jversion="1.6"
jversion_minor="1.6.1"
lc=0

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] [-f] -p param_value arg1 [arg2...]

Script description here.

Available options:

-h, --help      Print this help and exit
-v, --verbose   Print script debug info
-f, --flag      Some flag description
-p, --param     Some param description
EOF
  exit
}

cleanup() {
    tput cnorm
    stty echo
  #trap 'catch $? $LINENO' SIGTERM ERR EXIT
  #trap 'aborted' SIGINT 
}

setup_colors() {
    if [[ -t 2 ]] && [[ -z "${NO_COLOR-}" ]] && [[ "${TERM-}" != "dumb" ]]; then
        NOFORMAT='\033[0m' RED='\033[0;31m' GREEN='\033[0;32m' ORANGE='\033[0;33m' BLUE='\033[0;34m' PURPLE='\033[0;35m' CYAN='\033[0;36m' YELLOW='\033[1;33m'
    else
        NOFORMAT='' RED='' GREEN='' ORANGE='' BLUE='' PURPLE='' CYAN='' YELLOW=''
    fi

    success=" [ ${GREEN}\xE2\x9C\x94${NOFORMAT} ]"
    failure=" [ ${RED}\xE2\x9D\x8C${NOFORMAT} ]"
}

function check_dependencies() {
    deps=("wget" "sed" "stty" "tput")
    for p in $deps
    do
        ppath=$(which $p)
        if [ $? -eq 1 ]
        then
            die "${RED}Error:${NOFORMAT} could no find $p" 
        fi
    done
}


function catch() {
    msg "${RED}ERROR: ${NOFORMAT} $1 in line $2"
}

function aborted() {
    msg "${RED}ABORTING${NOFORMAT}"
}

get_input() {
    if [ $ACCEPT -eq 1 ]                     # no default value
    then
        #msg "$1[$2]"
        local res=$2
    else
        read -e -p "$1[$2]: " res
        : ${res:=$2}
    fi
    echo $res
}

confirm() {
    # call with a prompt string or use a default
    read -r -p "${1} [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            true
            ;;
        *)
            false
            ;;
    esac
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}


spinner()
{
    # taken from https://unix.stackexchange.com/questions/225179/display-spinner-while-waiting-for-some-process-to-finish/565551
    local LC_CTYPE=C
    local type=$1
    local pid=$!
    local delay=0.1

    case $((type)) in
    0)
        local spin='⠁⠂⠄⡀⢀⠠⠐⠈'
        local charwidth=3
    ;;
    1)
    local spin='-\|/'
    local charwidth=1
    ;;
    2)
    local spin="▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
    local charwidth=3
    ;;
    3)
    local spin="▉▊▋▌▍▎▏▎▍▌▋▊▉"
    local charwidth=3
    ;;
    4)
    local spin='←↖↑↗→↘↓↙'
    local charwidth=3
    ;;
    5)
    local spin='▖▘▝▗'
    local charwidth=3
    ;;
    6)
    local spin='┤┘┴└├┌┬┐'
    local charwidth=3
    ;;
    7)
    local spin='◢◣◤◥'
    local charwidth=3
    ;;
    8)
    local spin='◰◳◲◱'
    local charwidth=3
    ;;
    9)
    local spin='◴◷◶◵'
    local charwidth=3
    ;;
    10)
    local spin='◐◓◑◒'
    local charwidth=3
    ;;
    11)
    local spin='⣾⣽⣻⢿⡿⣟⣯⣷'
    local charwidth=3
    ;;
    esac

    tput civis
    stty -echo
    while kill -0 $pid 2>/dev/null
    do
        local i=$(((i + $charwidth) % ${#spin}))
        printf "${YELLOW}%s${NOFORMAT}" "${spin:$i:$charwidth}"
        echo -en " ] \033[7D [ "
        sleep $delay
    done
    echo -en "\b\b\b"
    stty echo
    tput cnorm
}

msg() {
    let lc++
    echo >&2 -e "${1-}"
}

status_msg() {
    let lc++
    s1=$(echo -e $1 | sed "s/$(echo -e "\e")[^m]*m//g")
    pad=$(printf '\x4.%.0s' $(seq "$cols") )
    if [ -z $! ]                    # only print status
    then
        s2=$(echo -e $2 | sed "s/$(echo -e "\e")[^m]*m//g")
        echo >&2 -e -n "$1"
        printf '%*.*s' 0 $((cols - ${#s1} - ${#s2} )) "$pad"
        echo >&2 -e "$2"
    else                            # in "waiting for background process" mode
        s2=$(echo -e $success | sed "s/$(echo -e "\e")[^m]*m//g")
        echo >&2 -e -n "$1"
        printf '%*.*s' 0 $((cols - ${#s1} - ${#s2} )) "$pad"
        spinner 9
        echo >&2 -e "$success"
    fi
}



parse_params() {
  # default values of variables set from params
  ACCEPT=0
  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    --no-color) NO_COLOR=1 ;;
    -y | --yes) ACCEPT=1 ;; # example flag
    *) break ;;
    esac
    shift
  done

  args=("$@")

  # check required params and arguments
  # [[ -z "${param-}" ]] && die "Missing required parameter: param"
  # [[ ${#args[@]} -eq 0 ]] && die "Missing script arguments"

  return 0
}

function install_linux() {
    arch_in=$(uname -m)
    [[ $arch_in == *"64" ]] && arch="x64" || arch="x86"
    jpath=$(which julia)
    if [ $? -eq 1 ]
    then
        msg "   (1): Julia installed at [$jpath]\t\t${GREEN}\u2714${NOFORMAT}"
    else
        jpath=$(get_input "Where should Julia be installed " "${HOME}/julia")
        while [ "${jpath:0:1}" != "/" ]; do
            jpath=$(get_input "Where should Julia be installed, please provide an absolute path " "${HOME}/julia")
        done
        mkdir -p $jpath || die
        cd $jpath || die
        fpath="linux/$arch/$jversion/julia-$jversion_minor-linux-$arch_in.tar.gz"

        url=https://julialang-s3.julialang.org/bin/$fpath
        (wget -c $url -O julia.tar.gz -q) & status_msg "Downloading Julia "
        if [ ! -f julia.tar.gz ]
        then
            if [ $? -ne 0 ]; then
              msg "${RED}Error:${NOFORMAT} could not download julia!"
              rm julia.tar.gz
              return
            fi
        fi 
        tar zxf julia.tar.gz -C $jpath --strip-components 1 & status_msg "Installing Julia "
        rm julia.tar.gz
        source $HOME/.bash_profile
        if [[ ":$PATH:" != *":$jpath/bin:"*  ]]
        then
            lastin=$(confirm "Should julia be added to your PATH?")
            if [ lastin ]
            then
                echo "export PATH=$jpath/bin:\$PATH" >>~/.bash_profile
                status_msg "Julia added to PATH" $success
            else
                status_msg "Julia added to PATH" $failure
            fi
        fi
    fi
}

function install_mac() {
    msg "${RED}Error:${NOFORMAT} mac installation not supported yet!"
}

function startup() {
    cols=$(stty size | cut -d ' ' -f 2)
    cols=$(( cols > 120 ? 120 : cols ))
    rows=$(stty size | cut -d ' ' -f 1)
    msg "This script will try to install julia, LadderDGA.jl and all its dependencies."
    msg "============================================================================="
    msg "\n"
    platform="$(uname -s)"
    case "${platform}" in
    Linux*) install_linux ;;
    Darwin*) install_mac ;;
    *)
        echo "Unsupported platform $(platform)" >&2
        exit 1
        ;;
esac
}

parse_params "$@"
setup_colors

# script logic here
check_dependencies
startup

#msg "${RED}Read parameters:${NOFORMAT}"
#msg "- flag: ${flag}"
#msg "- param: ${param}"
#msg "- arguments: ${args[*]-}"
