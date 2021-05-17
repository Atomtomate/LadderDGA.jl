#!/usr/bin/env bash

# template from: https://betterdev.blog/minimal-safe-bash-script-template/

#set -Eeuo pipefail
#trap cleanup SIGINT SIGTERM ERR EXIT

failstate=0
start_dir=$(pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)
jversion="1.6"
jversion_minor="1.6.1"
lc=0
screen_state=""
lDGApath="${HOME}/lDGATools"

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
    if [ $failstate -eq 1 ]
    then
        msg "There have been errors. A log has been written to $lDGApath/install.log"
    fi
    trap - SIGINT SIGTERM ERR EXIT
    tput cnorm
    stty echo
    cd $start_dir
}

setup_colors() {
    if [[ -t 2 ]] && [[ -z "${NO_COLOR-}" ]] && [[ "${TERM-}" != "dumb" ]]; then
        NOFORMAT='\033[0m' RED='\033[0;31m' GREEN='\033[0;32m' ORANGE='\033[0;33m' BLUE='\033[0;34m' PURPLE='\033[0;35m' CYAN='\033[0;36m' YELLOW='\033[1;33m'
    else
        NOFORMAT='' RED='' GREEN='' ORANGE='' BLUE='' PURPLE='' CYAN='' YELLOW=''
    fi

    success=" [ ${GREEN}\xE2\x9C\x94${NOFORMAT} ] "
    failure=". [ ${RED}\xE2\x9D\x8C${NOFORMAT}] "
}

check_dependencies() {
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

redraw() {
    clear
    printf "$screen_state"
}

confirm() {
    # call with a prompt string or use a default
    read -r -p "${1} [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0 ;;
        *)
            return 1 ;;
    esac
}

die() {
  failstate=1
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg, a log has been written to $lDGApath/install.log"
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
        v=$(printf "${YELLOW}%s${NOFORMAT}" "${spin:$i:$charwidth}")
        echo -en "\b\b\b\b\b\b [ $v ]"
        sleep $delay
    done
    echo -en "\b\b\b\b\b\b"
    stty echo
    tput cnorm
    wait $pid
    return $?
}

msg() {
    let lc++
    local tmp=$(echo >&2 -e "${1-}")
    printf "$tmp"
    screen_state="$screen_state$tmp\n"
}

center_msg() {
    let lc++
    s1=$(echo -e $1 | sed "s/$(echo -e "\e")[^m]*m//g")
    local tmp=$(printf '%*s\n' $(((${#s1}+$cols)/2)) "$1\n")
    printf "$tmp"
    screen_state="${screen_state}${tmp}"
}

status_msg() {
    let lc++
    s1=$(echo -e $1 | sed "s/$(echo -e "\e")[^m]*m//g")
    pad=$(printf '%0.1s' "."{1..120})
    s2=$(echo -e $success | sed "s/$(echo -e "\e")[^m]*m//g")
    local tmp=$(printf '%s ' "$1")
    printf "$tmp"
    screen_state="$screen_state$tmp"
    if [ -z $! ]                    # only print status
    then
        tmp=$(printf '%*.*s' 0 $((cols - ${#s1} - 12 )) "$pad")
        printf "$tmp"
        local state=$2
    else                            # in "waiting for background process" mode
        tmp=$(printf '%*.*s' 0 $((cols - ${#s1} - 6 )) "$pad")
        printf "$tmp"
        spinner 9
        local state=$?
    fi
    screen_state="${screen_state}${tmp}"
    if [ $state -eq 0 ]
    then
        tmp=$(echo >&2 -e "${success}")
        screen_state="$screen_state$success\n"
    else
        tmp=$(echo >&2 -e "${falure}")
        screen_state="$screen_state$failure\n"
        failstate=1
    fi
    printf "${tmp}"
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

install_linux() {
    arch_in=$(uname -m)
    [[ $arch_in == *"64" ]] && arch="x64" || arch="x86"
    jpath=$(which julia)
    if [ $? -eq 0 ]
    then
        status_msg "Julia installed " 0
    else
        jpath=$(get_input "Where should Julia be installed " "${HOME}/julia")
        while [ "${jpath:0:1}" != "/" ]; do
            jpath=$(get_input "Where should Julia be installed, please provide an absolute path " "${HOME}/julia")
        done
        redraw
        mkdir -p $jpath || die "Could no create $jpath"
        cd $jpath || die "Could no find $jpath"
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
        source $HOME/.bashrc
        if [[ ":$PATH:" != *":$jpath/bin:"*  ]]
        then
            if confirm "Should julia be added to your PATH?"
            then
                echo "export PATH=$jpath/bin:\$PATH" >>~/.bashrc
                status_msg "Julia added to PATH " 0
            else
                status_msg "Julia added to PATH " 1
            fi
        else
                status_msg "Julia added to PATH " 0
        fi
    fi
}

install_mac() {
    msg "${RED}Error:${NOFORMAT} mac installation not supported yet!"
}

git_clone(){
    for p in "EquivalenceClassesConstructor.jl" "Dispersions.jl" "SparseVertex" "LadderDGA.jl"
    do
        if [ ! -d $p ]
        then
           git clone "https://github.com/Atomtomate/$p" >> "$lDGApath/install.log" 2>&1 || die "Could not clone $p"
        fi
    done
}

setup_julia_deps() {
    julia -e 'using Pkg; Pkg.add.(["IJulia", "Plots", "HDF5", "JLD2", "Printf", "DataStructures"])' >> "$lDGApath/install.log" 2>&1 & status_msg "Updating Julia "
    local cw=$(pwd)
    cd $lDGApath
    for p in "EquivalenceClassesConstructor.jl" "Dispersions.jl" "SparseVertex" "LadderDGA.jl"
    do
        cd $p
        local cwd=$(pwd)
        git pull --quiet
        julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate();' >> "$lDGApath/install.log" 2>&1 & status_msg "Updating ${p-} "
        julia -e "using Pkg; Pkg.add(path=\"$cwd\")" >> "$lDGApath/install.log" 2>&1 & status_msg "Precompiling ${p-} "
        cd ..
    done
    cd $cw
}

install_lDGA() {
    lDGApath=$(get_input "Where should the ladder DGA tools be installed " "${HOME}/lDGATools")
    while [ "${jpath:0:1}" != "/" ]; do
        lDGApath=$(get_input "Where should the ladder DGA tools be installed " "${HOME}/lDGATools")
    done
    redraw
    mkdir -p $lDGApath || die "Could no create $lDGApath"
    cd $lDGApath || die "Could no find $lDGApath"
    > "install.log"
    git_clone & status_msg "Cloning lDGA tools "
}

start_example() {
    if confirm "LadderDGA.jl installed and updated. Would you like to start the tutorial?"
    then
        echo "starting Jupyter-Notebook. Press ctrl+c to exit."
#        cd "$lDGApath/LadderDGA.jl/notebooks"
 #       echo `pwd`
        local dir="$lDGApath/LadderDGA.jl/notebooks"
        julia -e "using IJulia; notebook(dir=\"$dir\")"
    else
        echo "all done, exiting."
    fi
}

startup() {
    cols=$(tput cols)
    cols=$(( cols > 120 ? 120 : cols ))
    rows=$(stty size | cut -d ' ' -f 1)
    printf "\033c"
    center_msg "============================================================================="
    center_msg "This script will try to install julia, LadderDGA.jl and all its dependencies."
    center_msg "If LadderDGA.jl is already installed, this script will try to"
    center_msg "update the packages instead."
    center_msg "All outputs of the LadderDGA.jl installation can be found in the install.log"
    center_msg "file, inside the install directory"
    center_msg "============================================================================="
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
    install_lDGA
    setup_julia_deps
    cleanup
    start_example
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
