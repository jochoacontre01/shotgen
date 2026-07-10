# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
*i*) ;;
*) return ;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
  debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
xterm-color | *-256color) color_prompt=yes ;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
  if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
    # We have color support; assume it's compliant with Ecma-48
    # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
    # a case would tend to support setf rather than setaf.)
    color_prompt=yes
  else
    color_prompt=
  fi
fi

if [ "$color_prompt" = yes ]; then
  PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
  PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm* | rxvt*)
  PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
  ;;
*)
  ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
  test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
  alias ls='ls --color=auto'
  #alias dir='dir --color=auto'
  #alias vdir='vdir --color=auto'

  alias grep='grep --color=auto'
  alias fgrep='fgrep --color=auto'
  alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
  . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

# cat ~/salute_art.txt
alias c='clear'
alias x='exit'
alias mun='cd "/mnt/3E0C05A50C055973/Documentos/MUN_MSc/100thesisresearchproject/codes/shotgen/" && conda activate msc'
alias sustainlabs='cd "/mnt/3E0C05A50C055973/Documentos/Sustainlabs/Hard copy/3. Codigo y datasets/3.1. Code - Climate Risk - Physical/risk-mod/" && conda activate risk-mod'
alias shortpath='PROMPT_DIRTRIM=1'
alias serialchmod='sudo chmod a+rw /dev/ttyUSB0'
alias opendtect='cd ~/Applications/OpendTect/7.0.0/ && ./start_dtect'
alias garfield='ssh jochoacontre@garfield.cs.mun.ca'
alias globalquake='cd ~/Applications/GlobalQuake-1.1.0/ && ./run.sh'
alias github-store='cd ~/Applications/ && ./GitHub-Store-x86_64.AppImage'
alias world-monitor='cd ~/Applications/ && ./koala73_worldmonitor_world.monitor_2.5.23_amd64.appimage'
alias bat='batcat'

batdiff() {
  git diff --name-only --relative --diff-filter=d -z | xargs -0 batcat --diff
}
export -f batdiff

# Function to display a file on the remote host's GUI
function showfile() {
  # Check if a filename was provided
  if [ -z "$1" ]; then
    echo "Usage: showfile <filename>"
    return 1
  fi

  # Execute the GUI command with all the required environment variables
  sudo -u jochoa \
    DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    DISPLAY=:0 \
    WAYLAND_DISPLAY=wayland-0 \
    gedit "$1" &
}

open-on-termux() {
  if [ -f "$1" ]; then
    # Force a clean path and use absolute paths
    local target_file=$(basename "$1")

    # Run inside a clean subshell that ignores aliases
    (
      export PATH=/usr/bin:/bin:/usr/local/bin
      echo "Moving file..." && /usr/bin/rsync -av --progress "$1" "pixel8pro:~/storage/downloads/" &&
        /usr/bin/ssh pixel8pro "termux-open ~/storage/downloads/$target_file"
    )
  else
    echo "Error: File '$1' not found."
  fi
}

# >>> juliaup initialize >>>

# !! Contents within this block are managed by juliaup !!

case ":$PATH:" in
*:/home/jochoa/.juliaup/bin:*)
  ;;

*)
  export PATH=/home/jochoa/.juliaup/bin${PATH:+:${PATH}}
  ;;
esac

# <<< juliaup initialize <<<

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/jochoa/miniconda3/bin/conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
  eval "$__conda_setup"
else
  if [ -f "/home/jochoa/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/home/jochoa/miniconda3/etc/profile.d/conda.sh"
  else
    export PATH="/home/jochoa/miniconda3/bin:$PATH"
  fi
fi
unset __conda_setup
# <<< conda initialize <<<

# opencode
export PATH=/home/jochoa/.opencode/bin:$PATH
export PATH="$PATH:/opt/nvim/"
export PATH="$HOME/.local/bin:$PATH"

# Added by Antigravity CLI installer
export PATH="/home/jochoa/.local/bin:$PATH"
