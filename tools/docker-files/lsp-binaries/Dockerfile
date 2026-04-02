FROM greyltc/archlinux-aur:yay

# Update system and install common language servers
RUN sudo -u ab -D~ bash -c 'yay -Syu --noconfirm && \
    yay -S --noconfirm --removemake --needed --noprogressbar \
    # Python
    basedpyright-bin \
    pyright \
    ruff \
    # TypeScript / JavaScript
    typescript-language-server \
    # Rust
    rust-analyzer \
    # Go
    gopls \
    # C/C++
    clang \
    # Bash
    bash-language-server \
    # Web (HTML, CSS, JSON, ESLint)
    vscode-langservers-extracted \
    # YAML
    yaml-language-server \
    # Docker
    dockerfile-language-server \
    # Clean up cache
    && yay -Scc --noconfirm \
    '
