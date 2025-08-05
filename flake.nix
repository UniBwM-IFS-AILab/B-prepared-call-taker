{
  description = "flake: Python dev shell with uv, PyTorch + CUDA (WSL2 + nix-vscode-server)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nix-gl-host.url = "github:numtide/nix-gl-host";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, nix-gl-host, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            cmake
            ninja
            cudatoolkit
            # cudaPackages.cudart
            cudaPackages.cupti
            cudaPackages.cudnn
            python312
            uv
            nix-gl-host.defaultPackage.${system}
          ];

          shellHook = ''
            echo "Entering devShell with PyTorch + CUDA support"

            # uv-managed venv setup
            uv sync
            . .venv/bin/activate
            echo "UV venv active: \$VIRTUAL_ENV"

            # Setup LD paths for CUDA libs and libcuda stub in WSL
            export LD_LIBRARY_PATH="${nix-gl-host.coreHostDeps}/${nix-gl-host.lib.makeLibraryPath}":$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib":$LD_LIBRARY_PATH
            export LD_LIBRARY_PATH="/usr/lib/wsl/lib":$LD_LIBRARY_PATH

            export CUDA_PATH=${pkgs.cudatoolkit}
            export Torch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
          '';
        };
      }
    );
}
