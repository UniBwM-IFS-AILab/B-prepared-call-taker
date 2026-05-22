{
  description = "Development shell for ems-prepared";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          commonPackages = with pkgs; [
            fd
            numactl
            pkg-config
            stdenv.cc
            cudaPackages.cuda_nvcc
            cudaPackages.cudatoolkit
          ];

          commonEnv = {
            CC = "${pkgs.stdenv.cc}/bin/cc";
            CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
            CPLUS_INCLUDE_PATH = "${pkgs.cudaPackages.cudatoolkit}/include";
            CPATH = "${pkgs.cudaPackages.cudatoolkit}/include";
            TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
            LD_LIBRARY_PATH =
              "${pkgs.lib.makeLibraryPath [ pkgs.numactl pkgs.cudaPackages.cudatoolkit ]}:/run/opengl-driver/lib";
          };

          miseShellHook = ''
            eval "$(mise env -s bash)"
          '';
        in
        {
          default = pkgs.mkShell ({
            packages = commonPackages;
            shellHook = miseShellHook;
          } // commonEnv);
        });
    };
}
