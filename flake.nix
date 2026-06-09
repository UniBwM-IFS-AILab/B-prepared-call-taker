# {
#   description = "Development shell for ems-prepared";

#   inputs = {
#     nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
#   };

#   outputs = { self, nixpkgs }:
#     let
#       systems = [ "x86_64-linux" "aarch64-linux" ];
#       forAllSystems = nixpkgs.lib.genAttrs systems;
#     in
#     {
#       devShells = forAllSystems (system:
#         let
#           pkgs = import nixpkgs {
#             inherit system;
#             config.allowUnfree = true;
#           };

#           commonPackages = with pkgs; [
#             fd
#             numactl
#             pkg-config
#             stdenv.cc
#             cudaPackages.cuda_nvcc
#             cudaPackages.cudatoolkit
#           ];

#           commonEnv = {
#             CC = "${pkgs.stdenv.cc}/bin/cc";
#             CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
#             CPLUS_INCLUDE_PATH = "${pkgs.cudaPackages.cudatoolkit}/include";
#             CPATH = "${pkgs.cudaPackages.cudatoolkit}/include";
#             TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
#             LD_LIBRARY_PATH =
#               "${pkgs.lib.makeLibraryPath [ pkgs.numactl pkgs.cudaPackages.cudatoolkit ]}:/run/opengl-driver/lib";
#           };

#           miseShellHook = ''
#             eval "$(mise env -s bash)"
#           '';
#         in
#         {
#           default = pkgs.mkShell ({
#             packages = commonPackages;
#             shellHook = miseShellHook;
#           } // commonEnv);
#         });
#     };
# }

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
            config = {
              allowUnfree = true;
            };
          };

          cuda = pkgs.cudaPackages_13_0;

          cudaRuntimeLibs = with cuda; [
            # cuda_cudart
          ];

          runtimeLibraryPath = pkgs.lib.makeLibraryPath (
            cudaRuntimeLibs ++ [
              pkgs.numactl
              pkgs.stdenv.cc.cc.lib
              pkgs.zlib
            ]
          );
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              fd
              mise
              numactl
              pkg-config
              stdenv.cc
            ] ++ cudaRuntimeLibs;

            CC = "${pkgs.stdenv.cc}/bin/cc";

            # CUDA runtime from Nix, NVIDIA driver from the running system.
            LD_LIBRARY_PATH = "${runtimeLibraryPath}:/run/opengl-driver/lib";
            TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";

            # Keep these if you want uv to consistently choose CUDA 13 wheels.
            UV_TORCH_BACKEND = "cu130";

            shellHook = ''
              eval "$(mise env -s bash)"
            '';
          };
        });
    };
}
