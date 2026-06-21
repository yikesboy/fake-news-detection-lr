{
  description = "Python ML development shell for fake news detection";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          linuxRuntimeLibs = with pkgs; [
            stdenv.cc.cc.lib
            zlib
          ];
        in
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              [
                python
                uv

                git
                pkg-config

                stdenv.cc.cc.lib
                zlib
                openssl
                libffi
                sqlite
                tk
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
                glibcLocales
              ];

            UV_PYTHON = "${python}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
            PYTHONNOUSERSITE = "1";

            shellHook = ''
              ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
                export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath linuxRuntimeLibs}:''${LD_LIBRARY_PATH:-}"
              ''}

              echo "Python: $(${python}/bin/python --version)"
              echo "uv: $(uv --version)"
              echo "Use 'uv sync' to create/update .venv from pyproject.toml."
            '';
          };
        }
      );
    };
}
