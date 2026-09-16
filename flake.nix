{
  description = "Super Based Flake for Text Generation Environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs, ... }:
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [
              python312
              uv
              ruff
              typst
            ];

            # The purpose of shellHook is to fully isolate the environments, so
            # nothing leaves the project, you are very welcome.
            shellHook = /* bash */ ''
              cache_map="
                UV_CACHE_DIR             uv
                HF_HOME                  huggingface
                NLTK_DATA                nltk_data
                PIP_CACHE_DIR            pip
                MPLCONFIGDIR             matplotlib
                TORCH_HOME               torch
                SCIKIT_LEARN_DATA        scikit_learn_data
                TYPST_PACKAGE_CACHE_PATH typst_packages
              "
              args=()
              while read -r var_name folder; do
                if [ -n "$var_name" ]; then
                  target="$PWD/.cache/$folder"
                  mkdir -p "$target"
                  export "$var_name"="$target"
                  args+=("--env" "$var_name" "$target")
                fi
              done <<< "$cache_map"

              if [ ! -d ".venv" ]; then
                uv venv --seed --system-site-packages --python 3.12
              fi

              source .venv/bin/activate

              if [ -f "requirements.txt" ]; then
                uv pip install -r requirements.txt
              fi

              echo "Environment ready, sir."
            '';
          };
        }
      );
    };
}
