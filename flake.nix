{
  # ECS170 course project - Nix dev environment (Python 3 + scikit-learn + PyTorch).
  #
  #   direnv allow     # once; .envrc already contains `use flake`
  #   nix develop      # or enter the shell manually
  #
  # Stage 1 template scripts (cwd matters: the template uses relative paths):
  #   nix run .#stage1-dt | .#stage1-mlp | .#stage1-svm | .#stage1-load
  description = "ECS170 project dev shell: Python, scikit-learn, PyTorch";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));
      # The course template used to live in src/; it now sits in this directory.
      # Every app/devShell path below is built from it.
      template_dir = "ECS170_Fall_2026_Source_Code_Template";

      # Python 3.x + everything the course template imports.
      # `torch` / `scikit-learn` are the cache.nixos.org CPU builds (no CUDA).
      pythonEnv =
        pkgs:
        pkgs.python3.withPackages (
          ps: with ps; [
            numpy
            scikit-learn
            torch
            torchvision
            pandas
            matplotlib
            ipython
            pyyaml
          ]
        );

      # Stage 1 template scripts: run with cwd = <template>/script/stage_1_script
      # and the template root on PYTHONPATH so `import local_code...` resolves.
      # The Stage 1 scripts use relative data paths, hence the cd.
      stage1App =
        pkgs: name: script:
        let
          py = pythonEnv pkgs;
          runner = pkgs.writeShellScriptBin name ''
            set -euo pipefail
            root=$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")
            cd "$root/${template_dir}/script/stage_1_script"
            export PYTHONPATH="$root/${template_dir}''${PYTHONPATH:+:$PYTHONPATH}"
            exec ${py}/bin/python ${script} "$@"
          '';
        in
        {
          type = "app";
          program = "${runner}/bin/${name}";
        };
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          py = pythonEnv pkgs;
        in
        {
          default = pkgs.mkShell {
            packages = [
              py
              pkgs.python3Packages.pip
              pkgs.jq
              pkgs.nix-direnv
              pkgs.git-lfs
            ];

            shellHook = ''
              # Make `import local_code.*` resolve from the repo root.
              export ECS170_TEMPLATE_DIR="$PWD/${template_dir}"
              export PYTHONPATH="$PWD/${template_dir}''${PYTHONPATH:+:$PYTHONPATH}"
              export ECS170_PYTHON="$(command -v python)"
              echo "ECS170 shell: $(python --version 2>&1)"
              echo "  interpreter : $ECS170_PYTHON"
              echo "  template    : $ECS170_TEMPLATE_DIR"
              echo "  check       : python -c 'import sklearn, torch; print(sklearn.__version__, torch.__version__)'"
              echo "  stage 1     : nix run .#stage1-mlp"
              echo "  stage 2     : cd ${template_dir} && python -m script.stage_2_script.script_mlp --validate"
            '';
          };
        }
      );

      apps = forAllSystems (
        pkgs:
        let
          apps = {
            stage1-dt = stage1App pkgs "stage1-dt" "script_decision_tree.py";
            stage1-mlp = stage1App pkgs "stage1-mlp" "script_mlp.py";
            stage1-svm = stage1App pkgs "stage1-svm" "script_svm.py";
            stage1-load = stage1App pkgs "stage1-load" "script_load_result.py";
          };
        in
        apps // { default = apps.stage1-mlp; }
      );

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}
