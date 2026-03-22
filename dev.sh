#!/usr/bin/env bash
# =============================================================================
# dev.sh — Build and Development script for NLP Engine
#
# Usage:
#   ./dev.sh              # Standard build (C++ only)
#   ./dev.sh clean        # Wipe build/ and rebuild
#   ./dev.sh --python     # Build with Python bindings (pybind11)
#   ./dev.sh --fastapi    # Full build (Python + Client) and run FastAPI
#   ./dev.sh --test       # Run C++ tests after build
#   ./dev.sh --asan       # Build with Address/UB Sanitizers (Debug)
#   ./dev.sh --release    # Release build type (default)
#   ./dev.sh --debug      # Debug build type
#   ./dev.sh --client     # Build the React frontend client
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()    { printf "${BLUE}[nlp-dev]${NC} %s\n"    "$1"; }
success() { printf "${GREEN}[nlp-dev]${NC} ✓ %s\n" "$1"; }
warn()    { printf "${YELLOW}[nlp-dev]${NC} ⚠ %s\n" "$1"; }
die()     { printf "${RED}[nlp-dev]${NC} ✗ %s\n"   "$1" >&2; exit 1; }
step()    { printf "\n${CYAN}── %s ${NC}\n" "$1"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="Release"
CLEAN=0
BUILD_PYTHON=0
RUN_TESTS=0
BUILD_CLIENT=0
RUN_FASTAPI=0
PROD_MODE=0
DO_FORMAT=0
USE_ASAN=0
USE_LLVM22=0

# ── Argument parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    clean)        CLEAN=1 ;;
    --python)     BUILD_PYTHON=1 ;;
    --fastapi)    BUILD_PYTHON=1; BUILD_CLIENT=1; RUN_FASTAPI=1; RUN_TESTS=1 ;;
    --prod)       PROD_MODE=1 ;;
    --test)       RUN_TESTS=1 ;;
    --asan)       USE_ASAN=1; BUILD_TYPE="Debug" ;;
    --llvm22)     USE_LLVM22=1 ;;
    --format)     DO_FORMAT=1 ;;
    --client)     BUILD_CLIENT=1 ;;
    --release)    BUILD_TYPE="Release" ;;
    --debug)      BUILD_TYPE="Debug" ;;
    --help|-h)
      sed -n '3,13p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *)  warn "Unknown argument: $arg" ;;
  esac
done

# ── Prerequisites ─────────────────────────────────────────────────────────────
step "Prerequisites"
command -v cmake >/dev/null || die "cmake not found — install from https://cmake.org"
[[ "$DO_FORMAT" -eq 1 ]] && { command -v clang-format >/dev/null || warn "clang-format not found — C++ formatting will be skipped. Install via 'brew install clang-format' or your package manager."; }
[[ "$BUILD_CLIENT" -eq 1 ]] && { command -v bun >/dev/null || die "bun not found (required for --client)"; }
[[ "$RUN_FASTAPI" -eq 1 ]] && { command -v uv >/dev/null || die "uv not found (required for --fastapi)"; }
success "System ready for ${BUILD_TYPE} build"

# ── Clean ─────────────────────────────────────────────────────────────────────
if [[ "$CLEAN" -eq 1 ]]; then
  step "Cleaning"
  rm -rf "$ROOT/build"
  success "build/ directory cleared"
fi

# ── Step 0: Formatting ────────────────────────────────────────────────────────
if [[ "$DO_FORMAT" -eq 1 ]]; then
  step "Formatting Code"

  # 1. C++ Formatting (cpp, hh only)
  if command -v clang-format >/dev/null; then
    info "Formatting C++..."
    # Exclude common dependency directories to prevent formatting third-party code
    find "$ROOT" \( -path "*/.venv" -o -path "*/venv" -o -path "*/build" -o -path "*/node_modules" -o -path "*/_deps" \) -prune \
      -o \( -name "*.cpp" -o -name "*.hh" \) -print0 | xargs -0 clang-format -i
  else
    warn "clang-format not found; skipping C++ formatting."
  fi

  # 2. JS/TS Formatting (Excluding JSON and Markdown)
  if command -v bun >/dev/null; then
    info "Formatting JS/TS files..."
    # Prettier automatically respects .gitignore and .prettierignore
    # Restricted to js, ts, tsx and explicitly excluding json/md
    bun x prettier --write "**/*.{js,ts,tsx}" --no-error-on-unmatched-pattern || true
  else
    warn "bun not found; skipping JS/TS formatting."
  fi

  success "Formatting complete."
fi

# ── Step 1: Frontend (Optional) ───────────────────────────────────────────────
if [[ "$BUILD_CLIENT" -eq 1 ]]; then
  step "Frontend Client"
  cd "$ROOT/examples/fastapi/client"
  info "Installing dependencies..."
  bun install

  if [[ "$PROD_MODE" -eq 1 ]]; then
    info "Building static app (Production)..."
    bun run build.ts --minify --sourcemap=none --define.import.meta.env.NODE_ENV=\"production\"
  else
    info "Building static app (Development)..."
    bun run build.ts --no-minify --sourcemap=linked --define.import.meta.env.NODE_ENV=\"development\"
  fi

  success "Client built in examples/fastapi/client/dist"
  cd "$ROOT"
fi

# ── Step 2: C++ Core & Bindings ───────────────────────────────────────────────
step "Native Build (CMake)"
CMAKE_ARGS=("-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")

# Ensure tests are built when doing full fastapi build or explicit tests
# This ensures CppUTest is fetched and headers are available
if [[ "$RUN_FASTAPI" -eq 1 || "$RUN_TESTS" -eq 1 ]]; then
  info "Enabling tests (BUILD_TESTING=ON)..."
  CMAKE_ARGS+=("-DBUILD_TESTING=ON")
fi

if [[ "$BUILD_PYTHON" -eq 1 ]]; then
  info "Enabling Python bindings..."
  CMAKE_ARGS+=("-DBUILD_PYTHON_BINDINGS=ON")

  # CRITICAL: Ensure we use the Python version managed by uv in the backend
  # This prevents version mismatch (e.g. building for 3.9 but running in 3.10)
  if [[ -d "$ROOT/examples/fastapi/backend" ]]; then
      cd "$ROOT/examples/fastapi/backend"
      UV_PYTHON=$(uv python find 2>/dev/null || echo "")
      cd "$ROOT"
      if [[ -n "$UV_PYTHON" ]]; then
          info "Using uv python for bindings: $UV_PYTHON"
          CMAKE_ARGS+=("-DPYTHON_EXECUTABLE=$UV_PYTHON")
      fi
  fi
else
  CMAKE_ARGS+=("-DBUILD_PYTHON_BINDINGS=OFF")
fi

# Use CMake Presets for local development if available, otherwise fallback to standard args
# This allows local LLVM-22 toolchain via CMakePresets.json while CI uses standard env.
if [[ -f "$ROOT/CMakePresets.json" && -z "${GITHUB_ACTIONS:-}" ]]; then
    if [[ "$USE_ASAN" -eq 1 ]]; then
        if [[ "$USE_LLVM22" -eq 1 ]]; then
            info "Using CMake Preset: llvm22-asan"
            cmake --preset llvm22-asan "${CMAKE_ARGS[@]}"
            info "Compiling with CMake Preset (LLVM 22 ASan)..."
            cmake --build --preset build-llvm22-asan
        else
            info "Using CMake Preset: asan-system"
            cmake --preset asan-system "${CMAKE_ARGS[@]}"
            info "Compiling with CMake Preset (System ASan)..."
            cmake --build --preset build-asan-system
        fi
    elif [[ "$USE_LLVM22" -eq 1 ]]; then
        info "Using CMake Preset: llvm22-custom"
        cmake --preset llvm22-custom "${CMAKE_ARGS[@]}"
        info "Compiling with CMake Preset (LLVM 22)..."
        cmake --build --preset build-llvm22
    else
        info "Using CMake Preset: default"
        cmake --preset default "${CMAKE_ARGS[@]}"
        info "Compiling with CMake Preset (Default)..."
        cmake --build --preset build-default
    fi
else
    info "No preset found or CI environment detected. Using standard configuration."
    mkdir -p "$ROOT/build"
    cd "$ROOT/build"
    cmake .. "${CMAKE_ARGS[@]}"
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    info "Compiling with ${NCPU} cores..."
    cmake --build . --parallel "$NCPU"
fi

# ── Step 3: Tests ─────────────────────────────────────────────────────────────
if [[ "$RUN_TESTS" -eq 1 ]]; then
  step "Testing"
  TEST_FAILED=0

  # Determine test directory based on build method
  if [[ -f "$ROOT/CMakePresets.json" && -z "${GITHUB_ACTIONS:-}" ]]; then
      # Using presets, the binary location depends on the preset configuration.
      # Usually it's in build/llvm22 or similar if specified, but our preset
      # might just put them in a standard place. Let's look in build/
      TEST_DIR="$ROOT/build"
  else
      TEST_DIR="$ROOT/build"
  fi

  # Execute all Catch2 test binaries
  for bin_name in "nlp_tests" "nlp_tests_async" "nlp_tests_markov" "nlp_tests_graph"; do
    # Find the binary recursively in the build directory
    test_bin=$(find "$TEST_DIR" -type f -name "$bin_name" | head -n 1)

    if [[ -n "$test_bin" && -x "$test_bin" ]]; then
      info "Running $(basename "$test_bin")..."
      if ! "$test_bin"; then
        warn "$(basename "$test_bin") failed"
        TEST_FAILED=1
      fi
    else
      warn "Test binary $bin_name not found or not executable in $TEST_DIR"
      TEST_FAILED=1
    fi
  done

  if [[ "$TEST_FAILED" -eq 0 ]]; then
    success "All C++ Tests passed"
  else
    die "Some tests failed"
  fi
fi

# ── Step 4: Environment Setup ─────────────────────────────────────────────────
cd "$ROOT"
step "Environment Setup"
if [[ "$BUILD_PYTHON" -eq 1 ]]; then
  PY_MOD=$(find "$ROOT/build" -name "nlp_engine*.so" -o -name "nlp_engine*.pyd" | head -n 1)
  if [[ -n "$PY_MOD" ]]; then
    MOD_DIR=$(dirname "${PY_MOD}")
    success "Python module found: $(basename "$PY_MOD")"

    # Generate .env file for the backend
    ENV_FILE="$ROOT/examples/fastapi/backend/.env"
    info "Updating $ENV_FILE..."
    # python-dotenv doesn't expand variables like $PYTHONPATH automatically,
    # so we provide the absolute path to the build directory.
    echo "PYTHONPATH=$MOD_DIR" > "$ENV_FILE"
    success "Environment configured for backend."
  else
    die "Python module build failed or not found."
  fi
fi

# ── Step 5: Run FastAPI (Optional) ───────────────────────────────────────────
if [[ "$RUN_FASTAPI" -eq 1 ]]; then
  step "Launching FastAPI"
  cd "$ROOT/examples/fastapi/backend"
  info "Starting server with uv..."
  # Use uv run to ensure dependencies from pyproject.toml are used
  # .env will be loaded by main.py
  exec uv run fastapi dev main.py
fi

# ── Done ──────────────────────────────────────────────────────────────────────
step "Build Complete"
info "Build artifacts in: $ROOT/build"
success "Ready for development."
