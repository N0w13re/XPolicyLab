#!/usr/bin/env bash
# Environment fixes required to run the RoboDojo Isaac Sim client on A100/A800-class hosts
# whose NVIDIA userspace has been switched to a CUDA forward-compatibility driver.
#
# Source this before launching any RoboDojo eval:
#
#   source scripts/robodojo_sim_env.sh /path/to/RoboDojo-eval
#
# Three independent problems are addressed; each was observed as a hard crash of the
# Isaac Sim client, and all three must be fixed for a single frame to render.
#
# 1. No NVIDIA Vulkan ICD is registered.
#    /usr/share/vulkan/icd.d/ ships only Intel/AMD/lavapipe manifests, so Kit reports
#    "No device could be created" and falls back to no GPU at all. On this driver build
#    the Vulkan entry points live in libEGL_nvidia.so.0; libGLX_nvidia.so.0 (what NVIDIA's
#    own packaging normally points at) does not expose them and enumerates zero devices.
#
# 2. libcuda.so.1 resolves to the CUDA forward-compatibility driver, not the installed one.
#    The kernel module is 535.261.03 but /lib/x86_64-linux-gnu/libcuda.so.1 is symlinked to
#    libcuda.so.590.48.01 from /usr/local/cuda-13.1/compat. Forward compatibility only
#    covers compute; the Vulkan/EGL stack stays at 535. Isaac Sim's renderer shares Vulkan
#    images and semaphores with CUDA, and a 590 CUDA driver cannot import handles produced
#    by a 535 Vulkan driver. The symptom is a cascade starting at
#    "NVTT block-compression failed" / cudaErrorIllegalAddress during the first texture
#    upload, which poisons the CUDA context and makes every later call fail, including
#    vkCreateRayTracingPipelinesKHR. That last error is misleading: it reads like the GPU
#    lacks ray tracing hardware, but A800 does expose VK_KHR_ray_tracing_pipeline and
#    renders fine once the driver mismatch is gone.
#    Only the simulator process is redirected to the stock 535 driver. Policy servers keep
#    the default library path, since they are separate processes and some of them need the
#    newer CUDA runtime. torch cu128 works against 535 through CUDA minor version
#    compatibility.
#
# 3. DLSS/NGX crashes during renderer startup.
#    Kit probes NVSDK_NGX_VULKAN_Init_Ext2 unconditionally; the bundled DLSS 310.1.0 then
#    segfaults inside libnvidia-ptxjitcompiler against this driver. Disabling NGX skips the
#    probe. Anti-aliasing and render mode are deliberately left at their defaults so output
#    stays as close as possible to an official run.
#
# The driver version check is also disabled: Kit rejects 535.261.03 outright, and the
# renderer is verified working on it by scripts/internal/repro_render.py.

set -u

ROBODOJO_ROOT="${1:-${ROBODOJO_ROOT:-}}"
ROBODOJO_SIM_ENV="${2:-${ROBODOJO_SIM_ENV:-RoboDojo}}"
if [[ -z "${ROBODOJO_ROOT}" ]]; then
  echo "[robodojo-env] usage: source scripts/robodojo_sim_env.sh <RoboDojo-eval root> [sim conda env]" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -d "${ROBODOJO_ROOT}" ]]; then
  echo "[robodojo-env] not a directory: ${ROBODOJO_ROOT}" >&2
  return 1 2>/dev/null || exit 1
fi
ROBODOJO_ROOT="$(cd "${ROBODOJO_ROOT}" && pwd)"
export ROBODOJO_ROOT

# --- 1. NVIDIA Vulkan ICD -----------------------------------------------------------------
ROBODOJO_VK_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [[ ! -f "${ROBODOJO_VK_ICD}" ]]; then
  echo "[robodojo-env] installing ${ROBODOJO_VK_ICD}"
  sudo mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
  sudo tee "${ROBODOJO_VK_ICD}" >/dev/null <<'ICD'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0",
        "api_version": "1.3.242"
    }
}
ICD
  sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<'EGL'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
EGL
fi

# --- 2. Stock CUDA driver for the simulator process ---------------------------------------
ROBODOJO_CUDA_NATIVE="${ROBODOJO_ROOT}/.cuda-native"
if [[ ! -e "${ROBODOJO_CUDA_NATIVE}/libcuda.so.1" ]]; then
  stock_libcuda="$(ls /lib/x86_64-linux-gnu/libcuda.so.*.* 2>/dev/null \
    | grep -v -- "-compat" \
    | while read -r lib; do
        # Keep only the build matching the loaded kernel module.
        kmod="$(sed -n 's/^NVRM version:.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null)"
        [[ -n "${kmod}" && "${lib}" == *"${kmod}" ]] && echo "${lib}"
      done | head -1)"
  if [[ -z "${stock_libcuda}" ]]; then
    echo "[robodojo-env] WARNING: no libcuda matching the loaded kernel module; leaving library path alone" >&2
  else
    echo "[robodojo-env] pinning simulator to ${stock_libcuda}"
    mkdir -p "${ROBODOJO_CUDA_NATIVE}"
    ln -sf "${stock_libcuda}" "${ROBODOJO_CUDA_NATIVE}/libcuda.so.1"
    ln -sf libcuda.so.1 "${ROBODOJO_CUDA_NATIVE}/libcuda.so"
  fi
fi
# Scope the override to the simulator conda env rather than exporting it here, so policy
# servers -- separate processes, some of which want the newer CUDA runtime -- are untouched.
# run_sim_env_client.sh reaches the simulator through `conda activate`, which runs activate.d.
if [[ -e "${ROBODOJO_CUDA_NATIVE}/libcuda.so.1" ]]; then
  sim_env_prefix="$(conda run -n "${ROBODOJO_SIM_ENV}" printenv CONDA_PREFIX 2>/dev/null | tail -1)"
  if [[ -z "${sim_env_prefix}" || ! -d "${sim_env_prefix}" ]]; then
    echo "[robodojo-env] WARNING: conda env '${ROBODOJO_SIM_ENV}' not found; not installing activate hook" >&2
  else
    mkdir -p "${sim_env_prefix}/etc/conda/activate.d"
    cat > "${sim_env_prefix}/etc/conda/activate.d/zz-robodojo-native-cuda.sh" <<HOOK
# Installed by XPolicyLab scripts/robodojo_sim_env.sh.
# Isaac Sim shares Vulkan memory with CUDA, so it must use the CUDA driver that matches the
# loaded kernel module rather than the forward-compatibility one on the default search path.
export LD_LIBRARY_PATH="${ROBODOJO_CUDA_NATIVE}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export OMNI_KIT_ACCEPT_EULA=YES
HOOK
    echo "[robodojo-env] installed activate hook in ${sim_env_prefix}"
  fi
fi

# --- 3. Kit settings ----------------------------------------------------------------------
export ROBODOJO_KIT_ARGS="${ROBODOJO_KIT_ARGS:-} --/rtx/verifyDriverVersion/enabled=false --/ngx/enabled=false"
export OMNI_KIT_ACCEPT_EULA=YES

# eval_policy.sh must forward ROBODOJO_KIT_ARGS to the Kit kernel. Upstream hardcodes an
# empty KIT_ARGS, so patch it in place when running against an unpatched checkout.
ROBODOJO_EVAL_SH="${ROBODOJO_ROOT}/scripts/eval_policy.sh"
if [[ -f "${ROBODOJO_EVAL_SH}" ]] && ! grep -q 'ROBODOJO_KIT_ARGS' "${ROBODOJO_EVAL_SH}"; then
  echo "[robodojo-env] patching ${ROBODOJO_EVAL_SH} to honour ROBODOJO_KIT_ARGS"
  sed -i 's/^KIT_ARGS=""$/KIT_ARGS="${ROBODOJO_KIT_ARGS:-}"/' "${ROBODOJO_EVAL_SH}"
fi

echo "[robodojo-env] ROBODOJO_ROOT=${ROBODOJO_ROOT}"
echo "[robodojo-env] ROBODOJO_SIM_ENV=${ROBODOJO_SIM_ENV}"
echo "[robodojo-env] ROBODOJO_KIT_ARGS=${ROBODOJO_KIT_ARGS}"
