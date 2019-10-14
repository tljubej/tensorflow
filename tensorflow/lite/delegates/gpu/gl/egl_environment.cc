/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>

#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// TODO(akulik): detect power management event when all contexts are destroyed
// and OpenGL ES is reinitialized. See eglMakeCurrent

Status InitDisplay(EGLDisplay* egl_display) {
  std::cout << "Before eglGetDisplay" << std::endl;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(eglGetDisplay, egl_display, EGL_DEFAULT_DISPLAY));
  std::cout << "After eglGetDisplay" << std::endl;
  if (*egl_display == EGL_NO_DISPLAY) {
    return UnavailableError("eglGetDisplay returned nullptr");
  }
  bool is_initialized;
  std::cout << "Before eGlInitialize" << std::endl;
  RETURN_IF_ERROR(TFLITE_GPU_CALL_EGL(eglInitialize, &is_initialized,
                                      *egl_display, nullptr, nullptr));
  std::cout << "After eGlInitialize" << std::endl;
  if (!is_initialized) {
    return InternalError("No EGL error, but eglInitialize failed");
  }
  return OkStatus();
}

}  // namespace

Status EglEnvironment::NewEglEnvironment(
    std::unique_ptr<EglEnvironment>* egl_environment) {
  *egl_environment = absl::make_unique<EglEnvironment>();
  std::cout << "Before NewEglEnvironment" << std::endl;
  RETURN_IF_ERROR((*egl_environment)->Init());
  return OkStatus();
}

EglEnvironment::~EglEnvironment() {
  if (dummy_framebuffer_ != GL_INVALID_INDEX) {
    glDeleteFramebuffers(1, &dummy_framebuffer_);
  }
  if (dummy_texture_ != GL_INVALID_INDEX) {
    glDeleteTextures(1, &dummy_texture_);
  }
}

Status EglEnvironment::Init() {
  bool is_bound;
  std::cout << "Before eglBindAPI" << std::endl;
  RETURN_IF_ERROR(
      TFLITE_GPU_CALL_EGL(eglBindAPI, &is_bound, EGL_OPENGL_ES_API));
  if (!is_bound) {
    return InternalError("No EGL error, but eglBindAPI failed");
  }

  std::cout << "Before eglGetCurrentContext" << std::endl;
  // Re-use context and display if it was created on this thread.
  if (eglGetCurrentContext() != EGL_NO_CONTEXT) {
    display_ = eglGetCurrentDisplay();
    context_ = EglContext(eglGetCurrentContext(), display_, EGL_NO_CONFIG_KHR);
  } else {
    std::cout << "Before InitDisplay" << std::endl;
    RETURN_IF_ERROR(InitDisplay(&display_));

    std::cout << "Before InitConfiglessContext" << std::endl;
    Status status = InitSurfacelessContext();
    std::cout << "After InitConfiglessContext" << std::endl;
    if (!status.ok()) {
      std::cout << "Before InitSurfacelessContext" << std::endl;
      status = InitSurfacelessContext();
    }
    if (!status.ok()) {
      std::cout << "Before InitPBufferContext" << std::endl;
      status = InitPBufferContext();
    }
    if (!status.ok()) {
      return status;
    }
  }

  if (gpu_info_.type == GpuType::UNKNOWN) {
    RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
  }
  // TODO(akulik): when do we need ForceSyncTurning?
  ForceSyncTurning();
  return OkStatus();
}

Status EglEnvironment::InitConfiglessContext() {
  RETURN_IF_ERROR(CreateConfiglessContext(display_, EGL_NO_CONTEXT, &context_));
  return context_.MakeCurrentSurfaceless();
}

Status EglEnvironment::InitSurfacelessContext() {
  std::cout << "Before CreateSurfacelessContext" << std::endl;
  RETURN_IF_ERROR(
      CreateSurfacelessContext(display_, EGL_NO_CONTEXT, &context_));
  std::cout << "Before MakeCurrentSurfaceless" << std::endl;
  Status status = context_.MakeCurrentSurfaceless();
  if (!status.ok()) {
    return status;
  }

  // PowerVR support EGL_KHR_surfaceless_context, but glFenceSync crashes on
  // PowerVR when it is surface-less.
  RETURN_IF_ERROR(RequestGpuInfo(&gpu_info_));
  if (gpu_info_.type == GpuType::POWERVR) {
    return UnavailableError(
        "Surface-less context is not properly supported on powervr.");
  }
  return OkStatus();
}

Status EglEnvironment::InitPBufferContext() {
  RETURN_IF_ERROR(CreatePBufferContext(display_, EGL_NO_CONTEXT, &context_));
  RETURN_IF_ERROR(CreatePbufferRGBSurface(context_.config(), display_, 1, 1,
                                          &surface_read_));
  RETURN_IF_ERROR(CreatePbufferRGBSurface(context_.config(), display_, 1, 1,
                                          &surface_draw_));
  return context_.MakeCurrent(surface_read_.surface(), surface_draw_.surface());
}

void EglEnvironment::ForceSyncTurning() {
  glGenFramebuffers(1, &dummy_framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, dummy_framebuffer_);

  glGenTextures(1, &dummy_texture_);
  glBindTexture(GL_TEXTURE_2D, dummy_texture_);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         dummy_texture_, 0);

  GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
  glDrawBuffers(1, draw_buffers);

  glViewport(0, 0, 4, 4);
  glClear(GL_COLOR_BUFFER_BIT);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
