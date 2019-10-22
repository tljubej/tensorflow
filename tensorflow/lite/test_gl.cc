#include <thread>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

void createGLcontext() {
    auto rc = eglBindAPI(EGL_OPENGL_ES_API);
    if (!rc) {
        throw std::runtime_error("Nemere bind api");
    }

    auto display = eglGetCurrentDisplay();

    static const EGLint attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};

    auto context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attributes);

    for (auto err = glGetError(); err != GL_NO_ERROR)
}

int main(void) {

}