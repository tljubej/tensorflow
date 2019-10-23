#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <string>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES3/gl31.h>

static const char gComputeShader[] = 
    "#version 320 es\n"
    "layout(local_size_x = XXYXX) in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    float data[];\n"
    "} input0;\n"
    "layout(binding = 1) writeonly buffer Output {\n"
    "    float data[];\n"
    "} output0;\n"
    "void main()\n"
    "{\n"
    "    uint idx = gl_GlobalInvocationID.x;\n"
    "    output0.data[idx] = input0.data[idx] * 2.0f;\n"
    "}\n";

#define CHECK() \
{\
    GLenum err = glGetError(); \
    if (err != GL_NO_ERROR) \
    {\
        printf("glGetError returns %d\n", err); \
    }\
}

GLuint loadShader(GLenum shaderType, const char* pSource) {
    GLuint shader = glCreateShader(shaderType);
    if (shader) {
        glShaderSource(shader, 1, &pSource, NULL);
        glCompileShader(shader);
        GLint compiled = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        if (!compiled) {
            GLint infoLen = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
            if (infoLen) {
                char* buf = (char*) malloc(infoLen);
                if (buf) {
                    glGetShaderInfoLog(shader, infoLen, NULL, buf);
                    fprintf(stderr, "Could not compile shader %d:\n%s\n",
                            shaderType, buf);
                    free(buf);
                }
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }
    return shader;
}

GLuint createComputeProgram(const char* pComputeSource) {
    GLuint computeShader = loadShader(GL_COMPUTE_SHADER, pComputeSource);
    if (!computeShader) {
        return 0;
    }

    GLuint program = glCreateProgram();
    if (program) {
        glAttachShader(program, computeShader);
        glLinkProgram(program);
        GLint linkStatus = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
        if (linkStatus != GL_TRUE) {
            GLint bufLength = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
            if (bufLength) {
                char* buf = (char*) malloc(bufLength);
                if (buf) {
                    glGetProgramInfoLog(program, bufLength, NULL, buf);
                    fprintf(stderr, "Could not link program:\n%s\n", buf);
                    free(buf);
                }
            }
            glDeleteProgram(program);
            program = 0;
        }
    }
    return program;
}

void setupSSBufferObject(GLuint& ssbo, GLuint index, float* pIn, GLuint count)
{
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);

    glBufferData(GL_SHADER_STORAGE_BUFFER, count * sizeof(float), pIn, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ssbo);
}

void tryComputeShader(size_t compute_size, size_t workgroup_size)
{
    GLuint computeProgram;
    GLuint input0SSbo;
    GLuint input1SSbo;
    GLuint outputSSbo;


    std::string gComputeShader = 
    "#version 320 es\n"
    "layout(local_size_x =" + std::to_string(workgroup_size) + ") in;\n"
    "layout(binding = 0) readonly buffer Input0 {\n"
    "    float data[];\n"
    "} input0;\n"
    "layout(binding = 1) writeonly buffer Output {\n"
    "    float data[];\n"
    "} output0;\n"
    "void main()\n"
    "{\n"
    "    uint idx = gl_GlobalInvocationID.x;\n"
    "    for (int i = 0; i < 50; i++) {\n"
    "        output0.data[idx] += 0.1f;\n"
    "    }\n"
    "}\n";

    CHECK();
    computeProgram = createComputeProgram(gComputeShader.c_str());
    CHECK();

    const GLuint arraySize = compute_size;
    float* f0 = new float[arraySize];
    for (GLuint i = 0; i < arraySize; ++i)
    {
        f0[i] = i;
    }
    setupSSBufferObject(input0SSbo, 0, f0, arraySize);
    setupSSBufferObject(outputSSbo, 1, NULL, arraySize);
    CHECK();

    glUseProgram(computeProgram);

    auto timer = std::chrono::system_clock::now();
    glDispatchCompute(arraySize / workgroup_size,1,1);   // arraySize/local_size_x
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    CHECK();


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, outputSSbo);
    float* pOut = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, arraySize * sizeof(float), GL_MAP_READ_BIT);
    auto timer_end = std::chrono::system_clock::now();
    auto elapsed = timer_end - timer;
    std::cout << std::endl << "Elapsed ms: " << std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000.0 << std::endl;
    for (GLuint i = 0; i < arraySize; ++i)
    {
        if (fabs(pOut[i] - (f0[i]*2.0)) > 0.0001)
        {
            printf("verification FAILED at array index %d, actual: %f, expected: %f\n", i, pOut[i], f0[i]*2.0);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            return;
        }
    }

    delete[] f0;
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    printf("verification PASSED\n");
    glDeleteProgram(computeProgram);
}

int main(int /*argc*/, char** argv)
{
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) {
        printf("eglGetDisplay returned EGL_NO_DISPLAY.\n");
        return 0;
    }

    EGLint majorVersion;
    EGLint minorVersion;
    EGLBoolean returnValue = eglInitialize(dpy, &majorVersion, &minorVersion);
    if (returnValue != EGL_TRUE) {
        printf("eglInitialize failed\n");
        return 0;
    }

    EGLConfig cfg;
    EGLint count;
    EGLint s_configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
            EGL_NONE };
    if (eglChooseConfig(dpy, s_configAttribs, &cfg, 1, &count) == EGL_FALSE) {
        printf("eglChooseConfig failed\n");
        return 0;
    }

    EGLint context_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, context_attribs);
    if (context == EGL_NO_CONTEXT) {
        printf("eglCreateContext failed\n");
        return 0;
    }
    returnValue = eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, context);
    if (returnValue != EGL_TRUE) {
        printf("eglMakeCurrent failed returned %d\n", returnValue);
        return 0;
    }

    tryComputeShader(strtol(argv[1], nullptr, 10), strtol(argv[2], nullptr, 10));

    eglDestroyContext(dpy, context);
    eglTerminate(dpy);

    return 0;
}
