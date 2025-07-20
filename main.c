// uHm acTuALY, tHIs CAn bE opTIMIzed -Henry the nerd
// i never met a henry in my life btw

#include <GL/glew.h> //eww
#include <GLFW/glfw3.h> //eww V2
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cglm/cglm.h> // i want to vomit

GLFWwindow* window;
GLuint shaderProgram_mesh;
GLuint shaderProgram_fxaa;
GLuint shaderProgram_heatmap;
GLuint fbo;
GLuint fboTexture;
GLuint vao_mesh;
GLuint vbo_mesh;
GLuint vao_quad;
GLuint vbo_quad;
int g_num_mesh_vertices;

// stuff
void init_glfw_opengl(int width, int height, const char* title);
void create_shaders();
void create_mesh_buffers(int detail);
void create_quad_buffers();
void setup_framebuffer(int width, int height);
void cleanup();

// Mesh utilities
void create_icosahedron(float** verts, unsigned int** faces, int* num_verts, int* num_faces);
void subdivide(float** verts, unsigned int** faces, int* num_verts, int* num_faces);
void make_mesh(int detail, float** pos, float** norm, int* num_elements_in_buffer);

// Shader compilation helper
GLuint compile_shader(const char* source, GLenum type);
GLuint create_program(const char* vert_source, const char* frag_source);

// delete this for better performance -Henry
void run_main_loop(int detail, int instances, int fxaa, int heatmap, int target_fps);
// and this too -Henry again
int main(int argc, char* argv[]) {
    int detail = 3;
    int instances = 9;
    int fxaa = 0;
    int heatmap = 0;
    int target_fps = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--detail") == 0 && (i + 1) < argc) {
            detail = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--instances") == 0 && (i + 1) < argc) {
            instances = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fxaa") == 0) {
            fxaa = 1;
        } else if (strcmp(argv[i], "--heatmap") == 0) {
            heatmap = 1;
        } else if (strcmp(argv[i], "--fps") == 0 && (i + 1) < argc) {
            target_fps = atoi(argv[++i]);
        }
    }

    if (target_fps != 0) {
        printf("The FPS Limit may be inaccurate when using high numbers\n");
    }

    if (fxaa && heatmap) {
        fprintf(stderr, "Error: Choose only one post-process.\n");
        return 1;
    }

    init_glfw_opengl(1280, 720, "GPU Stress");
    create_shaders();
    create_mesh_buffers(detail);
    create_quad_buffers();
    setup_framebuffer(1280, 720);

    run_main_loop(detail, instances, fxaa, heatmap, target_fps);

    cleanup();
    return 0;
}

void init_glfw_opengl(int width, int height, const char* title) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // disables vsync, set to 1 to enable it: glfwSwapInterval(1);

    // Initialize GLEW :(
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(EXIT_FAILURE);
    }

    // Set OpenGL clear color
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glEnable(GL_DEPTH_TEST);
}

// Shader compilation and program creation
GLuint compile_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        fprintf(stderr, "Shader compilation error: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }
    return shader;
}

GLuint create_program(const char* vert_source, const char* frag_source) {
    GLuint vertexShader = compile_shader(vert_source, GL_VERTEX_SHADER);
    GLuint fragmentShader = compile_shader(frag_source, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        fprintf(stderr, "Shader program linking error: %s\n", infoLog);
        exit(EXIT_FAILURE);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

// Whoever created this hates life... wait a seco-
void create_shaders() {
    const char* VERT_SHADER_SOURCE =
    "#version 330\n"
    "in vec3 in_pos;\n"
    "in vec3 in_norm;\n"
    "uniform mat4 Mvp;\n"
    "uniform mat4 Model;\n"
    "out vec3 v_norm;\n"
    "out vec3 v_pos;\n"
    "void main() {\n"
    "    vec4 world = Model * vec4(in_pos, 1.0);\n"
    "    v_pos = world.xyz;\n"
    "    v_norm = mat3(Model) * in_norm;\n"
    "    gl_Position = Mvp * vec4(in_pos, 1.0);\n"
    "}\n";

    const char* FRAG_SHADER_SOURCE =
    "#version 330\n"
    "in vec3 v_norm;\n"
    "in vec3 v_pos;\n"
    "out vec4 f_color;\n"
    "uniform vec3 LightPos;\n"
    "uniform vec3 ViewPos;\n"
    "void main() {\n"
    "    vec3 norm = normalize(v_norm);\n"
    "    vec3 L = normalize(LightPos - v_pos);\n"
    "    vec3 V = normalize(ViewPos - v_pos);\n"
    "    vec3 R = reflect(-L, norm);\n"
    "    float diff = max(dot(norm, L), 0.0);\n"
    "    float spec = pow(max(dot(V, R), 0.0), 32.0);\n"
    "    vec3 base = vec3(0.3,0.7,1.0);\n"
    "    vec3 col = 0.1*base + diff*base + spec*vec3(1.0);\n"
    "    f_color = vec4(col,1.0);\n"
    "}\n";

    const char* FXAA_FRAG_SHADER_SOURCE =
    "#version 330\n"
    "uniform sampler2D tex;\n"
    "in vec2 v_uv;\n"
    "out vec4 fragColor;\n"
    "#define FXAA_REDUCE_MIN   (1.0/128.0)\n"
    "#define FXAA_REDUCE_MUL   (1.0/8.0)\n"
    "#define FXAA_SPAN_MAX     8.0\n"
    "void main(){\n"
    "    vec3 rgbM = texture(tex, v_uv).rgb;\n"
    "    vec2 inv = vec2(textureSize(tex,0));\n"
    "    vec3 luma = vec3(0.299,0.587,0.114);\n"
    "    float l = dot(rgbM, luma);\n"
    "    vec2 uvNW = v_uv + (vec2(-1,-1))*inv;\n"
    "    vec2 uvNE = v_uv + (vec2(1,-1))*inv;\n"
    "    vec2 uvSW = v_uv + (vec2(-1,1))*inv;\n"
    "    vec2 uvSE = v_uv + (vec2(1,1))*inv;\n"
    "    float lNW = dot(texture(tex,uvNW).rgb, luma);\n"
    "    float lNE = dot(texture(tex,uvNE).rgb, luma);\n" //looks clean, but is shit (not just this part, but everything)
    "    float lSW = dot(texture(tex,uvSW).rgb, luma);\n"
    "    float lSE = dot(texture(tex,uvSE).rgb, luma);\n"
    "    float lMin = min(l, min(min(lNW,lNE), min(lSW,lSE)));\n"
    "    float lMax = max(l, max(max(lNW,lNE), max(lSW,lSE)));\n"
    "    vec2 dir = vec2(-((lNW+lNE)-(lSW+lSE)), ((lNW+lSW)-(lNE+lSE)));\n"
    "    float dirReduce = max(\n"
    "        (lNW+lNE+lSW+lSE)*0.25 * FXAA_REDUCE_MUL,\n"
    "        FXAA_REDUCE_MIN\n"
    "    );\n"
    "    float rcpDirMin = 1.0/(min(abs(dir.x),abs(dir.y)) + dirReduce);\n"
    "    dir = clamp(dir*rcpDirMin * FXAA_SPAN_MAX, -FXAA_SPAN_MAX, FXAA_SPAN_MAX)*inv;\n"
    "    vec3 rg1 = texture(tex, v_uv + dir* (1.0/3.0 - 0.5)).rgb;\n"
    "    vec3 rg2 = texture(tex, v_uv + dir* (2.0/3.0 - 0.5)).rgb;\n"
    "    fragColor = vec4((rg1+rg2)/2.0,1.0);\n"
    "}\n";

    const char* HEATMAP_FRAG_SHADER_SOURCE =
    "#version 330\n"
    "uniform sampler2D tex;\n"
    "in vec2 v_uv;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    vec3 col = texture(tex, v_uv).rgb;\n"
    "    float brightness = dot(col, vec3(0.333));\n"
    "    fragColor = vec4(brightness, 0.0, 1.0 - brightness, 0.5);\n"
    "}\n";

    const char* QUAD_VERT_SHADER_SOURCE =
    "#version 330\n"
    "in vec2 in_pos;\n"
    "out vec2 v_uv;\n"
    "void main(){\n"
    "    v_uv = in_pos * 0.5 + 0.5;\n"
    "    gl_Position = vec4(in_pos, 0.0, 1.0);\n"
    "}\n";

    //Hell is over :pray:

    shaderProgram_mesh = create_program(VERT_SHADER_SOURCE, FRAG_SHADER_SOURCE);
    shaderProgram_fxaa = create_program(QUAD_VERT_SHADER_SOURCE, FXAA_FRAG_SHADER_SOURCE);
    shaderProgram_heatmap = create_program(QUAD_VERT_SHADER_SOURCE, HEATMAP_FRAG_SHADER_SOURCE);
}

// Mesh creation
void create_icosahedron(float** verts, unsigned int** faces, int* num_verts, int* num_faces) {
    *num_verts = 12;
    *verts = (float*) malloc(*num_verts * 3 * sizeof(float));
    float t = (1 + sqrtf(5.0f)) / 2.0f; // Calculate 't' here
    float temp_verts[] = {
        -1, t, 0,    1, t, 0,    -1, -t, 0,   1, -t, 0,
        0, -1, t,    0, 1, t,    0, -1, -t,   0, 1, -t,
        t, 0, -1,    t, 0, 1,    -t, 0, -1,   -t, 0, 1,
    };
    // Copy data
    for (int i = 0; i < *num_verts * 3; ++i) {
        (*verts)[i] = temp_verts[i];
    }

    *num_faces = 20;
    *faces = (unsigned int*) malloc(*num_faces * 3 * sizeof(unsigned int));
    unsigned int temp_faces[] = {
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    };
    for (int i = 0; i < *num_faces * 3; ++i) {
        (*faces)[i] = temp_faces[i];
    }
}

void subdivide(float** verts, unsigned int** faces, int* num_verts, int* num_faces) {

    int old_num_verts = *num_verts;
    int old_num_faces = *num_faces;
    float* old_verts = *verts;
    unsigned int* old_faces = *faces;
    *num_faces = old_num_faces * 4;
    *num_verts = old_num_verts + old_num_faces * 3; // Max new midpoints

    float* new_verts = (float*) malloc(*num_verts * 3 * sizeof(float));
    unsigned int* new_faces = (unsigned int*) malloc(*num_faces * 3 * sizeof(unsigned int));

    int current_vert_idx = old_num_verts;

    // Copy old verts
    for (int i = 0; i < old_num_verts * 3; ++i) {
        new_verts[i] = old_verts[i];
    }
    // looks so nice, but also not
    int face_idx = 0;
    for (int i = 0; i < old_num_faces; ++i) {
        int v1_idx = old_faces[i * 3 + 0];
        int v2_idx = old_faces[i * 3 + 1];
        int v3_idx = old_faces[i * 3 + 2];

        float v1[3] = {old_verts[v1_idx*3], old_verts[v1_idx*3+1], old_verts[v1_idx*3+2]};
        float v2[3] = {old_verts[v2_idx*3], old_verts[v2_idx*3+1], old_verts[v2_idx*3+2]};
        float v3[3] = {old_verts[v3_idx*3], old_verts[v3_idx*3+1], old_verts[v3_idx*3+2]};

        float m12[3], m23[3], m31[3];
        glm_vec3_add(v1, v2, m12); glm_vec3_normalize(m12);
        glm_vec3_add(v2, v3, m23); glm_vec3_normalize(m23);
        glm_vec3_add(v3, v1, m31); glm_vec3_normalize(m31);

        // Add new midpoints to new_verts
        int a_idx = current_vert_idx++;
        new_verts[a_idx*3+0] = m12[0]; new_verts[a_idx*3+1] = m12[1]; new_verts[a_idx*3+2] = m12[2];
        int b_idx = current_vert_idx++;
        new_verts[b_idx*3+0] = m23[0]; new_verts[b_idx*3+1] = m23[1]; new_verts[b_idx*3+2] = m23[2];
        int c_idx = current_vert_idx++;
        new_verts[c_idx*3+0] = m31[0]; new_verts[c_idx*3+1] = m31[1]; new_verts[c_idx*3+2] = m31[2];

        // Create new faces
        new_faces[face_idx++] = v1_idx; new_faces[face_idx++] = a_idx; new_faces[face_idx++] = c_idx;
        new_faces[face_idx++] = v2_idx; new_faces[face_idx++] = b_idx; new_faces[face_idx++] = a_idx;
        new_faces[face_idx++] = v3_idx; new_faces[face_idx++] = c_idx; new_faces[face_idx++] = b_idx;
        new_faces[face_idx++] = a_idx;  new_faces[face_idx++] = b_idx; new_faces[face_idx++] = c_idx;
    }

    *num_verts = current_vert_idx;
    *num_faces = face_idx / 3;

    free(old_verts);
    free(old_faces);
    *verts = new_verts;
    *faces = new_faces;
}


void make_mesh(int detail, float** pos, float** norm, int* num_elements_in_buffer) {
    float* current_verts;
    unsigned int* current_faces;
    int num_current_verts;
    int num_current_faces;

    create_icosahedron(&current_verts, &current_faces, &num_current_verts, &num_current_faces);

    for (int i = 0; i < detail; ++i) {
        subdivide(&current_verts, &current_faces, &num_current_verts, &num_current_faces);
    }

    *num_elements_in_buffer = num_current_faces * 3 * 3; // 3 vertices per face, 3 components per vertex
    *pos = (float*) malloc(*num_elements_in_buffer * sizeof(float));
    *norm = (float*) malloc(*num_elements_in_buffer * sizeof(float));

    for (int i = 0; i < num_current_faces; ++i) {
        for (int j = 0; j < 3; ++j) { // For each vertex in the face
            int vert_idx = current_faces[i * 3 + j];
            for (int k = 0; k < 3; ++k) { // For each component (x, y, z)
                (*pos)[(i * 3 + j) * 3 + k] = current_verts[vert_idx * 3 + k];
                // For a sphere, the normal is often the same as the position (normalized)
                (*norm)[(i * 3 + j) * 3 + k] = current_verts[vert_idx * 3 + k];
            }
        }
    }
    g_num_mesh_vertices = num_current_faces * 3;

    free(current_verts);
    free(current_faces);
}


void create_mesh_buffers(int detail) {
    float* pos_data;
    float* norm_data;
    int num_elements_in_buffer_raw;
    make_mesh(detail, &pos_data, &norm_data, &num_elements_in_buffer_raw);

    float* interleaved_data = (float*) malloc(g_num_mesh_vertices * 6 * sizeof(float));
    if (interleaved_data == NULL) {
        fprintf(stderr, "Failed to allocate interleaved_data\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < g_num_mesh_vertices; ++i) {
        for (int j = 0; j < 3; ++j) { // X, Y, Z for position
            interleaved_data[i * 6 + j] = pos_data[i * 3 + j];
        }
        for (int j = 0; j < 3; ++j) { // X, Y, Z for normal
            interleaved_data[i * 6 + 3 + j] = norm_data[i * 3 + j];
        }
    }

    glGenVertexArrays(1, &vao_mesh);
    glBindVertexArray(vao_mesh);

    glGenBuffers(1, &vbo_mesh);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_mesh);
    glBufferData(GL_ARRAY_BUFFER, g_num_mesh_vertices * 6 * sizeof(float), interleaved_data, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(glGetAttribLocation(shaderProgram_mesh, "in_pos"), 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(glGetAttribLocation(shaderProgram_mesh, "in_pos"));
    // Normal attribute
    glVertexAttribPointer(glGetAttribLocation(shaderProgram_mesh, "in_norm"), 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(glGetAttribLocation(shaderProgram_mesh, "in_norm"));

    free(pos_data);
    free(norm_data);
    free(interleaved_data);
}

void create_quad_buffers() {
    float quad_vertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,

        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f
    };

    glGenVertexArrays(1, &vao_quad);
    glBindVertexArray(vao_quad);

    glGenBuffers(1, &vbo_quad);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_quad);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

    GLuint quad_prog_to_use = 0;
    if (shaderProgram_fxaa) {
        quad_prog_to_use = shaderProgram_fxaa;
    } else if (shaderProgram_heatmap) {
        quad_prog_to_use = shaderProgram_heatmap;
    }
    if (quad_prog_to_use) {
        glVertexAttribPointer(glGetAttribLocation(quad_prog_to_use, "in_pos"), 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(glGetAttribLocation(quad_prog_to_use, "in_pos"));
    }
}

void setup_framebuffer(int width, int height) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &fboTexture);
    glBindTexture(GL_TEXTURE_2D, fboTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTexture, 0);

    // Create a renderbuffer object for depth and stencil attachment
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "Framebuffer not complete!\n");
        exit(EXIT_FAILURE);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Bind back to default framebuffer
}

void run_main_loop(int detail, int instances, int fxaa, int heatmap, int target_fps) {
    double last_time = glfwGetTime();
    int frames = 0;
    double min_frame_time = 1.0 / target_fps;

    while (!glfwWindowShouldClose(window)) {
        double frame_start_time = glfwGetTime();

        frames++;
        if (frame_start_time - last_time >= 1.0) {
            char title[100];
            double fps = frames / (frame_start_time - last_time);
            sprintf(title, "Detail=%d, Inst=%d", detail, instances);
            if (fxaa) strcat(title, " | FXAA");
            if (heatmap) strcat(title, " | HEAT");
            char buffer[64];
            if (target_fps > 0) {
                sprintf(buffer, " FPS=%.1f/%d.0", fps, target_fps);
            } else {
                sprintf(buffer, " FPS=%.1f", fps);
            }

            strcat(title, buffer);
            glfwSetWindowTitle(window, title);
            last_time = frame_start_time;
            frames = 0;
        }

        if (fxaa || heatmap) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        } else {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        glViewport(0, 0, 1280, 720);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram_mesh);

        mat4 proj, view, model, mvp;
        glm_perspective(glm_rad(45.0f), (float)1280 / 720, 0.1f, 100.0f, proj);

        vec3 eye = {0.0f, 0.0f, 6.0f};
        vec3 center = {0.0f, 0.0f, 0.0f};
        vec3 up = {0.0f, 1.0f, 0.0f};
        glm_lookat(eye, center, up, view);

        // Get view position from inverse of view matrix
        vec4 view_pos_world = {0.0f, 0.0f, 0.0f, 1.0f};
        mat4 inv_view;
        glm_mat4_inv(view, inv_view);
        glm_mat4_mulv(inv_view, view_pos_world, view_pos_world);

        GLint lightPosLoc = glGetUniformLocation(shaderProgram_mesh, "LightPos");
        glUniform3f(lightPosLoc, 10.0f, 10.0f, 10.0f);
        GLint viewPosLoc = glGetUniformLocation(shaderProgram_mesh, "ViewPos");
        glUniform3f(viewPosLoc, view_pos_world[0], view_pos_world[1], view_pos_world[2]);

        glBindVertexArray(vao_mesh);

        float angle = glfwGetTime() * 0.6f;

        for (int i = 0; i < instances; ++i) {
            float x = (float)(i % 3 - 1) * 3.0f;
            float y = (float)(i / 3 - 1) * 3.0f;

            glm_mat4_identity(model);
            glm_translate(model, (vec3){x, y, 0.0f});
            glm_rotate_y(model, angle + i * 0.2f, model);

            glm_mat4_mul(proj, view, mvp); // proj * view
            glm_mat4_mul(mvp, model, mvp); // (proj * view) * model

            GLint mvpLoc = glGetUniformLocation(shaderProgram_mesh, "Mvp");
            glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, (float*)mvp);
            GLint modelLoc = glGetUniformLocation(shaderProgram_mesh, "Model");
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (float*)model);

            glDrawArrays(GL_TRIANGLES, 0, g_num_mesh_vertices);
        }

        if (fxaa || heatmap) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear default framebuffer
            glViewport(0, 0, 1280, 720);

            if (fxaa) {
                glUseProgram(shaderProgram_fxaa);
            } else { // heatmap
                glUseProgram(shaderProgram_heatmap);
            }

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, fboTexture);
            GLint texLoc = glGetUniformLocation(shaderProgram_fxaa ? shaderProgram_fxaa : shaderProgram_heatmap, "tex");
            glUniform1i(texLoc, 0);

            glBindVertexArray(vao_quad);
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        // FPS limiting
        double frame_end_time = glfwGetTime();
        double frame_duration = frame_end_time - frame_start_time;

        if (frame_duration < min_frame_time) {
            unsigned int sleep_time_us = (unsigned int)((min_frame_time - frame_duration) * 1e6);
            if (sleep_time_us > 0) {
                usleep(sleep_time_us);
            }
        }
    }
}

void cleanup() {
    glDeleteProgram(shaderProgram_mesh);
    glDeleteProgram(shaderProgram_fxaa);
    glDeleteProgram(shaderProgram_heatmap);
    glDeleteVertexArrays(1, &vao_mesh);
    glDeleteBuffers(1, &vbo_mesh);
    glDeleteVertexArrays(1, &vao_quad);
    glDeleteBuffers(1, &vbo_quad);
    glDeleteTextures(1, &fboTexture);
    glDeleteFramebuffers(1, &fbo);
    glfwDestroyWindow(window);
    glfwTerminate();
}
