cmake_minimum_required(VERSION 3.12)

project("Hw1"
    VERSION 0.1.0
    LANGUAGES CXX C CUDA
    DESCRIPTION "Hw1"
)

# ==========================================
# 1. 尋找外部依賴套件
# ==========================================
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ==========================================
# 2. 原始碼抓取與分離
# ==========================================
# 抓取 src 下的所有檔案
file(GLOB ALL_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cc"
    "src/*.c"
    "src/*.cu"
)

# 從 ALL_SOURCES 中排除兩個獨立的 main 檔案，剩下的作為共用編譯庫
list(REMOVE_ITEM ALL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main_ga.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main_aco.cpp"
)

# 定義兩個獨立的執行檔
set(EXEC_GA "GA_Optimizer")
set(EXEC_ACO "ACO_Runner")

# 分別將專屬的 main 程式與共用的原始碼綁定
add_executable(${EXEC_GA} src/main_ga.cpp ${ALL_SOURCES})
add_executable(${EXEC_ACO} src/main_aco.cpp ${ALL_SOURCES})

# ==========================================
# 3. 系統環境判定提示 (只印出一次)
# ==========================================
if(WIN32)
    message(STATUS "Detected System: Windows")
elseif(APPLE)
    message(STATUS "Detected System: macOS")
elseif(UNIX AND NOT APPLE)
    message(STATUS "Detected System: Linux")
else()
    message(WARNING "Unknown System")
endif()

# ==========================================
# 4. 統一套用編譯屬性至所有執行檔
# ==========================================
set(MY_EXECUTABLES ${EXEC_GA} ${EXEC_ACO})

foreach(EXEC IN LISTS MY_EXECUTABLES)
    # 系統環境 Macro 綁定
    if(WIN32)
        target_compile_definitions(${EXEC} PRIVATE WINDOWS)
    elseif(APPLE)
        target_compile_definitions(${EXEC} PRIVATE MACOS)
    elseif(UNIX AND NOT APPLE)
        target_compile_definitions(${EXEC} PRIVATE LINUX)
    endif()

    # 強制關閉所有編譯器 Warning
    if(MSVC)
        target_compile_options(${EXEC} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:/W0>
            $<$<COMPILE_LANGUAGE:CXX>:/W0>
        )
    else()
        target_compile_options(${EXEC} PRIVATE
            $<$<COMPILE_LANGUAGE:C>:-w>
            $<$<COMPILE_LANGUAGE:CXX>:-w>
        )
    endif()

    # 針對 CUDA (NVCC) 強制關閉警告
    target_compile_options(${EXEC} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-w>
    )

    # 設定編譯標準
    set_target_properties(${EXEC}
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )

    # 設定 Include 路徑
    target_include_directories(${EXEC} PRIVATE
        "include"
        ${STB_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # 連結函式庫
    target_link_libraries(${EXEC} PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        CUDA::cudart
    )
endforeach()